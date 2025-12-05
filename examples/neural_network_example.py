from mpi4py import MPI

import basix.ufl
import dolfinx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import ufl
from dolfinx_external_operator import FEMExternalOperator, functionspace
from jax.flatten_util import ravel_pytree

from fenicsx_jax.fem import (
    compile_external_operator_form,
    create_real_functionspace,
    pack_external_operator_data,
)

jax.config.update("jax_enable_x64", True)


import equinox as eqx

from mms import apply_mms, get_BC_function

key = jax.random.PRNGKey(42)

############################ PROBLEM ####################################

gdim = 2
N = 25
D = 3
NMC = 1000

if gdim == 1:
    uex_str = "sin(10*pi*x[0])*x[0]"
    uex_str = "sin(2*pi*x[0])"
    xtest = np.linspace(0, 1, 10000).reshape((1, -1))
elif gdim == 2:
    xx = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(xx, xx)
    xtest = np.vstack([X.flatten(), Y.flatten()])

    uex_str = "sin(10*pi*x[0])*x[0]*x[1]*sin(4*pi*x[1]) + exp(-4*((x[0]-0.5)**2 + (x[1]-0.5)**2))*16*x[0]*(1-x[0])*x[1]*(1-x[1])"
else:
    raise NotImplementedError

problem_data = apply_mms(uex_str)
u_ex = eval(problem_data["u_ex"])
f = eval(problem_data["f"])

bc_func = get_BC_function(gdim)

xtrain = jnp.array(np.random.randn(gdim, NMC))

############################ MODEL AND LOSS FUNCTION ####################################


class NeuralNetwork(eqx.Module):
    layers: list

    def __init__(self, key, N, D):
        if D < 2:
            raise NotImplementedError
        nn_dimensions = [[gdim, N]] + [[N, N] for i in range(D - 2)] + [[N, 1]]
        n_layers = len(nn_dimensions)
        keys = jax.random.split(key, n_layers)
        self.layers = [
            eqx.nn.Linear(nn_dimensions[i][0], nn_dimensions[i][1], key=keys[i])
            for i in range(n_layers - 1)
        ]
        self.layers.append(
            eqx.nn.Linear(nn_dimensions[-1][0], "scalar", use_bias=False, key=keys[-1])
        )
        self.init_linear_weight(jax.nn.initializers.glorot_uniform, key)

    def init_linear_weight(self, init_fn, key):
        get_weights = lambda m: [
            x.weight
            for x in jax.tree_util.tree_leaves(m)
            if isinstance(x, eqx.nn.Linear)
        ]
        weights = get_weights(self)
        new_weights = [
            init_fn(subkey, weight.shape)
            for weight, subkey in zip(weights, jax.random.split(key, len(weights)))
        ]
        self = eqx.tree_at(get_weights, self, new_weights)

    def __call__(self, x):
        phi = bc_func(x)
        for layer in self.layers[:-1]:
            x = jax.nn.sigmoid(layer(x))
        return (self.layers[-1](x) * phi).squeeze()

    def u(self, x):
        return jax.vmap(self, in_axes=1, out_axes=0)(x).squeeze()

    def du(self, x):
        return jax.vmap(jax.jacfwd(self), in_axes=1, out_axes=0)(x).T

    def hess(self, x):
        return jax.vmap(jax.jacfwd(jax.jacfwd(self)), in_axes=1, out_axes=0)(x).T

    def split_eval(self, *x):
        return self(jnp.array([*x]))

    def laplacian_scalar(self, x):
        return sum(
            jax.jacfwd(jax.jacfwd(self.split_eval, i), i)(*x) for i in range(gdim)
        )

    def laplacian(self, x):
        return sum(
            jax.vmap(jax.jacfwd(jax.jacfwd(self.split_eval, i), i))(*x)
            for i in range(gdim)
        )

    @eqx.filter_jit
    def residual(self, x):
        return -self.laplacian(x) - f(x)

    def residual_scalar(self, x):
        return -self.laplacian_scalar(x) - f(x)

    # computes the gradient of the residual wrt theta
    # NOTE: the output is a pytree!
    @eqx.filter_grad
    def residual_dtheta_scalar(self, x):
        return self.residual_scalar(x)

    # vectorised gradient (wrt theta) computation (OPTION 1)
    # NOTE: the output is a single pytree with 3D tensor nodes!
    def residual_dtheta_pytree(self, x):
        return jax.vmap(self.residual_dtheta_scalar, in_axes=1)(x)

    # vectorised gradient (wrt theta) computation (OPTION 2)
    # NOTE: the output is now a 2D array of shape (n_x_coordinates, len(theta))
    @eqx.filter_jit
    def residual_dtheta(self, x):
        flattened_gradient_dtheta = lambda x: ravel_pytree(
            self.residual_dtheta_scalar(x)
        )[0]
        return jax.vmap(flattened_gradient_dtheta, in_axes=1)(x)


model = NeuralNetwork(key, N, D)

# theta_values is a 1D jax.numpy.array containing all NN weights and biases
# Can reconstruct the NN from the theta_values by using unravel_fun
theta_values, unravel_fun = ravel_pytree(model)
original_model = unravel_fun(theta_values)

dNN_dtheta_pytree = model.residual_dtheta_pytree(xtrain)
dNN_dtheta = model.residual_dtheta(xtrain)

# example of recovering pytree from d_NN/d_theta at x_0
i = 0  # first coordinate of xtrain
dNN_dtheta_xi = unravel_fun(dNN_dtheta[i])  # pytree with same structure as model

############################ define external operator ###########################


def evaluate_NN(x, theta):
    x = x.reshape((gdim, -1))
    tt = theta[0, 0, :].flatten()
    return np.array(unravel_fun(tt).residual(x).flatten())


def evaluate_dNN_dtheta(x, theta):
    x = x.reshape((gdim, -1))
    tt = theta[0, 0, :].flatten()
    return np.array(unravel_fun(tt).residual_dtheta(x).flatten())


def u_NN_jax(gdim, derivatives):
    if derivatives == (0, 0):
        return evaluate_NN
    elif derivatives == (1, 0):
        raise NotImplementedError(f"No function is defined for the {derivatives=}.")
    elif derivatives == (0, 1):
        return evaluate_dNN_dtheta
    else:
        raise NotImplementedError(f"No function is defined for the {derivatives=}.")


############################ Prepare JAX wrapper for the FEniCSx functional ###########################


# NOTE: JAX does not support providing custom forward and backward diff rules at the same time so
#       need to write two wrappers. The one that is compatible with jax.grad is J_jax_vjp.
# NOTE: The only way to wrap functions into JAX so that autodiff is supported is to use jax.pure_callback
#       which assumes that the functions are pure, i.e., that they have no side-effects. This is of course
#       an assumption that does not hold here. The alternative is to do derivatives by hand.
def jax_wrapper(J, dJdtheta):
    @jax.custom_jvp
    def J_jax_jvp(theta):
        return jax.pure_callback(J, jax.ShapeDtypeStruct((), theta.dtype), theta)

    @J_jax_jvp.defjvp
    def my_func_jvp(primals, tangents):
        (theta,) = primals
        (dtheta,) = tangents
        Jval = J_jax_jvp(theta)
        grad_theta = jax.pure_callback(
            dJdtheta, jax.ShapeDtypeStruct(theta.shape, theta.dtype), theta
        )
        return Jval, jnp.dot(grad_theta, dtheta)  # forward JVP

    @jax.custom_vjp
    def J_jax_vjp(theta):
        return jax.pure_callback(J, jax.ShapeDtypeStruct((), theta.dtype), theta)

    def J_jax_vjp_fwd(theta):
        Jval = J_jax_vjp(theta)
        residual = theta
        return Jval, residual  # Residual for backward

    def dJdtheta_jax_bwd(residual, cotangent):
        theta = residual
        grad_theta = jax.pure_callback(
            dJdtheta, jax.ShapeDtypeStruct(theta.shape, theta.dtype), theta
        )
        return (cotangent * grad_theta,)  # J^T @ cotangent

    J_jax_vjp.defvjp(J_jax_vjp_fwd, dJdtheta_jax_bwd)

    return J_jax_jvp, J_jax_vjp


############################ pytest routine #####################################


def test_jax_wrapper(theta_values, eval_J, eval_dJdtheta):
    Jh = eval_J(theta_values)
    vec_dJdn = eval_dJdtheta(theta_values)

    J_jax_jvp, J_jax_vjp = jax_wrapper(eval_J, eval_dJdtheta)

    # BWD diff test
    g = lambda theta_values: jnp.sin(J_jax_vjp(theta_values))
    gval = g(theta_values)

    gjit = jax.jit(g)
    gjitval = gjit(theta_values)

    dgdtheta = jax.jit(jax.grad(g))
    dgdthetaval = dgdtheta(theta_values)

    ref_g = np.cos(Jh) * vec_dJdn

    assert np.allclose(dgdthetaval, ref_g)
    assert np.allclose(gjitval, np.sin(Jh))

    # FWD diff test
    h = lambda theta_values: jnp.sin(J_jax_jvp(theta_values))
    hval = h(theta_values)

    hjit = jax.jit(h)
    hjitval = hjit(theta_values)

    hjvp = jax.jit(lambda theta_values: jax.jvp(h, (theta_values,), (theta_values,))[1])
    hjvp_val = hjvp(theta_values)

    ref_h = np.dot(ref_g, theta_values)

    assert np.allclose(hjvp_val, ref_h)
    assert np.allclose(hjitval, np.sin(Jh))


def test_neural_network(cell_type, q_deg, N):
    tdim = dolfinx.cpp.mesh.cell_dim(cell_type)
    assert tdim == gdim
    if tdim == 1:
        mesh = dolfinx.mesh.create_unit_interval(MPI.COMM_WORLD, N)
    if tdim == 2:
        mesh = dolfinx.mesh.create_unit_square(
            MPI.COMM_WORLD, N, N, cell_type=cell_type
        )
    elif tdim == 3:
        mesh = dolfinx.mesh.create_unit_cube(
            MPI.COMM_WORLD, N, N, N, cell_type=cell_type
        )

    ntheta = len(theta_values)
    R = create_real_functionspace(mesh, value_shape=(ntheta,))
    theta = dolfinx.fem.Function(R)
    theta.x.array[:] = theta_values
    theta.x.scatter_forward()
    x = ufl.SpatialCoordinate(mesh)

    Qe = basix.ufl.quadrature_element(mesh.basix_cell(), degree=q_deg)
    Q = functionspace(mesh, Qe)
    alpha = dolfinx.fem.Constant(mesh, 1.0)

    # Define external operator and correct quadrature space
    pytest.importorskip("jax")
    N = FEMExternalOperator(
        x,
        theta,
        function_space=Q,
        external_function=lambda derivatives: u_NN_jax(mesh.geometry.dim, derivatives),
        name="exop",
    )

    V = functionspace(mesh, ("Lagrange", 1))
    phi = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": q_deg})

    def F(N, phi_h):
        a = ufl.inner(ufl.grad(phi), ufl.grad(v)) * dx
        L = N * v * dx
        _F = a - L
        return ufl.action(_F, phi_h)

    def compute_dFdn(F, theta, lmbda):
        dFdtheta = ufl.derivative(F, theta)
        form = ufl.adjoint(dFdtheta)
        return ufl.action(form, lmbda)

    def compute_J(N, phih):
        return (
            ufl.inner(ufl.grad(phih), ufl.grad(phih)) * dx
            + alpha * ufl.inner(N, N) * dx
        )

    def compute_dJdn(J, theta):
        dJdtheta = ufl.derivative(J, theta)
        return dJdtheta

    lmbda = dolfinx.fem.Function(V, name="lmbda")
    phih = dolfinx.fem.Function(V, name="phih")
    phih.interpolate(lambda x: np.sin(np.pi * x[0]))
    lmbda.interpolate(lambda x: np.cos(3 * np.pi * x[0]))

    J_ex_op = compute_J(N, phih)
    dJdn_ex = compute_dJdn(J_ex_op, theta)
    J_compiled = compile_external_operator_form(J_ex_op)
    dJdn_compiled = compile_external_operator_form(dJdn_ex)

    # Define J as a function of theta_values
    def eval_J(theta_values: np.ndarray) -> float:
        theta.x.array[:] = theta_values
        theta.x.scatter_forward()
        pack_external_operator_data(J_compiled)
        Jh_loc = dolfinx.fem.assemble_scalar(J_compiled)
        Jh = mesh.comm.allreduce(Jh_loc, op=MPI.SUM)
        return Jh

    # Define J as a function of theta_values
    def eval_dJdtheta(theta_values: np.ndarray) -> np.ndarray:
        theta.x.array[:] = theta_values
        theta.x.scatter_forward()
        pack_external_operator_data(dJdn_compiled)
        vec_dJdn = dolfinx.fem.assemble_vector(dJdn_compiled)
        return vec_dJdn.array.copy()

    test_jax_wrapper(theta_values, eval_J, eval_dJdtheta)

    Jh = eval_J(theta_values)
    vec_dJdn = eval_dJdtheta(theta_values)

    F_ex = F(N, phih)
    F_compiled = compile_external_operator_form(F_ex)
    pack_external_operator_data(F_compiled)
    vec = dolfinx.fem.assemble_vector(F_compiled)

    dFdn_ex = compute_dFdn(F_ex, theta, lmbda)
    dFdn_compiled = compile_external_operator_form(dFdn_ex)
    pack_external_operator_data(dFdn_compiled)
    vec_dFdn = dolfinx.fem.assemble_vector(dFdn_compiled).array.copy()

    return Jh, vec, vec_dFdn, vec_dJdn


if __name__ == "__main__":
    cell_type = dolfinx.mesh.CellType.triangle
    q_deg = 4
    N = 8

    out = test_neural_network(cell_type, q_deg, N)
