from functools import partial

from mpi4py import MPI

import basix.ufl
import dolfinx
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import ufl
from dolfinx_external_operator import FEMExternalOperator, functionspace
from jax.flatten_util import ravel_pytree

from fenicsx_jax.fem import (
    LinearProblem,
    compile_external_operator_form,
    create_real_functionspace,
    pack_external_operator_data,
)
from mms import apply_mms, get_BC_function

jax.config.update("jax_enable_x64", True)


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

    uex_str = (
        "sin(10*pi*x[0])*x[0]*x[1]*sin(4*pi*x[1]) +"
        + " exp(-4*((x[0]-0.5)**2 + (x[1]-0.5)**2))*16*x[0]*(1-x[0])*x[1]*(1-x[1])"
    )
else:
    raise NotImplementedError

problem_data = apply_mms(uex_str)
u_ex = eval(problem_data["u_ex"])
f = eval(problem_data["f"])

bc_func = get_BC_function(gdim)

xtrain = jnp.array(np.random.randn(gdim, NMC))

# ## MODEL AND LOSS FUNCTION


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
        def get_weights(m):
            return [
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

    @eqx.filter_jit
    def u(self, x):
        return jax.vmap(self, in_axes=1, out_axes=0)(x).squeeze()

    @eqx.filter_jit
    def du(self, x):
        return jax.vmap(jax.jacfwd(self), in_axes=1, out_axes=0)(x).T

    @eqx.filter_jit
    def dudtheta(self, x):
        def scalar_dudtheta(x):
            out = ravel_pytree(eqx.filter_grad(lambda network, x: network(x))(self, x))
            return out[0]

        return jax.vmap(scalar_dudtheta, in_axes=1, out_axes=0)(x)

    @eqx.filter_jit
    def d2udxdtheta(self, x):
        def scalar_dudtheta(x):
            return eqx.filter_jacrev(lambda network, x: jax.jacfwd(network)(x))(self, x)

        def ravel_and_stack(x):
            pytree_out = scalar_dudtheta(x)
            values = jnp.stack(
                [
                    ravel_pytree(jax.tree.map(lambda x: x[i], pytree_out))[0]
                    for i in range(gdim)
                ]
            )
            return values

        return jax.vmap(ravel_and_stack, in_axes=1, out_axes=1)(x)


model = NeuralNetwork(key, N, D)

# theta_values is a 1D jax.numpy.array containing all NN weights and biases
# Can reconstruct the NN from the theta_values by using unravel_fun
theta_values, unravel_fun = ravel_pytree(model)
original_model = unravel_fun(theta_values)

############################ define external operator ###########################


# FIXME: need to ask about output shape orderings
def u_NN_jax(x, theta, derivatives=None):
    x = x.reshape((gdim, -1))
    tt = theta[0, 0, :].flatten()

    network = unravel_fun(tt)

    if derivatives == (0, 0):
        return np.asarray(network.u(x)).flatten()
    elif derivatives == (1, 0):
        return np.asarray(network.du(x)).flatten()
    elif derivatives == (0, 1):
        return np.asarray(network.dudtheta(x)).flatten()
    elif derivatives == (1, 1):
        return np.asarray(network.d2udxdtheta(x)).flatten()
    else:
        raise NotImplementedError(f"No function is defined for the {derivatives=}.")


# ## Prepare JAX wrapper for the FEniCSx functional


# NOTE: JAX does not support providing custom forward and
#     backward diff rules at the same time so need to write two wrappers.
#     The one that is compatible with jax.grad is J_jax_vjp.
# NOTE: The only way to wrap functions into JAX so that autodiff is supported
#       is to use jax.pure_callback which assumes that the functions are pure,
#       i.e., that they have no side-effects. This is of course an assumption that
#       does not hold here. The alternative is to do derivatives by hand.
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


############################ test routine #####################################


def test_jax_wrapper(theta_values, eval_J, eval_dJdtheta):
    Jh = eval_J(theta_values)
    vec_dJdn = eval_dJdtheta(theta_values)

    J_jax_jvp, J_jax_vjp = jax_wrapper(eval_J, eval_dJdtheta)

    # BWD diff test
    def g(theta_values):
        return jnp.sin(J_jax_vjp(theta_values))

    # gval = g(theta_values)

    gjit = jax.jit(g)
    gjitval = gjit(theta_values)

    dgdtheta = jax.jit(jax.grad(g))
    dgdthetaval = dgdtheta(theta_values)

    ref_g = np.cos(Jh) * vec_dJdn

    assert np.allclose(dgdthetaval, ref_g)
    assert np.allclose(gjitval, np.sin(Jh))

    # FWD diff test
    def h(theta_values):
        return jnp.sin(J_jax_jvp(theta_values))

    # hval = h(theta_values)

    hjit = jax.jit(h)
    hjitval = hjit(theta_values)

    hjvp = jax.jit(lambda theta_values: jax.jvp(h, (theta_values,), (theta_values,))[1])
    hjvp_val = hjvp(theta_values)

    ref_h = np.dot(ref_g, theta_values)

    assert np.allclose(hjvp_val, ref_h)
    assert np.allclose(hjitval, np.sin(Jh))


def test_rvpinns(cell_type, q_deg, N):
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

    # Define external operator and correct quadrature space
    N = FEMExternalOperator(
        x,
        theta,
        function_space=Q,
        external_function=lambda derivatives: partial(
            u_NN_jax, derivatives=derivatives
        ),
        name="exop",
    )

    V = functionspace(mesh, ("Lagrange", 1))
    phi = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": q_deg})

    u_bc = dolfinx.fem.Function(V)
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    bc_dofs = dolfinx.fem.locate_dofs_topological(
        V, mesh.topology.dim - 1, boundary_facets
    )
    bc = dolfinx.fem.dirichletbc(u_bc, bc_dofs)
    bcs = [bc]

    lmbda = dolfinx.fem.Function(V, name="lmbda")
    phih = dolfinx.fem.Function(V, name="phih")

    a = ufl.inner(ufl.grad(phi), ufl.grad(v)) * dx + phi * v * dx

    L = ufl.action(a, N)

    # NOTE: dJdn = 0 here so things simplify a bit
    def compute_J(phih):
        return ufl.inner(ufl.grad(phih), ufl.grad(phih)) * dx + phih * phih * dx

    def compute_dFdn(F, theta, lmbda):
        dFdtheta = ufl.derivative(F, theta)
        form = ufl.adjoint(dFdtheta)
        return ufl.action(form, lmbda)

    F = ufl.action(a - L, phih)
    J = compute_J(phih)

    forward_problem = LinearProblem(
        a,
        L,
        u=phih,
        bcs=bcs,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "ksp_error_if_not_converged": True,
        },
        petsc_options_prefix="forward_",
    )

    dFdphi = ufl.derivative(F, phih)
    dJdphi = -ufl.derivative(J, phih)
    adjoint_problem = LinearProblem(
        dFdphi,
        dJdphi,
        u=lmbda,
        bcs=bcs,
        petsc_options={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
            "ksp_error_if_not_converged": True,
        },
        petsc_options_prefix="adjoint_",
    )

    dFdtheta = compute_dFdn(F, theta, lmbda)

    J_compiled = compile_external_operator_form(J)
    compiled_dFdtheta = compile_external_operator_form(dFdtheta)

    def eval_J(theta_values: np.ndarray) -> float:
        theta.x.array[:] = theta_values
        theta.x.scatter_forward()
        forward_problem.solve()
        pack_external_operator_data(J_compiled)
        local_contribution = dolfinx.fem.assemble_scalar(J_compiled)
        return mesh.comm.allreduce(local_contribution, op=MPI.SUM)

    def eval_dJdtheta(theta_values: np.ndarray) -> np.ndarray:
        theta.x.array[:] = theta_values
        theta.x.scatter_forward()

        forward_problem.solve()
        adjoint_problem.solve()
        pack_external_operator_data(compiled_dFdtheta)
        vec = dolfinx.fem.assemble_vector(compiled_dFdtheta)
        return vec.array.copy()

    Jh = eval_J(theta_values)
    vec_dJdn = eval_dJdtheta(theta_values)

    test_jax_wrapper(theta_values, eval_J, eval_dJdtheta)

    return Jh, vec_dJdn


if __name__ == "__main__":
    cell_type = dolfinx.mesh.CellType.triangle
    q_deg = 4
    N = 8

    out = test_rvpinns(cell_type, q_deg, N)
