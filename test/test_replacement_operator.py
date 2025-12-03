from mpi4py import MPI

import basix.ufl
import dolfinx
import numpy as np
import pytest
import ufl
from dolfinx_external_operator import FEMExternalOperator, functionspace

from fenicsx_jax.fem import (
    compile_external_operator_form,
    create_real_functionspace,
    pack_external_operator_data,
)

try:
    import jax

    jax.config.update("jax_enable_x64", True)

    def _u_NN(gdim, x, theta):
        if gdim == 1:
            return jax.numpy.sin(theta[0] * x[0])
        else:
            return jax.numpy.sin(theta[0] * x[0]) * jax.numpy.sin(theta[1] * x[1])

    u_NN_vectorized = jax.vmap(_u_NN, in_axes=(None, 0, 0))

    @jax.jit(static_argnums=0)
    def u_NN_jit(gdim, x, theta):
        x_vec = x.reshape(-1, gdim)
        theta_vec = theta.reshape(-1, 4)
        out = u_NN_vectorized(gdim, x_vec, theta_vec)
        return out.flatten().copy()

    d_u_NN_dtheta = jax.jacfwd(_u_NN, argnums=(2))
    d_u_NN_dtheta_vectorized = jax.vmap(d_u_NN_dtheta, in_axes=(None, 0, 0))

    @jax.jit(static_argnums=0)
    def du_NN_dtheta_jit(gdim, x, theta):
        x_vec = x.reshape(-1, gdim)
        theta_vec = theta.reshape(-1, 4)
        out = d_u_NN_dtheta_vectorized(gdim, x_vec, theta_vec)
        return out.reshape(-1).copy()

    def u_NN_jax(gdim, derivatives):
        if derivatives == (0, 0):
            return lambda x, theta: u_NN_jit(gdim, x, theta)
        elif derivatives == (1, 0):
            raise NotImplementedError(f"No function is defined for the {derivatives=}.")
        elif derivatives == (0, 1):
            return lambda x, theta: du_NN_dtheta_jit(gdim, x, theta)
        else:
            raise NotImplementedError(f"No function is defined for the {derivatives=}.")

except ImportError:
    pass


def u_NN(gdim, mod, x, theta):
    if gdim == 1:
        return mod.sin(theta[0] * x[0])
    else:
        return mod.sin(theta[0] * x[0]) * mod.sin(theta[1] * x[1])


def u_NN_impl(gdim, x, theta):
    x_vec = x.reshape(-1, gdim)
    theta_vec = theta.reshape(-1, 4)
    out = u_NN(gdim, np, x_vec.T, theta_vec.T)
    return out.flatten().copy()


def du_NN_dtheta_impl(gdim, x, theta):
    x_vec = x.reshape(-1, gdim).T
    theta_vec = theta.reshape(-1, 4).T
    if gdim == 1:
        out = np.array(
            [
                x_vec[0] * np.cos(theta_vec[0] * x_vec[0]),
                np.zeros_like(x_vec[0]),
                np.zeros_like(x_vec[0]),
                np.zeros_like(x_vec[0]),
            ]
        ).T
    else:
        out = np.array(
            [
                x_vec[0]
                * np.cos(theta_vec[0] * x_vec[0])
                * np.sin(theta_vec[1] * x_vec[1]),
                x_vec[1]
                * np.sin(theta_vec[0] * x_vec[0])
                * np.cos(theta_vec[1] * x_vec[1]),
                np.zeros_like(x_vec[0]),
                np.zeros_like(x_vec[1]),
            ]
        ).T
    return out.reshape(-1).copy()


def u_NN_np(gdim, derivatives):
    if derivatives == (0, 0):
        return lambda x, theta: u_NN_impl(gdim, x, theta)
    elif derivatives == (1, 0):
        raise NotImplementedError(
            f"No function is defined for the derivatives {derivatives}."
        )
    elif derivatives == (0, 1):
        return lambda x, theta: du_NN_dtheta_impl(gdim, x, theta)
    else:
        raise NotImplementedError(
            f"No function is defined for the derivatives {derivatives}."
        )


@pytest.mark.parametrize("q_deg", [1, 4, 8])
@pytest.mark.parametrize("N", [4, 8, 12])
@pytest.mark.parametrize(
    "cell_type",
    [
        dolfinx.mesh.CellType.interval,
        dolfinx.mesh.CellType.triangle,
        dolfinx.mesh.CellType.quadrilateral,
        dolfinx.mesh.CellType.tetrahedron,
        dolfinx.mesh.CellType.hexahedron,
    ],
)
@pytest.mark.parametrize("use_jax", [True, False])
def test_replacement_operator(cell_type, N, q_deg, use_jax):
    tdim = dolfinx.cpp.mesh.cell_dim(cell_type)
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

    R = create_real_functionspace(mesh, value_shape=(4,))
    theta = dolfinx.fem.Function(R)
    theta.x.array[:] = [0.31, 0.42, 0.62, 0.82]
    theta.x.scatter_forward()
    x = ufl.SpatialCoordinate(mesh)

    Qe = basix.ufl.quadrature_element(mesh.basix_cell(), degree=q_deg)
    Q = functionspace(mesh, Qe)
    alpha = dolfinx.fem.Constant(mesh, 1.0)

    # Define external operator and correct quadrature space
    if use_jax:
        pytest.importorskip("jax")
        N = FEMExternalOperator(
            x,
            theta,
            function_space=Q,
            external_function=lambda derivatives: u_NN_jax(
                mesh.geometry.dim, derivatives
            ),
            name="exop",
        )
    else:
        N = FEMExternalOperator(
            x,
            theta,
            function_space=Q,
            external_function=lambda derivatives: u_NN_np(
                mesh.geometry.dim, derivatives
            ),
            name="exop",
        )

    N_ufl = u_NN(mesh.geometry.dim, ufl, x, theta)

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
    J_compiled = compile_external_operator_form(J_ex_op)
    pack_external_operator_data(J_compiled)
    Jh_loc = dolfinx.fem.assemble_scalar(J_compiled)
    Jh = mesh.comm.allreduce(Jh_loc, op=MPI.SUM)

    J_ex = compute_J(N_ufl, phih)
    J_exact_form = dolfinx.fem.form(J_ex)
    J_exact = mesh.comm.allreduce(dolfinx.fem.assemble_scalar(J_exact_form), op=MPI.SUM)
    assert np.isclose(Jh, J_exact)

    F_ex = F(N, phih)
    F_compiled = compile_external_operator_form(F_ex)
    pack_external_operator_data(F_compiled)
    vec = dolfinx.fem.assemble_vector(F_compiled)

    F_ref = F(N_ufl, phih)
    vec_ref = dolfinx.fem.assemble_vector(dolfinx.fem.form(F_ref))
    np.testing.assert_allclose(vec.array, vec_ref.array)

    dFdn_ex = compute_dFdn(F_ex, theta, lmbda)
    dFdn_compiled = compile_external_operator_form(dFdn_ex)
    pack_external_operator_data(dFdn_compiled)
    vec_dFdn = dolfinx.fem.assemble_vector(dFdn_compiled)

    dFdn_ref = compute_dFdn(F_ref, theta, lmbda)
    vec_dFdn_ref = dolfinx.fem.assemble_vector(dolfinx.fem.form(dFdn_ref))
    np.testing.assert_allclose(vec_dFdn.array, vec_dFdn_ref.array)

    dJdn_ex = compute_dJdn(J_ex_op, theta)
    dJdn_compiled = compile_external_operator_form(dJdn_ex)
    pack_external_operator_data(dJdn_compiled)
    vec_dJdn = dolfinx.fem.assemble_vector(dJdn_compiled)

    dJdn_ref = compute_dJdn(J_ex, theta)
    vec_dJdn_ref = dolfinx.fem.assemble_vector(dolfinx.fem.form(dJdn_ref))
    np.testing.assert_allclose(vec_dJdn.array, vec_dJdn_ref.array)
