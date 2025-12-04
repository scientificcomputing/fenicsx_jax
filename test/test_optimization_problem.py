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
    LinearProblem
)

try:
    import jax

    jax.config.update("jax_enable_x64", True)

    def _u_NN(gdim, x, theta):
        val = theta[0]
        for i in range(gdim):
            val *= jax.numpy.sin(theta[i+1] * x[i])
        return val

    u_NN_vectorized = jax.vmap(_u_NN, in_axes=(None, 0, 0))

    @jax.jit(static_argnums=0)
    def u_NN_jit(gdim, x, theta):
        x_vec = x.reshape(-1, gdim)
        theta_vec = theta.reshape(-1, gdim+1)
        out = u_NN_vectorized(gdim, x_vec, theta_vec)
        return out.flatten().copy()

    d_u_NN_dtheta = jax.jacfwd(_u_NN, argnums=(2))
    d_u_NN_dtheta_vectorized = jax.vmap(d_u_NN_dtheta, in_axes=(None, 0, 0))

    @jax.jit(static_argnums=0)
    def du_NN_dtheta_jit(gdim, x, theta):
        x_vec = x.reshape(-1, gdim)
        theta_vec = theta.reshape(-1, gdim+1)
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
    val = theta[0]
    for i in range(gdim):
        val *= mod.sin(theta[i+1] * x[i])
    return val


def u_NN_impl(gdim, x, theta):
    x_vec = x.reshape(-1, gdim)
    theta_vec = theta.reshape(-1, gdim+1)
    out = u_NN(gdim, np, x_vec.T, theta_vec.T)
    return out.flatten().copy()


def du_NN_dtheta_impl(gdim, x, theta):
    x_vec = x.reshape(-1, gdim).T
    theta_vec = theta.reshape(-1, gdim+1).T
    out = np.zeros((gdim + 1, x_vec.shape[1]))
    out[0] = u_NN(gdim, np,  x_vec, theta_vec.copy()) / theta_vec[0]
    for i in range(1, gdim+1):
        out[i] = theta_vec[0]
        for j in range(gdim):
            if i == j + 1:
                out[i] *= x_vec[j] * np.cos(theta_vec[j+1] * x_vec[j])
            else:
                out[i] *= np.sin(theta_vec[j + 1] * x_vec[j])
    return out.T.reshape(-1).copy()


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
def test_adjoint_problem(cell_type, N, q_deg, use_jax):
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

    gdim = mesh.geometry.dim
    R = create_real_functionspace(mesh, value_shape=(gdim+1,))
    theta = dolfinx.fem.Function(R)
    theta.x.array[:] = np.arange(gdim+1) + 0.2
    theta.x.scatter_forward()
    x = ufl.SpatialCoordinate(mesh)

    Qe = basix.ufl.quadrature_element(mesh.basix_cell(), degree=q_deg)
    Q = functionspace(mesh, Qe)
    alpha = dolfinx.fem.Constant(mesh, 1e-6)

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


    V = functionspace(mesh, ("Lagrange", 1))
    phi = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    dx = ufl.Measure("dx", domain=mesh, metadata={"quadrature_degree": q_deg})

    u_bc = dolfinx.fem.Function(V)
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    boundary_facets = dolfinx.mesh.exterior_facet_indices(mesh.topology)
    bc_dofs = dolfinx.fem.locate_dofs_topological(V, mesh.topology.dim-1, boundary_facets)
    bc = dolfinx.fem.dirichletbc(u_bc, bc_dofs)
    bcs = [bc]

    a = ufl.inner(ufl.grad(phi), ufl.grad(v)) * dx
    L = N * v * dx

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

    F = ufl.action(a - L, phih)
    J = compute_J(N, phih)

    forward_problem = LinearProblem(a, L, u=phih,bcs=bcs, petsc_options={"ksp_type": "preonly",
                                                                       "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",
                                                                       "ksp_error_if_not_converged": True},
                                                                       petsc_options_prefix="forward_")
    
    dFdphi = ufl.derivative(F, phih)
    dJdphi = -ufl.derivative(J, phih)
    adjoint_problem = LinearProblem(dFdphi, dJdphi, u=lmbda,bcs=bcs, petsc_options={"ksp_type": "preonly",
                                                                       "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",
                                                                       "ksp_error_if_not_converged": True},
                                                                       petsc_options_prefix="adjoint_")
    
    J_compiled = compile_external_operator_form(J)


    dFdtheta = compute_dFdn(F, theta, lmbda)
    dJdtheta = compute_dJdn(J, theta)
    compiled_dFdtheta = compile_external_operator_form(dFdtheta)
    compiled_dJdtheta = compile_external_operator_form(dJdtheta)

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
        pack_external_operator_data(compiled_dJdtheta)
        vec = dolfinx.fem.assemble_vector(compiled_dJdtheta)
        dolfinx.fem.assemble_vector(vec.array, compiled_dFdtheta)
        vec.scatter_reverse(dolfinx.la.InsertMode.add)
        return vec.array.copy()


    init_J = eval_J(theta.x.array)
    init_dJdtheta = eval_dJdtheta(theta.x.array)


    # Create reference problem with "pure" UFL expressions
    phi_ref = dolfinx.fem.Function(V, name="phih_ref")
    N_ufl = u_NN(mesh.geometry.dim, ufl, x, theta)
    L_ref = N_ufl * v * dx
    F_ref = ufl.action(a - L_ref, phi_ref)
    dFdphi_ref = ufl.derivative(F_ref, phi_ref)
    J_ref = compute_J(N_ufl, phi_ref)
    dJdphi_ref = -ufl.derivative(J_ref, phi_ref)
    dFdtheta_ref = compute_dFdn(F_ref, theta, lmbda)
    dJdtheta_ref = compute_dJdn(J_ref, theta)
    forward_problem_ref = LinearProblem(a, L_ref, u=phi_ref,bcs=bcs, petsc_options={"ksp_type": "preonly",
                                                                       "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",
                                                                       "ksp_error_if_not_converged": True},
                                                                       petsc_options_prefix="forward_ref_")
    adjoint_problem_ref = LinearProblem(dFdphi_ref, dJdphi_ref, u=lmbda,bcs=bcs, petsc_options={"ksp_type": "preonly",
                                                                       "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",
                                                                       "ksp_error_if_not_converged": True},
                                                                       petsc_options_prefix="adjoint_ref_")
    def eval_J_ref(theta_values: np.ndarray) -> float:
        theta.x.array[:] = theta_values
        theta.x.scatter_forward()
        forward_problem_ref.solve()
        local_contribution = dolfinx.fem.assemble_scalar(dolfinx.fem.form(J_ref))
        return mesh.comm.allreduce(local_contribution, op=MPI.SUM)
    
    def eval_dJdtheta_ref(theta_values: np.ndarray) -> np.ndarray:
        theta.x.array[:] = theta_values
        theta.x.scatter_forward()
    
        forward_problem_ref.solve()
        adjoint_problem_ref.solve()
        vec = dolfinx.fem.assemble_vector(dolfinx.fem.form(dJdtheta_ref))
        dolfinx.fem.assemble_vector(vec.array, dolfinx.fem.form(dFdtheta_ref))
        vec.scatter_reverse(dolfinx.la.InsertMode.add)
        return vec.array.copy()


    init_ref_J = eval_J_ref(theta.x.array)
    init_ref_dJdtheta = eval_dJdtheta_ref(theta.x.array)
    np.testing.assert_allclose(init_J, init_ref_J)
    np.testing.assert_allclose(init_dJdtheta, init_ref_dJdtheta)
