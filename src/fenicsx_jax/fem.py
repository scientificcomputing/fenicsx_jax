# Copyright (C) Simula Research Laboratory, Jørgen S. Dokken
# SPDX-License-Identifier:    LGPL-3.0-or-later

import basix
import dolfinx.fem.petsc
import dolfinx_external_operator.fem
import numpy as np
import ufl
from dolfinx_external_operator import (
    FEMExternalOperator,
    evaluate_external_operators,
    replace_external_operators,
)

__all__ = [
    "compile_external_operator_form",
    "pack_external_operator_data",
    "LinearProblem",
]

try:
    import scifem

    __all__.append("create_real_functionspace")

    def create_real_functionspace(
        mesh: dolfinx.mesh.Mesh, value_shape: tuple[int, ...] = ()
    ) -> dolfinx.fem.FunctionSpace:
        """Create a real function space.

        Args:
            mesh: The mesh the real space is defined on.
            value_shape: The shape of the values in the real space.

        Returns:
            The real valued function space.
        Note:
            For scalar elements value shape is ``()``.

        """

        dtype = mesh.geometry.x.dtype
        ufl_e = basix.ufl.element(
            "P",
            mesh.basix_cell(),
            0,
            dtype=dtype,
            discontinuous=True,
            shape=value_shape,
        )

        if (dtype := mesh.geometry.x.dtype) == np.float64:
            cppV = scifem._scifem.create_real_functionspace_float64(
                mesh._cpp_object, value_shape
            )
        elif dtype == np.float32:
            cppV = scifem._scifem.create_real_functionspace_float32(
                mesh._cpp_object, value_shape
            )
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")
        return dolfinx_external_operator.fem.FunctionSpace(mesh, ufl_e, cppV)
except ImportError:
    pass


def evaluate_operands(
    external_operators: list[FEMExternalOperator],
    entities: np.ndarray | None = None,
) -> dict[ufl.core.expr.Expr | int, np.ndarray]:
    """Evaluates operands of external operators.

    This function is based on evaluate_operands from DOLFINx external operator,
    under the LGPL-3.0-or-later license. Original author: Andrey Latyshev.
    This version is modified to avoid extra memory allocation for real spaces.

    Args:
        external_operators: A list with external operators required to be updated.
        entities: A dictionary mapping between parent mesh and sub mesh
        entities with respect to `eval` function of `fem.Expression`.
    Returns:
        A map between UFL operand and the `ndarray`, the evaluation of the
        operand.

    Note:
        User is responsible to ensure that `entities` are correctly constructed
        with respect to the codimension of the external operator.
    """
    if len(external_operators) == 0:
        return {}
    ref_function_space = external_operators[0].ref_function_space
    ufl_element = ref_function_space.ufl_element()
    mesh = ref_function_space.mesh
    quadrature_points = basix.make_quadrature(
        ufl_element.cell_type, ufl_element.degree
    )[0]

    # If no entity map is provided, assume that there is no sub-meshing
    if entities is None:
        map_c = mesh.topology.index_map(mesh.topology.dim)
        num_cells = map_c.size_local + map_c.num_ghosts
        cells = np.arange(0, num_cells, dtype=np.int32)
        entities = cells

    # Evaluate unique operands in external operators
    evaluated_operands = {}
    for external_operator in external_operators:
        # TODO: Is it possible to get the basix information out here?
        for operand in external_operator.ufl_operands:
            try:
                evaluated_operands[operand]
            except KeyError:
                # Check if we have a sub-mesh with different codim
                operand_domain = ufl.domain.extract_unique_domain(operand)
                operand_mesh = dolfinx.mesh.Mesh(
                    operand_domain.ufl_cargo(), operand_domain
                )
                # TODO: Stop recreating the expression every time
                expr = dolfinx.fem.Expression(
                    operand,
                    quadrature_points,
                    dtype=external_operator.ref_coefficient.dtype,
                )
                # NOTE: Using expression eval might be expensive
                if isinstance(operand, dolfinx.fem.Function):
                    # Check for real function space
                    if operand.function_space.dofmap.index_map.size_global == 1:
                        if entities.ndim == 1:
                            entity = entities[:1]
                        elif entities.ndim == 2:
                            entity = entities[:1, :]
                        else:
                            raise ValueError("Entities array has too many dimensions.")
                        evaluated_operand_at_entity = expr.eval(operand_mesh, entity)
                        c_size = evaluated_operand_at_entity.shape[-1]
                        evaluated_operand = np.lib.stride_tricks.as_strided(
                            evaluated_operand_at_entity,
                            shape=(len(entities), quadrature_points.shape[0], c_size),
                            strides=(0, 0, evaluated_operand_at_entity.itemsize),
                            writeable=False,
                        )
                else:
                    evaluated_operand = expr.eval(operand_mesh, entities)

            evaluated_operands[operand] = evaluated_operand
    return evaluated_operands


def compile_external_operator_form(
    form: dolfinx.fem.Form,
    jit_options: dict | None = None,
    form_compiler_options: dict | None = None,
    entity_maps: list[dolfinx.mesh.EntityMap] | None = None,
) -> dolfinx.fem.Form:
    form_replaced, ex_ops = replace_external_operators(form)
    compiled_form = dolfinx.fem.form(
        form_replaced,
        jit_options=jit_options,
        form_compiler_options=form_compiler_options,
        entity_maps=entity_maps,
    )
    compiled_form._ex_ops = ex_ops  # type: ignore[attr-defined]
    return compiled_form


def pack_external_operator_data(form: dolfinx.fem.Form | list[dolfinx.fem.Form]):
    if isinstance(form, dolfinx.fem.Form):
        external_operators = form._ex_ops  # type: ignore[attr-defined]
        if len(external_operators) == 0:
            return
        operands = evaluate_operands(external_operators)
        evaluate_external_operators(external_operators, operands)
    else:
        for f in form:
            external_operators = f._ex_ops  # type: ignore[attr-defined]
            if len(external_operators) == 0:
                continue
            operands = evaluate_operands(external_operators)
            evaluate_external_operators(external_operators, operands)


class LinearProblem(dolfinx.fem.petsc.LinearProblem):
    def __init__(
        self,
        a,
        L,
        u=None,
        bcs=None,
        petsc_options=None,
        petsc_options_prefix=None,
        entity_maps=None,
    ):
        a_replaced, a_ex_ops = replace_external_operators(a)
        L_replaced, L_ex_ops = replace_external_operators(L)
        super().__init__(
            a_replaced,
            L_replaced,
            u=u,
            bcs=bcs,
            petsc_options=petsc_options,
            petsc_options_prefix=petsc_options_prefix,
            entity_maps=entity_maps,
        )
        self._a._ex_ops = a_ex_ops
        self._L._ex_ops = L_ex_ops

    def solve(self):
        pack_external_operator_data([self._a, self._L])
        return super().solve()
