# Copyright (C) Simula Research Laboratory, Jørgen S. Dokken
# SPDX-License-Identifier:    LGPL-3.0-or-later

import basix
import dolfinx
import dolfinx_external_operator.fem
import numpy as np
import scifem
from dolfinx_external_operator import (
    evaluate_external_operators,
    evaluate_operands,
    replace_external_operators,
)

__all__ = [
    "compile_external_operator_form",
    "pack_external_operator_data",
    "LinearProblem",
    "create_real_functionspace",
]


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
    compiled_form._ex_ops = ex_ops
    return compiled_form


def pack_external_operator_data(form: dolfinx.fem.Form | list[dolfinx.fem.Form]):
    if isinstance(form, dolfinx.fem.Form):
        external_operators = form._ex_ops
        if len(external_operators) == 0:
            return
        operands = evaluate_operands(external_operators)
        evaluate_external_operators(external_operators, operands)
    else:
        for f in form:
            external_operators = f._ex_ops
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
        "P", mesh.basix_cell(), 0, dtype=dtype, discontinuous=True, shape=value_shape
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
