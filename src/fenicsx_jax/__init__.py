# Copyright (C) Simula Research Laboratory, Jørgen S. Dokken
# SPDX-License-Identifier:    LGPL-3.0-or-later


from .fem import compile_external_operator_form, pack_external_operator_data

from importlib.metadata import metadata

__all__ = ["compile_external_operator_form", "pack_external_operator_data", "__version__", "__author__", "__license__", "__email__", "__program_name__"]


meta = metadata("fenicsx_jax")
__version__ = meta["Version"]
__author__ = meta.get("Author", "")
__license__ = meta["License"]
__email__ = meta["Author-email"]
__program_name__ = meta["Name"]
