# fenicsx_jax

Example of how to interface JAX and FEniCSx using external operators

To install the software, first install `fenics-dolfinx` (>=v0.10.0) on your system.
This can for instance be done through:

- Docker (`ghcr.io/fenics/dolfinx/dolfinx:stable`)
- Conda (`fenics-dolfinx`)
- Spack (`py-fenics-dolfinx`)

Secondly, to use real-valued spaces, install `scifem`:

- If you used docker above
  ```bash
  python3 -m pip install packaging scikit-build-core[pyproject] nanobind setuptools packaging pkgconfig
  python3 -m pip install scifem --no-build-isolation
  ```
- If you used conda, `conda install -c conda-forge scifem`
- If you used spack, `spack install py-scifem`

Finally install this package with `python3 -m pip install --no-build-isolation -e .[test]`
