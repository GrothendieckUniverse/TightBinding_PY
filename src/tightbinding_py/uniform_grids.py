"""Uniform grids in crystal and Cartesian coordinates.

Faithful port of ``src/uniform_grids.jl`` from the Julia package
``TightBinding.jl``.

A :class:`Uniform_Grids` object is mostly used for momentum-space grids: the
basis vectors are then the reciprocal vectors of a real-space lattice, and
the twisted phases shift the grid off the high-symmetry points.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from .lattice import Real_Space_Lattice


@dataclass
class Uniform_Grids:
    """Uniform grid (mostly for momentum-space grids).

    Julia counterpart: ``struct Uniform_Grids{T,U}``.  It can either be
    constructed independently, or from a real-space lattice with all-
    direction periodic boundary conditions.  Note: for the latter case the
    real-space sublattice degrees of freedom are transferred to the
    dimension of the single-particle k-space Hamiltonian, which should be
    handled elsewhere.

    Attributes:
        name: name of the uniform grid.
        dim: dimension of the uniform grid.
        sample_size: number of grid points in each direction.
        basis_vec_list: list of basis vectors (for a k-space lattice, these
            are just the reciprocal vectors).
        cell_volume: volume of the unit cell of the uniform grid (in
            momentum space — distinguish it from the real-space cell
            volume!).
        twisted_phases_over_2π: twisted phases (over 2π); these shift the
            ``site_crys_list`` as well as the ``site_cart_list``.
        site_int_list: list of integer site indices of the grid points
            (first axis fastest, as in the lattice).
        site_int_to_index_map: hashmap ``site_int → i_site`` (**1-based**).
        site_crys_list: positions of the grid points in crystal
            coordinates ``(site_int + twist) ./ sample_size``.
        site_cart_list: positions of the grid points in Cartesian
            coordinates.
        nsite: number of grid points.
    """

    name: str
    dim: int
    sample_size: list[int]

    basis_vec_list: list[list[float]]
    cell_volume: float

    twisted_phases_over_2π: list[float]
    site_int_list: list[tuple[int, ...]]
    site_int_to_index_map: dict[tuple[int, ...], int]
    site_crys_list: list[np.ndarray]
    site_cart_list: list[np.ndarray]

    nsite: int

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Uniform_Grids(name={self.name!r}, dim={self.dim}, "
            f"sample_size={self.sample_size}, nsite={self.nsite})"
        )


def _enumerate_site_ints(sample_size: list[int]) -> list[tuple[int, ...]]:
    """Enumerate integer grid indices (first axis fastest — Julia order).

    Args:
        sample_size: number of grid points in each direction.

    Returns:
        list[tuple[int, ...]]: the integer index tuples.
    """
    ranges = [range(n) for n in sample_size]
    return [
        tuple(reversed(c)) for c in product(*reversed(ranges))
    ]


def _assemble_grid(
    name: str,
    sample_size: list[int],
    basis_vec_list: list[list[float]],
    twisted_phases_over_2π: list[float],
) -> Uniform_Grids:
    """Shared field computation of both :class:`Uniform_Grids` constructors.

    Args:
        name: grid name.
        sample_size: grid points per direction.
        basis_vec_list: basis vectors.
        twisted_phases_over_2π: twisted phases.

    Returns:
        Uniform_Grids: the assembled grid.
    """
    basis_vec_mat = np.asarray(basis_vec_list, dtype=np.float64).T  # columns
    cell_volume = float(abs(np.linalg.det(basis_vec_mat)))

    site_int_list = _enumerate_site_ints(sample_size)
    nsite = len(site_int_list)
    site_int_to_index_map = {
        site_int: i_site
        for i_site, site_int in enumerate(site_int_list, start=1)
    }

    sample = np.asarray(sample_size, dtype=np.float64)
    twist = np.asarray(twisted_phases_over_2π, dtype=np.float64)
    site_crys_list = [
        (np.asarray(site_int, dtype=np.float64) + twist) / sample
        for site_int in site_int_list
    ]
    basis_rows = np.asarray(basis_vec_list, dtype=np.float64)
    site_cart_list = [crys @ basis_rows for crys in site_crys_list]

    return Uniform_Grids(
        name=name,
        dim=len(sample_size),
        sample_size=[int(n) for n in sample_size],
        basis_vec_list=[[float(x) for x in v] for v in basis_vec_list],
        cell_volume=cell_volume,
        twisted_phases_over_2π=[float(x) for x in twisted_phases_over_2π],
        site_int_list=site_int_list,
        site_int_to_index_map=site_int_to_index_map,
        site_crys_list=site_crys_list,
        site_cart_list=site_cart_list,
        nsite=nsite,
    )


def initialize_uniform_grids(
    *,
    basis_vec_list: list[list[float]],
    sample_size: list[int],
    name: str = "",
    twisted_phases_over_2π: list[float],
) -> Uniform_Grids:
    """Constructor of :class:`Uniform_Grids` (independent of any lattice).

    Julia counterpart: the first method of ``initialize_uniform_grids``.

    Args:
        basis_vec_list: list of basis vectors for the uniform grid (for a
            k-space grid, the reciprocal vectors).
        sample_size: number of grid points in each direction.
        name: name of the uniform grid.
        twisted_phases_over_2π: twisted phases (over 2π); these shift the
            ``site_crys_list`` as well as the ``site_cart_list``.

    Returns:
        Uniform_Grids: the grid.

    Raises:
        ValueError: if the basis vectors are inconsistent with the sample
            size.
    """
    dim = len(sample_size)
    if len(basis_vec_list) != dim:
        raise ValueError(
            "The length of `basis_vec_list` must be the same as the "
            "dimension of the uniform grid!"
        )
    if any(len(v) != dim for v in basis_vec_list):
        raise ValueError(
            "Every basis vector in `basis_vec_list` must have the same "
            "dimension as the uniform grid!"
        )
    return _assemble_grid(name, sample_size, basis_vec_list, twisted_phases_over_2π)


def initialize_uniform_grids_from_lattice(
    r_data: Real_Space_Lattice,
    *,
    twisted_phases_over_2π: list[float] | None = None,
) -> Uniform_Grids:
    """Constructor of :class:`Uniform_Grids` from a real-space lattice.

    Julia counterpart: the second method of ``initialize_uniform_grids``
    (dispatch on ``r_data::Real_Space_Lattice``).  The lattice must satisfy
    PBC in ALL directions.

    Args:
        r_data: the underlying real-space lattice (PBC in ALL directions).
        twisted_phases_over_2π: twisted phases (over 2π); defaults to
            ``r_data.twisted_phases_over_2π``.

    Returns:
        Uniform_Grids: the k-space grid (name = ``lattice_name``,
        basis vectors = reciprocal vectors).
    """
    if twisted_phases_over_2π is None:
        twisted_phases_over_2π = r_data.twisted_phases_over_2π
    if (
        np.linalg.norm(
            np.asarray(twisted_phases_over_2π)
            - np.asarray(r_data.twisted_phases_over_2π)
        )
        > 1.0e-10
    ):
        print(
            "WARNING: Input `twisted_phases_over_2π`="
            f"{twisted_phases_over_2π} differs from the lattice's stored "
            f"value `r_data.twisted_phases_over_2π`="
            f"{r_data.twisted_phases_over_2π}. This may lead to unexpected "
            "behavior!"
        )

    dim = r_data.dim
    sample_size = r_data.sample_size

    dual_basis_vec_mat = np.asarray(r_data.brav_vec_list, dtype=np.float64).T
    # for the momentum-space lattice, the `dual_basis_vec_mat` is the
    # real-space Bravais matrix
    basis_vec_mat = 2.0 * np.pi * np.linalg.inv(dual_basis_vec_mat).T

    cell_volume = float(abs(np.linalg.det(basis_vec_mat)))
    basis_vec_list = [basis_vec_mat[:, i].tolist() for i in range(dim)]

    grid = _assemble_grid(
        r_data.lattice_name,
        sample_size,
        basis_vec_list,
        list(twisted_phases_over_2π),
    )
    grid.cell_volume = cell_volume  # recomputed identically in _assemble_grid
    return grid


__all__ = [
    "Uniform_Grids",
    "initialize_uniform_grids",
    "initialize_uniform_grids_from_lattice",
]
