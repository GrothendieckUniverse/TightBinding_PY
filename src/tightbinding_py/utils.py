"""Utility functions: reciprocal basis, H(k), Chern numbers, bilinear terms.

Faithful port of ``src/utils.jl`` from the Julia package ``TightBinding.jl``.
"""

from __future__ import annotations

import cmath
import math
from typing import Callable

import numpy as np
import scipy.sparse as sp

from .lattice import Real_Space_Lattice
from .tb_model import Real_Space_TightBinding_Model


def dual_basis_vec_mat(basis_vec_mat) -> np.ndarray:
    """Dual basis vector matrix of a given basis vector matrix.

    Julia counterpart: ``dual_basis_vec_mat``.  The dual basis satisfies
    ``dual_basis_vec_mat.T @ basis_vec_mat = 2π * I``, where both matrices
    are stored **in columns** (``basis_vec_mat = [v1 v2 …]``).  Useful to
    transform between real-space and momentum-space bases.

    Args:
        basis_vec_mat: square matrix whose columns are the basis vectors
            (or a list of basis vectors, which is ``hcat``-ed into
            columns).

    Returns:
        np.ndarray: the dual basis matrix ``2π * inv(basis_vec_mat).T``
        (columns are the dual vectors).

    Raises:
        AssertionError: if the duality relation is violated numerically.
    """
    if isinstance(basis_vec_mat, (list, tuple)):
        basis_vec_mat = np.asarray(basis_vec_mat, dtype=np.float64).T
    basis_vec_mat = np.asarray(basis_vec_mat, dtype=np.float64)
    res_vec_mat = 2.0 * math.pi * np.linalg.inv(basis_vec_mat).T
    assert np.linalg.norm(res_vec_mat.T @ basis_vec_mat - 2.0 * math.pi * np.eye(basis_vec_mat.shape[0])) < 1.0e-10, (
        "The computed `dual_basis_vec_mat` does not satisfy the relation "
        "`dual_basis_vec_mat.T @ basis_vec_mat = 2π * I`!"
    )
    return res_vec_mat


def dual_basis_vec_list(basis_vec_list: list[list[float]]) -> list[list[float]]:
    """Dual basis vector list of a given basis vector list.

    Julia counterpart: ``dual_basis_vec_list``.  Uses
    :func:`dual_basis_vec_mat` to satisfy
    ``dual_basis_vec_mat.T @ basis_vec_mat = 2π * I`` (matrices stored in
    columns).

    Args:
        basis_vec_list: Bravais vectors for the real-space lattice, or
            reciprocal vectors for a k-space lattice.

    Returns:
        list[list[float]]: the dual vectors (columns of the dual matrix).
    """
    basis_vec_mat = np.asarray(basis_vec_list, dtype=np.float64).T
    return [col.tolist() for col in dual_basis_vec_mat(basis_vec_mat).T]


def build_Hk_crys(
    tb_model: Real_Space_TightBinding_Model,
) -> Callable[[np.ndarray], np.ndarray]:
    """Construct the crystal-momentum Hamiltonian ``H(k)`` in periodic gauge.

    Julia counterpart: ``build_Hk_crys``.  The result strictly satisfies
    :math:`H(\\mathbf k + \\mathbf G) = H(\\mathbf k)`.  (If instead one
    builds :math:`H^{\\alpha\\beta} = \\sum_{\\mathbf R'} t^{0,\\alpha;
    \\mathbf R',\\beta} e^{i\\mathbf k\\cdot[(\\mathbf R'+
    \\boldsymbol\\tau_\\beta) - (0+\\boldsymbol\\tau_\\alpha)]}`, the gauge
    is **not** periodic in the BZ.)

    Args:
        tb_model: the real-space tight-binding model.  If
            ``input_hopping_map`` is non-empty the infinite-system hopping
            templates are used (no finite-size wrapping); otherwise the
            graph-generated finite-torus hoppings of ``full_hopping_map``
            are compressed into one hopping template (divided by the number
            of unit cells).

    Returns:
        Callable[[np.ndarray], np.ndarray]: the function sending the
        crystal k-vector to the ``n_sub × n_sub`` Hamiltonian matrix
        ``H(k)``.
    """
    l = tb_model.lattice
    n_sub = l.n_sub

    def Hk_crys(k_crys) -> np.ndarray:
        k = np.asarray(k_crys, dtype=np.float64)
        Hk = np.zeros((n_sub, n_sub), dtype=np.complex128)

        if tb_model.input_hopping_map:
            # Infinite-system hopping templates: no finite-size wrapping.
            for ((site_from, site_to), amp) in tb_model.input_hopping_map.items():
                (cell_from, sub_from) = site_from
                (cell_to, sub_to) = site_to

                Δ_cell = np.asarray(cell_to, dtype=np.float64) - np.asarray(
                    cell_from, dtype=np.float64
                )

                Hk[sub_from - 1, sub_to - 1] += amp * cmath.exp(
                    2j * math.pi * float(np.dot(k, Δ_cell))
                )
        else:
            # Graph-generated finite torus hoppings:
            # compress translated copies into one hopping template.
            reduced: dict[tuple, complex] = {}
            for ((site_from, site_to), amp) in tb_model.full_hopping_map.items():
                (cell_from, sub_from) = site_from
                (cell_to, sub_to) = site_to

                Δ_cell = np.asarray(cell_to, dtype=np.float64) - np.asarray(
                    cell_from, dtype=np.float64
                )
                from .lattice import _wrap_Δ_crys

                Δ_cell = _wrap_Δ_crys(
                    Δ_cell,
                    sample_size=l.sample_size,
                    pbc_indicator=l.pbc_indicator,
                )

                Δ_key = tuple(int(round(c)) for c in Δ_cell)
                key = (sub_from, sub_to, Δ_key)
                reduced[key] = reduced.get(key, 0.0 + 0.0j) + amp

            # Divide by the number of unit cells because full_hopping_map
            # contains one translated copy per cell.
            for ((sub_from, sub_to, Δ_key), amp_sum) in reduced.items():
                Δ_cell = np.asarray(Δ_key, dtype=np.float64)
                amp = amp_sum / l.n_cell

                Hk[sub_from - 1, sub_to - 1] += amp * cmath.exp(
                    2j * math.pi * float(np.dot(k, Δ_cell))
                )

        return Hk

    return Hk_crys


def Chern_number_Fukui_Hatsugai_Suzuki(
    Hk_crys: Callable[[np.ndarray], np.ndarray], *, band: int, nk: int = 51
) -> float:
    """Chern number of one band via the Fukui–Hatsugai–Suzuki method.

    Julia counterpart: ``Chern_number_Fukui_Hatsugai_Suzuki``.  A single-band
    Chern number is computed for an ``n_sub``-band tight-binding Hamiltonian
    ``H(k)`` discretized on a single ``nk × nk`` k-grid in the crystal
    momentum BZ.

    Reference:
        T. Fukui, Y. Hatsugai, & H. Suzuki, *J. Phys. Soc. Jpn.* **74**,
        1674–1677 (2005).

    Args:
        Hk_crys: function ``Hk_crys(k_crys) → Hamiltonian matrix``.
        band: **1-based** band index (``1`` = lowest, ``2`` = next).
        nk: number of k-points per direction of the grid.

    Returns:
        float: the Chern number of the given band.
    """
    H0 = np.asarray(Hk_crys([0.0, 0.0]))
    n_sub = H0.shape[0]

    vecs = np.empty((n_sub, nk, nk), dtype=np.complex128)

    for i in range(nk):
        for j in range(nk):
            k = np.asarray([i / nk, j / nk], dtype=np.float64)
            Hk = np.asarray(Hk_crys(k))
            H_herm = (Hk + Hk.conj().T) / 2.0
            F = np.linalg.eigh(H_herm)
            vecs[:, i, j] = F[1][:, band - 1]

    def link(u: np.ndarray, v: np.ndarray) -> complex:
        z = np.vdot(u, v)  # vdot conjugates the first argument (like Julia)
        return z / abs(z)

    total_flux = 0.0

    for i in range(nk):
        for j in range(nk):
            ip = (i + 1) % nk
            jp = (j + 1) % nk

            u = vecs[:, i, j]
            ux = vecs[:, ip, j]
            uy = vecs[:, i, jp]
            uxy = vecs[:, ip, jp]

            Ux = link(u, ux)
            Uy = link(u, uy)
            Ux_y = link(uy, uxy)
            Uy_x = link(ux, uxy)

            total_flux += cmath.phase(Ux * Uy_x / (Ux_y * Uy))

    return total_flux / (2.0 * math.pi)


def _compute_winding(
    cell_to_new: list[int], L: list[int], pbc: list[bool], dim: int
) -> list[int] | None:
    """Compute winding numbers across periodic boundaries and wrap in place.

    Julia counterpart: ``_compute_winding!`` (the Python version mutates
    the input list in place and returns the winding numbers, or ``None``).

    Args:
        cell_to_new: proposed destination cell (modified **in place**).
        L: sample size in each direction.
        pbc: periodic-boundary-condition indicators.
        dim: dimension of the lattice.

    Returns:
        list[int] | None: the winding numbers, or ``None`` to skip (the
        cell lies outside the sample in some open direction).
    """
    winding = [0] * dim
    for d in range(dim):
        if pbc[d]:
            winding[d] = cell_to_new[d] // L[d]  # fld: floor division
            cell_to_new[d] = cell_to_new[d] % L[d]  # julia mod
        else:
            # Open boundary: drop hoppings that leave the sample
            if cell_to_new[d] < 0 or cell_to_new[d] >= L[d]:
                return None
    return winding


def generate_bilinear_terms(
    tb_model: Real_Space_TightBinding_Model,
    *,
    twisted_phases_over_2π: list[float] | None = None,
) -> list[tuple[int, int, complex]]:
    """Generate bilinear terms ``t_ij c†_i c_j`` as ``(i, j, amplitude)``.

    Julia counterpart: ``generate_bilinear_terms``.  Extracts all bilinear
    operator terms from either ``input_hopping_map`` (template hoppings) or
    ``full_hopping_map`` (graph-generated hoppings), *applying the
    specified twisted boundary phases*: a hopping that crosses the periodic
    boundary in direction ``d`` with winding number ``w_d`` acquires an
    extra factor ``exp(i·2π·twisted_phases_over_2π[d]·w_d)``.

    Args:
        tb_model: the tight-binding model.
        twisted_phases_over_2π: twisted phases φ/(2π).  Falls back to
            ``tb_model.lattice.twisted_phases_over_2π`` when ``None``.

    Returns:
        list[tuple[int, int, complex]]: list of ``(i_site, j_site,
        amplitude)`` terms with **1-based** site indices.
    """
    lattice = tb_model.lattice
    dim = lattice.dim
    L = lattice.sample_size
    pbc = lattice.pbc_indicator

    if twisted_phases_over_2π is None:
        twisted_phases_over_2π = lattice.twisted_phases_over_2π

    terms: list[tuple[int, int, complex]] = []

    use_input = bool(tb_model.input_hopping_map)

    if use_input:
        # --- Build from input_hopping_map (template hoppings) ---
        for ((site_from, site_to), amp) in tb_model.input_hopping_map.items():
            (cell_from, sub_from) = site_from
            (cell_to, sub_to) = site_to

            Δ_cell_template = [
                c2 - c1 for c1, c2 in zip(cell_from, cell_to)
            ]

            for cell in lattice.cell_int_list:
                cell_to_new = [cell[d] + Δ_cell_template[d] for d in range(dim)]

                winding = _compute_winding(cell_to_new, L, pbc, dim)
                if winding is None:
                    continue

                phase = cmath.exp(
                    2j
                    * math.pi
                    * sum(
                        t * w
                        for t, w in zip(twisted_phases_over_2π, winding)
                    )
                )

                i_site = lattice.site_to_index_map[(cell, sub_from)]
                j_site = lattice.site_to_index_map[
                    (tuple(cell_to_new), sub_to)
                ]

                terms.append((i_site, j_site, amp * phase))
    else:
        # --- Build from full_hopping_map (already translated & wrapped) ---
        # Reconstruct templates: collect all hoppings starting from cell 0.
        template_map: dict[tuple, complex] = {}
        for ((site_from, site_to), amp) in tb_model.full_hopping_map.items():
            (cell_from, sub_from) = site_from
            (cell_to, sub_to) = site_to
            if all(c == 0 for c in cell_from):
                Δ = list(cell_to)
                key = (sub_from, sub_to, tuple(Δ))
                template_map[key] = template_map.get(key, 0.0 + 0.0j) + amp

        for ((sub_from, sub_to, Δ_cell_template), amp) in template_map.items():
            for cell in lattice.cell_int_list:
                cell_to_new = [cell[d] + Δ_cell_template[d] for d in range(dim)]

                winding = _compute_winding(cell_to_new, L, pbc, dim)
                if winding is None:
                    continue

                phase = cmath.exp(
                    2j
                    * math.pi
                    * sum(
                        t * w
                        for t, w in zip(twisted_phases_over_2π, winding)
                    )
                )

                i_site = lattice.site_to_index_map[(cell, sub_from)]
                j_site = lattice.site_to_index_map[
                    (tuple(cell_to_new), sub_to)
                ]

                terms.append((i_site, j_site, amp * phase))

    return terms


def build_real_space_tb_Hamiltonain(
    tb_model: Real_Space_TightBinding_Model,
    *,
    twisted_phases_over_2π: list[float] | None = None,
) -> sp.csc_matrix:
    """Real-space tight-binding Hamiltonian matrix (sparse, twisted BC).

    Julia counterpart: ``build_real_space_tb_Hamiltonain`` (returns
    ``SparseMatrixCSC``; the Python port returns a scipy CSC matrix, the
    natural analogue).  Constructs the ``n_site × n_site`` real-space
    Hamiltonian using :func:`generate_bilinear_terms` to extract all
    ``(i_to, j_from, t_ij)`` triples.  A hopping crossing the periodic
    boundary in direction ``d`` with winding ``w_d`` acquires the extra
    phase ``exp(i·2π·twisted_phases_over_2π[d]·w_d)``.

    Args:
        tb_model: the tight-binding model.
        twisted_phases_over_2π: twisted phases φ/(2π); falls back to
            ``tb_model.lattice.twisted_phases_over_2π`` when ``None``.

    Returns:
        scipy.sparse.csc_matrix: the ``n_site × n_site`` sparse Hamiltonian.
    """
    n_site = tb_model.lattice.n_site

    bilinear_terms = generate_bilinear_terms(
        tb_model, twisted_phases_over_2π=twisted_phases_over_2π
    )

    rows = np.zeros(len(bilinear_terms), dtype=np.int64)
    cols = np.zeros(len(bilinear_terms), dtype=np.int64)
    vals = np.zeros(len(bilinear_terms), dtype=np.complex128)

    for (k, (i_to, j_from, amp)) in enumerate(bilinear_terms):
        cols[k] = j_from - 1  # 1-based → 0-based
        rows[k] = i_to - 1
        vals[k] = amp

    return sp.csc_matrix(
        (vals, (rows, cols)), shape=(n_site, n_site)
    )


def many_body_Chern_number_Fukui_Hatsugai_Suzuki(
    tb_model: Real_Space_TightBinding_Model, *, n_occ: int, nθ: int = 21
) -> float:
    """Many-body Chern number via the Fukui–Hatsugai–Suzuki method.

    Julia counterpart: ``many_body_Chern_number_Fukui_Hatsugai_Suzuki``.
    For a non-interacting tight-binding model, the many-body Chern number is
    computed by discretizing the flux torus ``(θ₁, θ₂) ∈ [0,1]²`` and
    applying the Fukui–Hatsugai–Suzuki method to the many-body ground-state
    Slater determinant.  It equals the sum of the single-particle Chern
    numbers of all occupied bands.

    Args:
        tb_model: the tight-binding model (must have PBC in ALL directions).
        n_occ: number of occupied single-particle states (filling).
        nθ: number of θ-points per direction of the flux grid.

    Returns:
        float: the many-body Chern number.
    """
    l = tb_model.lattice
    assert all(l.pbc_indicator), (
        "All directions must be periodic (torus) for the many-body "
        "Chern number."
    )

    n_site = l.n_site
    assert 1 <= n_occ <= n_site, (
        f"n_occ={n_occ} must be between 1 and n_site={n_site}."
    )

    # Pre-allocate storage for the occupied eigenvectors at each θ-point
    occ_vecs: list[list[np.ndarray]] = [[None] * nθ for _ in range(nθ)]

    for i in range(nθ):
        θ1 = i / nθ
        for j in range(nθ):
            θ2 = j / nθ
            θ_vec = [θ1, θ2]  # crystal-coordinate flux phases

            H = build_real_space_tb_Hamiltonain(
                tb_model, twisted_phases_over_2π=θ_vec
            ).toarray()
            H_herm = (H + H.conj().T) / 2.0
            F = np.linalg.eigh(H_herm)
            occ_vecs[i][j] = F[1][:, :n_occ]

    def slater_link(
        V1: np.ndarray, V2: np.ndarray
    ) -> complex:
        """Overlap phase of two Slater determinants (occupied columns)."""
        S = V1.conj().T @ V2  # (n_occ × n_occ) overlap matrix
        z = np.linalg.det(S)
        return z / abs(z)

    total_flux = 0.0

    for i in range(nθ):
        for j in range(nθ):
            ip = (i + 1) % nθ
            jp = (j + 1) % nθ

            V = occ_vecs[i][j]
            Vx = occ_vecs[ip][j]
            Vy = occ_vecs[i][jp]
            Vxy = occ_vecs[ip][jp]

            Ux = slater_link(V, Vx)
            Uy = slater_link(V, Vy)
            Ux_y = slater_link(Vy, Vxy)
            Uy_x = slater_link(Vx, Vxy)

            total_flux += cmath.phase(Ux * Uy_x / (Ux_y * Uy))

    return total_flux / (2.0 * math.pi)


__all__ = [
    "dual_basis_vec_mat",
    "dual_basis_vec_list",
    "build_Hk_crys",
    "Chern_number_Fukui_Hatsugai_Suzuki",
    "generate_bilinear_terms",
    "build_real_space_tb_Hamiltonain",
    "many_body_Chern_number_Fukui_Hatsugai_Suzuki",
]
