"""Band-structure and band-contour plotting.

Faithful port of ``src/band_plot.jl`` from the Julia package
``TightBinding.jl``.  (The Julia module exports ``plot_band_counter`` — a
typo — while defining :func:`plot_band_contour`; the Python port provides
the actual function :func:`plot_band_contour`.)
"""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np

from .uniform_grids import Uniform_Grids


def plot_bands(
    Hk_crys: Callable[[np.ndarray], np.ndarray],
    k_data: Uniform_Grids,
    *,
    k_path: list[list[float]],
    k_path_name_list: list[str] | None = None,
    nband_range: Sequence[int] | range = range(1, 2),
    nk: int = 30,
    save_path: str | None = None,
):
    """Plot a band structure along a k-path with automatic x-scaling.

    Julia counterpart: ``plot_bands``.  The x-coordinates of the k-points
    are scaled by the Cartesian length of each k-path segment.

    Args:
        Hk_crys: the k-space Hamiltonian function with ``k_crys`` input.
        k_data: the k-space uniform grid (defines the basis vectors).
        k_path: a list of turning k-points in crystal coordinates.
        k_path_name_list: names of the turning points in ``k_path``
            (default: ``"A_1", "A_2", …``).
        nband_range: the range or list of band indices to plot
            (**1-based**; default ``range(1, 2)`` = lowest band only).
        nk: number of k-points per path segment.
        save_path: optional path to save the figure.

    Returns:
        tuple: ``(fig, ax)``.

    Raises:
        ValueError: on inconsistent inputs or a non-Hermitian ``H(k)``.
    """
    import matplotlib.pyplot as plt

    dim = k_data.dim
    if len(k_path) < 2:
        raise ValueError(
            "The input `k_path` must contain at least two k-points to form "
            "a path!"
        )
    if any(len(k_crys) != dim for k_crys in k_path):
        raise ValueError(
            "Every k-point in `k_path` must have the same dimension as the "
            "k-space lattice!"
        )
    if k_path_name_list is None:
        k_path_name_list = []
    if k_path_name_list and len(k_path_name_list) != len(k_path):
        raise ValueError(
            "Every k-point in `k_path` must have a corresponding name in "
            "`k_path_name_list`!"
        )

    k_path_f = [[float(x) for x in k] for k in k_path]

    # prepare the k-point list for the band plot
    k_crys_list: list[np.ndarray] = []
    vline_pos_list: list[float] = [0.0]
    k_point_xs: list[float] = []

    basis_rows = np.asarray(k_data.basis_vec_list, dtype=np.float64)
    for k_path_id in range(len(k_path_f) - 1):
        k_head_crys = np.asarray(k_path_f[k_path_id], dtype=np.float64)
        k_tail_crys = np.asarray(k_path_f[k_path_id + 1], dtype=np.float64)

        k_head_cart = k_head_crys @ basis_rows
        k_tail_cart = k_tail_crys @ basis_rows
        δk_cart = float(np.linalg.norm(k_tail_cart - k_head_cart)) / nk
        for i in range(nk):
            k_crys_list.append(
                k_head_crys + (k_tail_crys - k_head_crys) * (i / nk)
            )
            k_point_xs.append(vline_pos_list[-1] + δk_cart * i)
        vline_pos_list.append(vline_pos_list[-1] + δk_cart * nk)

    fig, ax = plt.subplots(figsize=(4.0, 4.0))

    vline_ticks = (
        (vline_pos_list, k_path_name_list)
        if k_path_name_list
        else (vline_pos_list, [f"A_{i}" for i in range(len(vline_pos_list))])
    )
    ax.set_xticks(vline_ticks[0])
    ax.set_xticklabels(vline_ticks[1])

    # Implementation difference from the Julia source (documented): Julia
    # draws one `scatter!` per (k, band) point; the Python port collects
    # all eigenvalues into arrays and draws one `ax.plot` (line + markers)
    # per band — identical content, but orders of magnitude faster with
    # matplotlib's artist model.
    nband_list = sorted(set(nband_range))
    eig_by_band = {b: [] for b in nband_list}
    for k_crys in k_crys_list:
        Hk_mat = np.asarray(Hk_crys(k_crys))
        if any(band_index > Hk_mat.shape[0] for band_index in nband_range):
            raise ValueError(
                "A requested band index exceeds the Hamiltonian size."
            )
        if np.linalg.norm(Hk_mat - (Hk_mat + Hk_mat.conj().T) / 2.0) > 1.0e-8:
            raise ValueError(
                f"The k-space Hamiltonian is not Hermitian at `k_crys={k_crys}`!"
            )
        eig_vals = np.linalg.eigvalsh((Hk_mat + Hk_mat.conj().T) / 2.0)
        for i_band in nband_list:
            eig_by_band[i_band].append(eig_vals[i_band - 1])

    for i_band in nband_list:
        ax.plot(
            k_point_xs,
            eig_by_band[i_band],
            "-o",
            markersize=3,
            linewidth=1.0,
            color=f"C{i_band - 1}",
        )

    for vline_pos in vline_pos_list:
        ax.axvline(vline_pos, color="black", alpha=0.3, linewidth=2)

    ax.set_xlabel("k path")
    ax.set_ylabel("Energy")
    fig.tight_layout()

    if save_path is not None:
        import os

        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax


def find_1st_BZ_k_cart_list(
    reciprocal_vec_list: list[list[float]], *, max_shell: int = 3
) -> list[list[float]]:
    """Cartesian coordinates of the 1st-BZ vertices (half-plane method).

    Julia counterpart: ``find_1st_BZ_k_cart_list``.  Grows shells of
    reciprocal lattice points until the perpendicular bisectors of the
    shortest non-zero vectors form a valid polygon; the vertices are the
    pairwise intersections satisfying all half-plane constraints, sorted
    counter-clockwise.

    Args:
        reciprocal_vec_list: the reciprocal basis vectors (2D).
        max_shell: maximum shell index to try.

    Returns:
        list[list[float]]: the 1st-BZ vertices in Cartesian coordinates
        (empty if not found).
    """
    (b1, b2) = reciprocal_vec_list
    err_tol = 1.0e-8

    for shell in range(1, max_shell + 1):
        Gs = []
        for n1 in range(-shell, shell + 1):
            for n2 in range(-shell, shell + 1):
                if n1 == 0 and n2 == 0:
                    continue
                Gs.append(n1 * np.asarray(b1) + n2 * np.asarray(b2))

        norms = [float(np.linalg.norm(G)) for G in Gs]
        minnorm = min(norms)
        nn_Gs = [G for (G, n) in zip(Gs, norms) if n <= minnorm * (1 + 1e-8)]

        ns = [np.asarray(G, dtype=np.float64) for G in nn_Gs]
        ds = [float(np.dot(n, n)) / 2.0 for n in ns]

        vertices_k_cart_list: list[list[float]] = []
        for i in range(len(ns) - 1):
            n1v = ns[i]
            d1 = ds[i]
            for j in range(i + 1, len(ns)):
                n2v = ns[j]
                d2 = ds[j]
                A = np.asarray([[n1v[0], n1v[1]], [n2v[0], n2v[1]]])
                if abs(np.linalg.det(A)) < err_tol:
                    continue
                x = np.linalg.solve(A, np.asarray([d1, d2]))
                # Must lie within all half-planes
                if all(
                    float(np.dot(nv, x)) <= dv + err_tol
                    for (nv, dv) in zip(ns, ds)
                ):
                    # Uniqueness filter
                    if all(
                        float(np.linalg.norm(x - np.asarray(v))) > err_tol
                        for v in vertices_k_cart_list
                    ):
                        vertices_k_cart_list.append([float(c) for c in x])

        if len(vertices_k_cart_list) >= 3:
            angles = [
                math.atan2(v[1], v[0]) for v in vertices_k_cart_list
            ]
            order = sorted(range(len(vertices_k_cart_list)), key=lambda k: angles[k])
            return [vertices_k_cart_list[k] for k in order]
    return []


def plot_band_contour(
    hk_cart: Callable[[np.ndarray], np.ndarray],
    k_data: Uniform_Grids,
    *,
    k_cart_ranges: list[np.ndarray] | None = None,
    levels: int = 10,
    band_idx: int = 1,
    normalize_k_cart_range_with_lattice_constant: bool = True,
    show_BZ: bool = True,
    show_band_width_band_gap_info: bool = True,
    save_path: str | None = None,
):
    """Plot a band contour for 2D systems.

    Julia counterpart: ``plot_band_contour``.

    Args:
        hk_cart: the k-space Hamiltonian function with ``k_cart`` input.
        k_data: the k-space uniform grid.
        k_cart_ranges: the kx and ky ranges for the contour plot
            (default ``[-1.5π, 1.5π]`` with step ``0.05`` in both
            directions).
        levels: number of contour levels.
        band_idx: the band index to plot (**1-based**).
        normalize_k_cart_range_with_lattice_constant: whether to normalize
            the k ranges by the lattice constant (‖a₁‖).
        show_BZ: whether to show the 1st Brillouin-zone boundary.
        show_band_width_band_gap_info: whether to print band-width and
            band-gap information.
        save_path: optional path to save the figure.

    Returns:
        tuple: ``(fig, ax)``.

    Raises:
        ValueError: for non-2D grids or inconsistent inputs.
    """
    import matplotlib.pyplot as plt

    if k_data.dim != 2:
        raise ValueError("Only 2D systems are supported for band contour plot!")

    if k_cart_ranges is None:
        k_cart_ranges = [
            np.arange(-1.5 * math.pi, 1.5 * math.pi, 0.05),
            np.arange(-1.5 * math.pi, 1.5 * math.pi, 0.05),
        ]
    if len(k_cart_ranges) != 2:
        raise ValueError(
            "The length of `k_cart_ranges` must be 2 for 2D systems!"
        )

    if normalize_k_cart_range_with_lattice_constant:
        brav_vec_list = _dual_basis_vec_list_py(k_data.basis_vec_list)
        a = float(np.linalg.norm(brav_vec_list[0]))
        if not math.isclose(a, 1.0):
            k_cart_ranges = [r / a for r in k_cart_ranges]
            print(
                f"INFO: Normalized `k_cart_range` by the lattice constant "
                f"a={a}. New ranges: {[list(r) for r in k_cart_ranges]}"
            )
    (kx_range, ky_range) = k_cart_ranges

    n_kx = len(kx_range)
    n_ky = len(ky_range)

    nband = np.asarray(hk_cart([0.1, 0.2])).shape[0]
    energy_spec = []
    for n in range(nband):
        eigvals_mat = np.zeros((n_kx, n_ky))
        for (i_kx, kx) in enumerate(kx_range):
            for (i_ky, ky) in enumerate(ky_range):
                k_cart = np.asarray([kx, ky])
                Hk_mat = np.asarray(hk_cart(k_cart))
                if np.linalg.norm(Hk_mat - Hk_mat.conj().T) > 1.0e-10:
                    raise ValueError(
                        f"The Hamiltonian matrix is not Hermitian at "
                        f"k-point {list(k_cart)}!"
                    )
                eigvals_mat[i_kx, i_ky] = np.linalg.eigvalsh(
                    (Hk_mat + Hk_mat.conj().T) / 2.0
                )[n]
        energy_spec.append(eigvals_mat)

    if show_band_width_band_gap_info:
        for n in range(nband - 1):
            current_band_min = np.min(energy_spec[n])
            current_band_max = np.max(energy_spec[n])
            next_band_min = np.min(energy_spec[n + 1])
            next_band_max = np.max(energy_spec[n + 1])
            print(
                f"band {n + 1} width: {current_band_max - current_band_min},"
                f"\t band gap to band {n + 2}: "
                f"{next_band_min - current_band_max}"
            )

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.set_xlabel("kx")
    ax.set_ylabel("ky")
    p = ax.contourf(
        kx_range, ky_range, energy_spec[band_idx - 1].T, levels=levels, cmap="viridis"
    )
    fig.colorbar(p, ax=ax)
    ax.set_aspect("equal")

    if show_BZ:
        # 1st BZ (Wigner–Seitz cell in k-space) via perpendicular bisectors
        b1, b2 = k_data.basis_vec_list
        if not (len(b1) == 2 and len(b2) == 2):
            raise ValueError("Only 2D BZ plotting is supported.")

        bz_vertices = find_1st_BZ_k_cart_list(
            k_data.basis_vec_list, max_shell=2
        )
        if not bz_vertices:
            raise ValueError(
                "Failed to compute BZ vertices! Check the reciprocal "
                "lattice vectors."
            )

        poly_x = [v[0] for v in bz_vertices] + [bz_vertices[0][0]]
        poly_y = [v[1] for v in bz_vertices] + [bz_vertices[0][1]]
        ax.plot(poly_x, poly_y, color="black", linewidth=2)

    if save_path is not None:
        import os

        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    return fig, ax


def _dual_basis_vec_list_py(
    basis_vec_list: list[list[float]],
) -> list[np.ndarray]:
    """Dual basis vectors of a basis-vector list (2π-inverse-transpose).

    Local helper equivalent to ``TightBinding.dual_basis_vec_list`` (kept
    private here to avoid a circular import with :mod:`utils`).

    Args:
        basis_vec_list: basis vectors.

    Returns:
        list[np.ndarray]: the dual vectors.
    """
    basis_vec_mat = np.asarray(basis_vec_list, dtype=np.float64).T
    dual = 2.0 * math.pi * np.linalg.inv(basis_vec_mat).T
    return [dual[:, i] for i in range(dual.shape[1])]


__all__ = [
    "plot_bands",
    "plot_band_contour",
    "find_1st_BZ_k_cart_list",
]
