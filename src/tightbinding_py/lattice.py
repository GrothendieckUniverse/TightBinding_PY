"""Real-space lattice data structures.

Faithful port of ``src/real_space_lattice.jl`` from the Julia package
``TightBinding.jl``.

A :class:`Real_Space_Lattice` describes a finite sample of a Bravais lattice
with an internal (sublattice) structure:

- ``sample_size`` unit cells along each Bravais vector, enumerated in
  *crystal coordinates* (first axis fastest, as in Julia's
  ``Iterators.product``);
- ``n_sub`` sublattices per unit cell at positions ``sub_crys_list``;
- optional periodic boundary conditions with twisted phases;
- the nearest-neighbor graph of the finite sample, built by comparing
  minimal Euclidean distances with the PBC minimum-image convention.

All site indices are **1-based**, exactly as in the Julia package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Hashable

import numpy as np

# ---------------------------------------------------------------------------
# Type aliases & the igraph-backed graph container
# ---------------------------------------------------------------------------

#: A site in the finite sample: ``(cell_int, i_sub)`` with ``cell_int`` a
#: tuple of integers (hashable — used as dict keys) and ``i_sub`` the
#: **1-based** sublattice index.  Julia counterpart: ``Site = Tuple{Vector{Int},Int}``.
Site = tuple[tuple[int, ...], int]

# ═════════════════════════════════════════════════════════════════════════════
# Implementation difference from the Julia source (documented):
#   Julia stores the nearest-neighbor graph as a `Graphs.SimpleGraph`.
#   Python stores it as an **igraph.Graph** — the igraph library has no
#   Julia equivalent and provides a far richer toolkit (community
#   detection, isomorphism, layout, export).  :class:`LatticeGraph` is a
#   thin adapter exposing the small Graphs.jl API subset used by this
#   package (``neighbors``, ``gdistances``, ``edges``) over the igraph
#   object, so all downstream code stays faithful to the Julia logic.
# ═════════════════════════════════════════════════════════════════════════════


class LatticeGraph:
    """Undirected nearest-neighbor graph of the finite sample, backed by igraph.

    Adapter over ``igraph.Graph`` mirroring the subset of the
    ``Graphs.SimpleGraph`` API used by the Julia package.  The igraph
    object itself is exposed as :attr:`igraph_graph` for advanced use
    (community detection, layouts, export — functionality with no Julia
    counterpart).

    Attributes:
        igraph_graph: the underlying ``igraph.Graph`` (0-based vertex ids,
            vertex attribute ``"site"`` = the ``Site`` tuple).
    """

    def __init__(self, n_site: int):
        import igraph as ig

        self.igraph_graph = ig.Graph(n=n_site)
        self.igraph_graph.vs["site"] = [None] * n_site

    def set_site(self, i_site: int, site: Site) -> None:
        """Attach the ``Site`` tuple to vertex ``i_site`` (1-based).

        Args:
            i_site: 1-based site index.
            site: the ``(cell_int, i_sub)`` tuple.
        """
        self.igraph_graph.vs[i_site - 1]["site"] = site

    def add_edge(self, i: int, j: int) -> None:
        """Add the undirected edge ``(i, j)`` (1-based endpoints).

        Args:
            i: first endpoint (1-based).
            j: second endpoint (1-based).
        """
        self.igraph_graph.add_edge(i - 1, j - 1)

    def neighbors(self, i: int) -> set[int]:
        """Neighbors of vertex ``i`` (1-based).

        Args:
            i: vertex index (1-based).

        Returns:
            set[int]: the 1-based neighbor indices.
        """
        return {v + 1 for v in self.igraph_graph.neighbors(i - 1)}

    def gdistances(self, source: int) -> dict[int, int]:
        """Graph (unweighted BFS) distance of every vertex from ``source``.

        Args:
            source: the source vertex (1-based).

        Returns:
            dict[int, int]: ``vertex → graph distance``
            (unreachable vertices are absent).
        """
        row = self.igraph_graph.distances(source=source - 1)[0]
        out = {}
        for v, d in enumerate(row):
            if d >= 0 and not np.isinf(d):
                out[v + 1] = int(d)
        return out

    def edges(self) -> list[tuple[int, int]]:
        """All undirected edges (1-based endpoints, ``i < j``).

        Returns:
            list[tuple[int, int]]: the edge list.
        """
        return [(u + 1, v + 1) for (u, v) in self.igraph_graph.get_edgelist()]

    def n_edges(self) -> int:
        """Number of undirected edges.

        Returns:
            int: the edge count.
        """
        return int(self.igraph_graph.ecount())

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"LatticeGraph(n_site={self.igraph_graph.vcount()}, "
            f"n_edges={self.n_edges()})"
        )


# ---------------------------------------------------------------------------
# Real_Space_Lattice
# ---------------------------------------------------------------------------


@dataclass
class Real_Space_Lattice:
    """Finite real-space lattice with PBC support and NN graph.

    Julia counterpart: ``mutable struct Real_Space_Lattice{T}``.

    Attributes:
        lattice_name: name of the lattice.
        dim: dimension of the lattice.
        sample_size: number of unit cells along each Bravais vector.
        cell_int_list: list of integer cell indices (crystal coordinates,
            first axis fastest).
        n_cell: number of unit cells.
        brav_vec_list: list of Bravais vectors for the real-space lattice.
        cell_volume: volume of the unit cell in real space
            (:math:`|\\det(\\mathrm{brav})|`).
        n_sub: number of sublattices in each unit cell.
        sub_crys_list: list of sublattice positions *in crystal
            coordinates*.
        sub_name_list: list of sublattice names (default ``"A1", "A2", …``).
        pbc_indicator: whether to apply periodic boundary conditions in
            direction ``i``.
        twisted_phases_over_2π: twisted phases :math:`\\varphi/(2\\pi)` along
            each periodic direction.  Non-zero values are only allowed where
            ``pbc_indicator[d]`` is true.  Inserting fluxes is equivalent to
            shifting the crystal-momentum grid by :math:`\\varphi/L`.
        n_site: total number of sites in the lattice.
        site_list: list of site positions in each cell as
            ``(cell_int, i_sub)`` (**1-based** sublattice index).
        site_crys_list: site positions in crystal coordinates (list of
            ``np.ndarray``).
        site_cart_list: site positions in Cartesian coordinates (list of
            ``np.ndarray``).
        site_to_index_map: hashmap ``(cell_int, i_sub) → i_site``
            (**1-based** site index).
        graph: undirected *nearest-neighbor graph* on the finite sample
            (:class:`LatticeGraph`), built by comparing minimal Euclidean
            distances with PBC.  ``None`` if symbolic entries prevent
            numerical construction.
    """

    lattice_name: str
    dim: int
    sample_size: list[int]

    cell_int_list: list[tuple[int, ...]]
    n_cell: int

    brav_vec_list: list[list[float]]
    cell_volume: float

    n_sub: int
    sub_crys_list: list[list[float]]
    sub_name_list: list[str]

    pbc_indicator: list[bool]
    twisted_phases_over_2π: list[float]

    n_site: int
    site_list: list[Site]
    site_crys_list: list[np.ndarray]
    site_cart_list: list[np.ndarray]
    site_to_index_map: dict[Site, int]

    graph: LatticeGraph | None = field(default=None)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Real_Space_Lattice(name={self.lattice_name!r}, dim={self.dim}, "
            f"sample_size={self.sample_size}, n_sub={self.n_sub}, "
            f"n_site={self.n_site})"
        )


#: Preset lattices overriding the explicit arguments when ``lattice_name``
#: matches exactly (Julia counterpart: the ``@match`` block in
#: ``initialize_real_space_lattice``).
_PRESET_LATTICES: dict[str, tuple[list[list[float]], list[list[float]], list[tuple[int, int]] | None]] = {
    "square": ([[1.0, 0.0], [0.0, 1.0]], [[0.0, 0.0]], None),
    "honeycomb": (
        [[1.0, 0.0], [1 / 2, np.sqrt(3) / 2]],
        [[0.0, 0.0], [1 / 3, 1 / 3]],
        None,
    ),
    "kagome": (
        [[1.0, 0.0], [1 / 2, np.sqrt(3) / 2]],
        [[0.0, 0.0], [1 / 2, 0], [0, 1 / 2]],
        None,
    ),
    "Lieb": (
        [[1.0, 0.0], [0.0, 1.0]],
        [[0.0, 0.0], [1 / 2, 0], [0, 1 / 2]],
        None,
    ),
    "dice": (
        [[1.0, 0.0], [1 / 2, np.sqrt(3) / 2]],
        [[0.0, 0.0], [1 / 3, 1 / 3], [2 / 3, 2 / 3]],
        [(1, 2), (2, 3)],
    ),
}


def initialize_real_space_lattice(
    *,
    brav_vec_list: list[list[float]] | None = None,
    sample_size: list[int] | None = None,
    sub_crys_list: list[list[float]] | None = None,
    lattice_name: str = "",
    pbc_indicator: list[bool] | None = None,
    twisted_phases_over_2π: list[float] | None = None,
    allowed_bonds: list[tuple[int, int]] | None = None,
) -> Real_Space_Lattice:
    """Constructor for :class:`Real_Space_Lattice`.

    Julia counterpart: ``initialize_real_space_lattice``.

    Args:
        brav_vec_list: list of Bravais vectors for the real-space lattice
            (default ``[[1.0, 0.0], [0.0, 1.0]]``).
        sample_size: number of unit cells in each direction
            (default ``[2, 2]``).
        sub_crys_list: list of sublattice positions *in crystal
            coordinates* (default ``[[0.0, 0.0]]``).
        lattice_name: name of the lattice.  If set to ``"square"``,
            ``"honeycomb"``, ``"kagome"``, ``"Lieb"``, or ``"dice"``
            (exact match), it overrides the above three arguments with the
            corresponding defaults (for ``"dice"`` also
            ``allowed_bonds = [(1, 2), (2, 3)]``).
        pbc_indicator: whether to apply periodic boundary conditions in
            direction ``i`` (default ``[True, True]``).
        twisted_phases_over_2π: twisted phases :math:`\\varphi/(2\\pi)` along
            each periodic direction.  Defaults to zeros.  Non-zero values
            are only allowed where ``pbc_indicator[d]`` is true.
        allowed_bonds: optional list of *allowed* sublattice pairs
            ``(sub_i, sub_j)`` for graph construction (**1-based** indices).
            When ``None`` (default), all pairs are allowed (original
            Euclidean-distance algorithm).  When provided, only edges
            between the specified sublattice pairs are kept; the pair is
            checked symmetrically.

    Returns:
        Real_Space_Lattice: the constructed lattice.

    Raises:
        ValueError: on inconsistent inputs (bad dimension, twisted phases on
            open directions, …).
    """
    if brav_vec_list is None:
        brav_vec_list = [[1.0, 0.0], [0.0, 1.0]]
    if sample_size is None:
        sample_size = [2, 2]
    if sub_crys_list is None:
        sub_crys_list = [[0.0, 0.0]]
    if pbc_indicator is None:
        pbc_indicator = [True, True]

    # lattice_name presets override the explicit arguments (exact match)
    if lattice_name in _PRESET_LATTICES:
        brav_vec_list, sub_crys_list, preset_bonds = _PRESET_LATTICES[lattice_name]
        if preset_bonds is not None:
            allowed_bonds = preset_bonds

    dim = len(brav_vec_list)

    if twisted_phases_over_2π is None:
        twisted_phases_over_2π = [0.0] * dim
    if len(twisted_phases_over_2π) != dim:
        raise ValueError(
            f"twisted_phases_over_2π must have length = dim = {dim}."
        )
    for d in range(dim):
        if not pbc_indicator[d] and twisted_phases_over_2π[d] != 0.0:
            raise ValueError(
                "Check input! Twisted phases are only allowed along "
                f"periodic directions: twisted_phases_over_2π[{d}] = "
                f"{twisted_phases_over_2π[d]} is non-zero, but "
                f"pbc_indicator[{d}] = False."
            )

    brav_vec_mat = np.asarray(brav_vec_list, dtype=np.float64).T  # columns
    cell_volume = float(abs(np.linalg.det(brav_vec_mat)))

    if dim not in (2, 3):
        raise ValueError(f"dim must be 2 or 3, got dim = {dim}.")
    if len(brav_vec_list) != len(sample_size):
        raise ValueError(
            "length(brav_vec_list) must equal length(sample_size)."
        )
    n_sub = len(sub_crys_list)
    if n_sub < 1:
        raise ValueError("at least one sublattice is required.")

    # cell enumeration: FIRST axis fastest (Julia Iterators.product order)
    ranges = [range(n) for n in sample_size]
    cell_int_list = [
        tuple(cell)
        for cell in product(*reversed(ranges))
    ]
    cell_int_list = [tuple(reversed(c)) for c in cell_int_list]
    n_cell = len(cell_int_list)

    if any(len(s) != dim for s in sub_crys_list):
        raise ValueError("every sublattice position must have length dim.")
    sub_name_list = [f"A{i}" for i in range(1, n_sub + 1)]  # the default names

    site_list: list[Site] = [
        (cell_int, i_sub)
        for cell_int in cell_int_list
        for i_sub in range(1, n_sub + 1)
    ]
    n_site = len(site_list)
    site_crys_list = [
        np.asarray(cell_int, dtype=np.float64)
        + np.asarray(sub_crys_list[i_sub - 1], dtype=np.float64)
        for (cell_int, i_sub) in site_list
    ]
    brav_rows = np.asarray(brav_vec_list, dtype=np.float64)
    site_cart_list = [crys @ brav_rows for crys in site_crys_list]

    # site_to_index_map is 1-based, exactly as in the Julia package
    site_to_index_map = {
        site: i_site for i_site, site in enumerate(site_list, start=1)
    }

    # Build the nearest-neighbor graph by comparing minimal Euclidean
    # distances.  Falls back to `None` for symbolic (non-numeric) inputs.
    graph: LatticeGraph | None
    try:
        graph = _build_nearest_neighbor_graph_by_Euclidean_distance(
            site_crys_list=site_crys_list,
            site_list=site_list,
            brav_vec_list=brav_vec_list,
            sample_size=sample_size,
            pbc_indicator=pbc_indicator,
            allowed_bonds=allowed_bonds,
        )
        if graph is not None:
            # attach the site tuples to the igraph vertices
            for i_site, site in enumerate(site_list, start=1):
                graph.set_site(i_site, site)
    except (TypeError, ValueError):
        graph = None

    return Real_Space_Lattice(
        lattice_name=lattice_name,
        dim=dim,
        sample_size=[int(n) for n in sample_size],
        cell_int_list=cell_int_list,
        n_cell=n_cell,
        brav_vec_list=[[float(x) for x in v] for v in brav_vec_list],
        cell_volume=cell_volume,
        n_sub=n_sub,
        sub_crys_list=[[float(x) for x in v] for v in sub_crys_list],
        sub_name_list=sub_name_list,
        pbc_indicator=[bool(b) for b in pbc_indicator],
        twisted_phases_over_2π=[float(x) for x in twisted_phases_over_2π],
        n_site=n_site,
        site_list=site_list,
        site_crys_list=site_crys_list,
        site_cart_list=site_cart_list,
        site_to_index_map=site_to_index_map,
        graph=graph,
    )


def _wrap_Δ_crys(
    Δ_crys: np.ndarray,
    *,
    sample_size: list[int],
    pbc_indicator: list[bool],
) -> np.ndarray:
    """Wrap a crystal displacement with respect to the boundary conditions.

    Julia counterpart: ``_wrap_Δ_crys!`` (**in-place**; the Python version
    returns a new array — pass a copy when the input must be preserved).

    The strategy is to apply periodic boundary conditions by wrapping the
    crystal displacement into the range ``[-L/2, L/2]`` for each direction
    where ``pbc_indicator`` is true.

    Args:
        Δ_crys: displacement in crystal coordinates.
        sample_size: vector of sample sizes.
        pbc_indicator: boundary-condition indicators.

    Returns:
        np.ndarray: the wrapped displacement.
    """
    out = Δ_crys.astype(np.float64).copy()
    for d in range(len(out)):  # loop over dimensions
        if pbc_indicator[d]:
            # wrap Δc to [-L/2, L/2] where L = sample_size[d]
            out[d] -= round(out[d] / sample_size[d]) * sample_size[d]
    return out


def _build_nearest_neighbor_graph_by_Euclidean_distance(
    *,
    site_crys_list: list[np.ndarray],
    site_list: list[Site],
    brav_vec_list: list[list[float]],
    sample_size: list[int],
    pbc_indicator: list[bool],
    allowed_bonds: list[tuple[int, int]] | None = None,
) -> LatticeGraph:
    """Build the nearest-neighbor graph by minimal Euclidean distance.

    Julia counterpart: ``_build_nearest_neighbor_graph_by_Euclidean_distance``.

    For each site pair we first search for the minimum distance between any
    two distinct sites (restricted to the allowed sublattice pairs), then
    connect pairs whose distance falls within that value.

    Args:
        site_crys_list: site crystal-coordinate list.
        site_list: site list with ``(cell_int, i_sub)`` for each site.
        brav_vec_list: Bravais vectors.
        sample_size: sample size.
        pbc_indicator: boundary-condition indicators.
        allowed_bonds: optional list of allowed sublattice pairs
            ``(sub_i, sub_j)`` (**1-based**).  When provided, only edges
            between the specified sublattice pairs are considered; the pair
            is checked symmetrically.  When ``None``, all pairs are allowed.

    Returns:
        LatticeGraph: the undirected nearest-neighbor graph (1-based
        vertices).
    """
    n_site = len(site_crys_list)

    # Build a set of allowed sublattice pairs (symmetric) for fast lookup
    allowed_pairs: set[tuple[int, int]] | None
    if allowed_bonds is None:
        allowed_pairs = None
    else:
        allowed_pairs = set()
        for (a, b) in allowed_bonds:
            allowed_pairs.add((a, b))
            allowed_pairs.add((b, a))

    def _is_pair_allowed(i: int, j: int) -> bool:
        if allowed_pairs is None:
            return True
        sub_i = site_list[i - 1][1]
        sub_j = site_list[j - 1][1]
        return (sub_i, sub_j) in allowed_pairs

    brav_rows = np.asarray(brav_vec_list, dtype=np.float64)

    # Pass 1: find the nearest-neighbor distance (only among allowed pairs)
    nn_dist = float("inf")
    for i in range(1, n_site + 1):
        for j in range(i + 1, n_site + 1):
            if not _is_pair_allowed(i, j):
                continue
            Δ_crys = site_crys_list[j - 1] - site_crys_list[i - 1]
            Δ_crys = _wrap_Δ_crys(
                Δ_crys, sample_size=sample_size, pbc_indicator=pbc_indicator
            )
            d = float(np.linalg.norm(Δ_crys @ brav_rows))
            if d != 0 and d < nn_dist:
                nn_dist = d

    # Pass 2: build the graph
    g = LatticeGraph(n_site)
    if np.isinf(nn_dist):
        return g  # no edges (e.g. single-site lattice)

    threshold = nn_dist * (1.0 + 1.0e-10)
    for i in range(1, n_site + 1):
        for j in range(i + 1, n_site + 1):
            if not _is_pair_allowed(i, j):
                continue
            Δ_crys = site_crys_list[j - 1] - site_crys_list[i - 1]
            Δ_crys = _wrap_Δ_crys(
                Δ_crys, sample_size=sample_size, pbc_indicator=pbc_indicator
            )
            d = float(np.linalg.norm(Δ_crys @ brav_rows))
            if d <= threshold:
                g.add_edge(i, j)
    return g


def plot_real_space_lattice(lattice: Real_Space_Lattice, **kwargs):
    """Plot the 2D real-space lattice from its nearest-neighbor graph.

    Julia counterpart: ``plot_real_space_lattice`` (CairoMakie; the Python
    port draws the same content with matplotlib).

    - **Bulk edges** (minimum-image displacement ≡ raw displacement): solid
      black.
    - **Wrapped edges** (PBC wrapping shortens the bond): dashed, 42%
      transparency, with ghost sites plotted at the *unwrapped* positions
      (outside the sample region) at 42% transparency.
    - Every site (bulk and ghost) is labelled with its linear index from
      ``site_list``.
    - The unit cell (parallelogram spanned by the Bravais vectors) is
      outlined, with arrowed Bravais basis vectors.

    Args:
        lattice: the real-space lattice (2D only).
        **kwargs: forwarded to matplotlib figure creation (e.g.
            ``save_path`` to persist the figure).

    Returns:
        tuple: ``(fig, ax)``.

    Raises:
        ValueError: for non-2D lattices.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    if lattice.dim != 2:
        raise ValueError(
            "plot_real_space_lattice currently only supports 2D lattices"
        )

    # --- convert everything to float64 -----------------------------------
    brav_vec = np.asarray(
        [[float(x) for x in v] for v in lattice.brav_vec_list],
        dtype=np.float64,
    )
    site_crys = np.asarray(
        [c.astype(np.float64) for c in lattice.site_crys_list],
        dtype=np.float64,
    )
    n_site = lattice.n_site
    L = np.asarray(lattice.sample_size, dtype=np.float64)

    def to_cart(c: np.ndarray) -> np.ndarray:
        return c @ brav_vec

    site_cart = np.asarray([to_cart(c) for c in site_crys])

    # --- figure & axis ----------------------------------------------------
    default_fig_size = [12.0, 12.0]
    scale = np.sqrt(np.prod(lattice.sample_size)) / 6.0
    fig, ax = plt.subplots(
        figsize=(default_fig_size[0] * scale, default_fig_size[1] * scale)
    )
    ax.set_aspect("equal")

    scaled_marker_size = 20
    scaled_font_size = 10

    def _cycled(i_sub: int):
        palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
        return palette[(i_sub - 1) % len(palette)]

    if lattice.graph is not None:
        g = lattice.graph
        alpha_wrap = 0.42

        for (i, j) in g.edges():
            Δc_raw = site_crys[j - 1] - site_crys[i - 1]
            Δc_min = _wrap_Δ_crys(
                Δc_raw.copy(),
                sample_size=lattice.sample_size,
                pbc_indicator=lattice.pbc_indicator,
            )

            if np.linalg.norm(Δc_min - Δc_raw) < 1e-10:
                # ---- bulk edge -------------------------------------------
                ax.plot(
                    [site_cart[i - 1][0], site_cart[j - 1][0]],
                    [site_cart[i - 1][1], site_cart[j - 1][1]],
                    color="black",
                    alpha=1.0,
                    linewidth=2,
                )
            else:
                # ---- wrapped edge: bulk ↔ ghost --------------------------
                ghost_j_crys = site_crys[i - 1] + Δc_min
                ghost_i_crys = site_crys[j - 1] - Δc_min
                ghost_j_cart = to_cart(ghost_j_crys)
                ghost_i_cart = to_cart(ghost_i_crys)

                i_sub_i = lattice.site_list[i - 1][1]
                i_sub_j = lattice.site_list[j - 1][1]
                color_i = _cycled(i_sub_i)
                color_j = _cycled(i_sub_j)

                ax.plot(
                    [site_cart[i - 1][0], ghost_j_cart[0]],
                    [site_cart[i - 1][1], ghost_j_cart[1]],
                    color=color_j,
                    alpha=alpha_wrap,
                    linewidth=2,
                    linestyle="--",
                )
                ax.plot(
                    [site_cart[j - 1][0], ghost_i_cart[0]],
                    [site_cart[j - 1][1], ghost_i_cart[1]],
                    color=color_i,
                    alpha=alpha_wrap,
                    linewidth=2,
                    linestyle="--",
                )

                ax.scatter(
                    [ghost_j_cart[0], ghost_i_cart[0]],
                    [ghost_j_cart[1], ghost_i_cart[1]],
                    color=[color_j, color_i],
                    alpha=alpha_wrap,
                    s=scaled_marker_size,
                )
                ax.text(
                    ghost_j_cart[0],
                    ghost_j_cart[1],
                    f"{j}",
                    color="white",
                    fontsize=scaled_font_size,
                    ha="center",
                    va="center",
                )
                ax.text(
                    ghost_i_cart[0],
                    ghost_i_cart[1],
                    f"{i}",
                    color="white",
                    fontsize=scaled_font_size,
                    ha="center",
                    va="center",
                )

    # --- plot bulk sites & labels ----------------------------------------
    for i_site in range(1, n_site + 1):
        (cell_int, i_sub) = lattice.site_list[i_site - 1]
        x, y = site_cart[i_site - 1]
        ax.scatter([x], [y], color=_cycled(i_sub), s=scaled_marker_size)
        ax.text(
            x,
            y,
            f"{i_site}",
            color="white",
            fontsize=scaled_font_size,
            ha="center",
            va="center",
        )

    # --- unit cell & bravais arrows ---------------------------------------
    a1 = brav_vec[0]
    a2 = brav_vec[1]
    origin = np.zeros(2)
    cell_cx = [origin[0], a1[0], a1[0] + a2[0], a2[0], origin[0]]
    cell_cy = [origin[1], a1[1], a1[1] + a2[1], a2[1], origin[1]]
    ax.plot(cell_cx, cell_cy, color="black", alpha=0.5, linewidth=2, linestyle="-.")
    for avec in (a1, a2):
        ax.add_patch(
            FancyArrowPatch(
                origin,
                origin + avec,
                arrowstyle="-|>",
                mutation_scale=18,
                color="tomato",
                alpha=0.64,
                linewidth=2.4,
            )
        )

    save_path = kwargs.pop("save_path", None)
    if save_path is not None:
        import os
        from pathlib import Path

        save_path = Path(save_path)
        os.makedirs(save_path.parent, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    if kwargs:
        raise TypeError(f"unexpected keyword arguments: {sorted(kwargs)}")
    return fig, ax


__all__ = [
    "Site",
    "LatticeGraph",
    "Real_Space_Lattice",
    "initialize_real_space_lattice",
    "plot_real_space_lattice",
]
