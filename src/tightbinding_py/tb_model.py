"""Real-space tight-binding model data structures.

Faithful port of ``src/real_space_tb_model.jl`` from the Julia package
``TightBinding.jl``.

A :class:`Real_Space_TightBinding_Model` stores translation-invariant
hopping templates (:attr:`input_hopping_map`) together with their
translation-expanded, PBC-wrapped versions (:attr:`full_hopping_map`), plus
the underlying real-space lattice.
"""

from __future__ import annotations

import cmath
import os
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .lattice import LatticeGraph, Real_Space_Lattice, Site, _wrap_Δ_crys

#: A hopping term as passed to :func:`add_hopping_term`: a pair
#: ``((cell_from, sub_from), (cell_to, sub_to)), amplitude`` — the Python
#: analogue of Julia's ``Pair{Tuple{Site,Site},T}`` (``=>``).
HoppingTerm = tuple[tuple[Site, Site], complex]

#: A hopping key ``(cell_int, i_sub)`` pair — hashable (``cell_int`` is a
#: tuple of ints, ``i_sub`` is **1-based**).
SitePair = tuple[Site, Site]


@dataclass
class Real_Space_TightBinding_Model:
    """Real-space tight-binding model (mutable, as in the Julia source).

    Julia counterpart: ``mutable struct Real_Space_TightBinding_Model{T,U}``.

    Attributes:
        lattice: the underlying real-space lattice.
        model_name: name of the tight-binding model.
        input_hopping_map: hashmap ``(cell_int, i_sub) → t`` of the
            *translation-invariant hopping templates*.  This includes
            hoppings within and across unit cells.  Hermiticity is already
            implemented when building the map.
        full_hopping_map: hashmap ``(cell_int, i_sub) → t`` that includes
            and expands **all** hoppings of the model.  Translation
            symmetry is already implemented for bulk hopping terms (PBC
            wrapping included).
        H_hop: placeholder for the real-space hopping Hamiltonian (kept for
            parity with the Julia struct; use
            :func:`tightbinding_py.utils.build_real_space_tb_Hamiltonain`
            instead).
    """

    lattice: Real_Space_Lattice
    model_name: str
    input_hopping_map: dict[SitePair, complex] = field(default_factory=dict)
    full_hopping_map: dict[SitePair, complex] = field(default_factory=dict)
    H_hop: Any = field(default=None)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Real_Space_TightBinding_Model(model_name={self.model_name!r}, "
            f"lattice={self.lattice.lattice_name!r}, "
            f"n_input_hoppings={len(self.input_hopping_map)}, "
            f"n_full_hoppings={len(self.full_hopping_map)})"
        )


def initialize_real_space_tightbinding_model(
    lattice: Real_Space_Lattice, *, model_name: str = ""
) -> Real_Space_TightBinding_Model:
    """Constructor for :class:`Real_Space_TightBinding_Model`.

    Julia counterpart: ``initialize_real_space_tightbinding_model``.

    Args:
        lattice: the underlying real-space lattice.
        model_name: name of the tight-binding model.

    Returns:
        Real_Space_TightBinding_Model: the (empty) model.
    """
    if len(lattice.pbc_indicator) != lattice.dim:
        raise ValueError("length(pbc_indicator) must equal lattice.dim.")
    return Real_Space_TightBinding_Model(
        lattice=lattice,
        model_name=model_name,
        input_hopping_map={},
        full_hopping_map={},
        H_hop=None,
    )


def add_hopping_term(
    tb_model: Real_Space_TightBinding_Model,
    input_hopping_term: HoppingTerm,
    *,
    is_hermitian: bool = True,
) -> None:
    """Manually add a hopping term to the tight-binding model.

    Julia counterpart: ``add_hopping_term!``.

    Args:
        tb_model: the real-space tight-binding model to which the hopping
            term is added (mutated in place).
        input_hopping_term: the input hopping term of the form
            ``(((cell_from, sub_from), (cell_to, sub_to)), hopping_strength)``
            with **1-based** sublattice indices.  Note: it also applies to
            chemical potentials when ``cell_from == cell_to`` and
            ``sub_from == sub_to``.
        is_hermitian: whether to add the Hermitian conjugate of the input
            hopping term to the model.

    Raises:
        ValueError: if a sublattice index is out of range.
    """
    n_sub = tb_model.lattice.n_sub

    (site_from, site_to), hopping_strength = input_hopping_term
    (cell_from, sub_from) = site_from
    (cell_to, sub_to) = site_to
    cell_from = tuple(int(c) for c in cell_from)
    cell_to = tuple(int(c) for c in cell_to)

    # check the validity of the input hopping term
    if not (1 <= sub_from <= n_sub and 1 <= sub_to <= n_sub):
        raise ValueError(
            "The input sublattice indices for `input_hopping_map`="
            f"{input_hopping_term} are invalid for `sample_size`="
            f"{tb_model.lattice.sample_size}!"
        )

    key = ((cell_from, sub_from), (cell_to, sub_to))
    if key in tb_model.input_hopping_map:
        print(
            "WARNING: the input hopping term "
            f"`{input_hopping_term}` already exists in `input_hopping_map`!\n"
            " --- The old hopping term will be overwritten, rather than "
            "being added to the existing value!"
        )
    tb_model.input_hopping_map[key] = complex(hopping_strength)

    if is_hermitian:
        hc_key = ((cell_to, sub_to), (cell_from, sub_from))
        if hc_key in tb_model.input_hopping_map:
            tb_model.input_hopping_map[hc_key] += np.conj(hopping_strength)
        else:
            tb_model.input_hopping_map[hc_key] = np.conj(hopping_strength)

    # rebuild the `full_hopping_map` using translation symmetry
    cell_shift = tuple(c2 - c1 for c1, c2 in zip(cell_from, cell_to))
    for new_cell_from in tb_model.lattice.cell_int_list:
        new_cell_to = [
            new_cell_from[d] + cell_shift[d] for d in range(len(cell_shift))
        ]
        for d in range(len(tb_model.lattice.pbc_indicator)):
            if tb_model.lattice.pbc_indicator[d]:  # handle PBC
                new_cell_to[d] = new_cell_to[d] % tb_model.lattice.sample_size[d]
        new_cell_to = tuple(new_cell_to)

        new_site_from = (new_cell_from, sub_from)
        new_site_to = (new_cell_to, sub_to)

        fkey = (new_site_from, new_site_to)
        if fkey in tb_model.full_hopping_map:
            tb_model.full_hopping_map[fkey] += complex(hopping_strength)
        else:
            tb_model.full_hopping_map[fkey] = complex(hopping_strength)

        if is_hermitian:
            fkey_hc = (new_site_to, new_site_from)
            if fkey_hc in tb_model.full_hopping_map:
                tb_model.full_hopping_map[fkey_hc] += np.conj(hopping_strength)
            else:
                tb_model.full_hopping_map[fkey_hc] = np.conj(hopping_strength)
    return None


def add_hopping_term_to_full_hopping_map(
    tb_model: Real_Space_TightBinding_Model,
    hopping_term: HoppingTerm,
    *,
    is_hermitian: bool = True,
) -> None:
    """Add one hopping term by updating ``full_hopping_map`` (no expansion).

    Julia counterpart: ``add_hopping_term_to_full_hopping_map!``.

    Args:
        tb_model: the model to update (mutated in place).
        hopping_term: the hopping term of the form
            ``(((cell_from, sub_from), (cell_to, sub_to)), hopping_strength)``.
        is_hermitian: whether to add the Hermitian conjugate.

    Note:
        Unlike :func:`add_hopping_term`, the hopping is **not** expanded by
        translation symmetry — the pair is inserted as given.
    """
    (site_from, site_to), hopping_strength = hopping_term

    if (site_from, site_to) in tb_model.full_hopping_map:
        print(
            "INFO: the hopping term `{hopping_term}` already exists in "
            "`full_hopping_map`\n --- The new hopping term will be added to "
            "the existing value!"
        )
        tb_model.full_hopping_map[(site_from, site_to)] += complex(
            hopping_strength
        )
    else:
        tb_model.full_hopping_map[(site_from, site_to)] = complex(
            hopping_strength
        )

    if is_hermitian:
        if (site_to, site_from) in tb_model.full_hopping_map:
            tb_model.full_hopping_map[(site_to, site_from)] += np.conj(
                hopping_strength
            )
        else:
            tb_model.full_hopping_map[(site_to, site_from)] = np.conj(
                hopping_strength
            )
    return None


# ---------------------------------------------------------------------------
# Graph-distance-based hopping generation
# ---------------------------------------------------------------------------


def _require_graph(tb_model: Real_Space_TightBinding_Model) -> LatticeGraph:
    """Return the lattice graph or raise if unavailable.

    Args:
        tb_model: the tight-binding model.

    Returns:
        LatticeGraph: the lattice nearest-neighbor graph.

    Raises:
        ValueError: if the graph is ``None``.
    """
    g = tb_model.lattice.graph
    if g is None:
        raise ValueError(
            "The lattice graph is `None` — cannot traverse by graph "
            "distance. Ensure the lattice was built with numerical Bravais "
            "vectors."
        )
    return g


def add_hoppings_by_graph_distance(
    tb_model: Real_Space_TightBinding_Model,
    graph_distance: int,
    amplitude: complex | Callable[[int, int], complex],
    *,
    is_hermitian: bool = True,
) -> None:
    """Add hoppings between ALL pairs at a given graph distance.

    Julia counterpart: ``add_hoppings_by_graph_distance!`` (both methods:
    uniform complex ``amplitude`` and per-pair ``amplitude_func(i, j)``).

    - ``graph_distance = 0``: on-site terms (chemical potential);
    - ``graph_distance = 1``: nearest-neighbor hoppings (one graph edge);
    - ``graph_distance ≥ 2``: hoppings between sites connected by
      ``graph_distance`` graph edges.

    The graph must be available (``tb_model.lattice.graph`` is not
    ``None``).  Hoppings are added to ``full_hopping_map`` via
    :func:`add_hopping_term_to_full_hopping_map` (no translation
    expansion).

    Args:
        tb_model: the model to update (mutated in place).
        graph_distance: the graph distance of the pairs to connect.
        amplitude: either a complex number (uniform amplitude) or a
            callable ``amplitude(i_site, j_site) -> complex`` receiving two
            **1-based** linear site indices (direction-dependent
            amplitudes, e.g. the complex NNN hoppings of the Haldane
            model).  For the callable form, zero amplitudes are skipped.
        is_hermitian: whether to add the Hermitian conjugate.

    Examples::

        # Isotropic NN hopping on any lattice
        add_hoppings_by_graph_distance(tb_model, 1, -1.0)

        # On-site chemical potential
        add_hoppings_by_graph_distance(tb_model, 0, 0.5, is_hermitian=False)

        # Isotropic NNN hopping
        add_hoppings_by_graph_distance(tb_model, 2, -0.3)

    Raises:
        ValueError: if the graph is unavailable or the distance is negative.
    """
    g = _require_graph(tb_model)
    if graph_distance < 0:
        raise ValueError(f"Graph distance must be ≥ 0, got {graph_distance}.")

    n_site = tb_model.lattice.n_site
    is_callable = callable(amplitude)

    if graph_distance == 0:
        # on-site: each site to itself
        for i_site in range(1, n_site + 1):
            site = tb_model.lattice.site_list[i_site - 1]
            amp = amplitude(i_site, i_site) if is_callable else amplitude
            if amp != 0:
                add_hopping_term_to_full_hopping_map(
                    tb_model, ((site, site), amp), is_hermitian=False
                )
        return None

    for i_site in range(1, n_site + 1):
        site_i = tb_model.lattice.site_list[i_site - 1]
        dists = g.gdistances(i_site)
        for j_site in range(i_site + 1, n_site + 1):
            if dists.get(j_site) == graph_distance:
                amp = amplitude(i_site, j_site) if is_callable else amplitude
                if amp != 0:
                    site_j = tb_model.lattice.site_list[j_site - 1]
                    add_hopping_term_to_full_hopping_map(
                        tb_model, ((site_i, site_j), amp), is_hermitian=is_hermitian
                    )
    return None


def haldane_nnn_hopping_amplitude(
    i_site: int,
    j_site: int,
    *,
    t2: float = 0.3,
    φ: float = np.pi / 2,
    tb_model: Real_Space_TightBinding_Model,
) -> complex:
    """Complex NNN hopping amplitude of the Haldane model.

    Julia counterpart: ``haldane_nnn_hopping_amplitude``.  The strategy is
    to first search for the common nearest neighbor (which must be a single
    vertex for the honeycomb lattice), then compute the chirality
    :math:`\\nu = \\mathrm{sign}\\big[(\\mathbf r_k - \\mathbf r_i)\\times
    (\\mathbf r_j - \\mathbf r_k)\\big] = \\pm 1`.

    Args:
        i_site: **1-based** linear index of site i.
        j_site: **1-based** linear index of site j.
        t2: NNN hopping amplitude.
        φ: Haldane phase.
        tb_model: the tight-binding model (lattice must be the
            ``"honeycomb"`` preset with a numerical graph).

    Returns:
        complex: the complex NNN hopping amplitude
        :math:`t_2\\,e^{i\\varphi\\,\\nu}`.

    Raises:
        ValueError: if the lattice is not honeycomb, the graph is missing,
            or the two sites do not share exactly one common neighbor.
    """
    l = tb_model.lattice
    if l.lattice_name != "honeycomb":
        raise ValueError(
            "Haldane NNN hopping is defined for the honeycomb lattice only."
        )
    g = l.graph
    if g is None:
        raise ValueError(
            "The lattice graph is `None` — cannot compute Haldane NNN hopping."
        )

    i_site_nn_sites = set(g.neighbors(i_site))
    j_site_nn_sites = set(g.neighbors(j_site))
    common = i_site_nn_sites & j_site_nn_sites
    if len(common) != 1:
        raise ValueError(
            f"Sites {i_site} and {j_site} do not share exactly one common "
            "neighbor (graph distance ≠ 2?)."
        )
    k_site = next(iter(common))

    # cross product of the two NN hops determines the chirality
    site_i_crys = np.asarray(l.site_crys_list[i_site - 1], dtype=np.float64)
    site_j_crys = np.asarray(l.site_crys_list[j_site - 1], dtype=np.float64)
    site_k_crys = np.asarray(l.site_crys_list[k_site - 1], dtype=np.float64)

    Δ_ik_wrapped = _wrap_Δ_crys(
        site_k_crys - site_i_crys,
        sample_size=l.sample_size,
        pbc_indicator=l.pbc_indicator,
    )
    Δ_kj_wrapped = _wrap_Δ_crys(
        site_j_crys - site_k_crys,
        sample_size=l.sample_size,
        pbc_indicator=l.pbc_indicator,
    )

    brav_rows = np.asarray(l.brav_vec_list, dtype=np.float64)
    Δ_ik_cart = Δ_ik_wrapped @ brav_rows
    Δ_kj_cart = Δ_kj_wrapped @ brav_rows

    cross_2D = (
        Δ_ik_cart[0] * Δ_kj_cart[1] - Δ_ik_cart[1] * Δ_kj_cart[0]
    )
    chirality = 1 if cross_2D > 0 else -1

    return t2 * cmath.exp(chirality * 1j * φ)


def plot_real_space_tightbinding_model(tb_model: Real_Space_TightBinding_Model, **kwargs):
    """Plot hoppings and sites of the real-space tight-binding model.

    Julia counterpart: ``plot_real_space_tightbinding_model`` (CairoMakie;
    the Python port draws the same content with matplotlib).

    Draws the lattice background (sites + faint graph edges) and overlays
    hoppings as **curved annotated arcs** so they do not overlap with the
    background:

    - bulk hoppings: solid steelblue arcs with amplitude labels;
    - wrapped hoppings (across the PBC boundary): dashed tomato arcs with
      faded labels;
    - on-site terms are annotated as text next to the site marker.

    Args:
        tb_model: the model to visualize (2D lattice with numeric
            ``site_cart_list``).
        **kwargs: forwarded to matplotlib figure creation (e.g.
            ``save_path``).

    Returns:
        tuple: ``(fig, ax)``.

    Raises:
        ValueError: for non-2D or non-numeric lattices.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    l = tb_model.lattice
    if l.dim != 2:
        raise ValueError("Only 2D lattices are supported for plotting.")
    if not all(np.isrealobj(c) for c in l.site_crys_list):
        raise ValueError(
            "Cartesian coordinates must be numeric for plotting."
        )

    brav_vec = np.asarray(l.brav_vec_list, dtype=np.float64)
    site_crys = np.asarray(
        [c.astype(np.float64) for c in l.site_crys_list]
    )
    site_cart = np.asarray(
        [c.astype(np.float64) for c in l.site_cart_list]
    )

    def to_cart(c: np.ndarray) -> np.ndarray:
        return c @ brav_vec

    default_fig_size = [12.0, 12.0]
    scale = np.sqrt(np.prod(l.sample_size)) / 6.0
    fig, ax = plt.subplots(
        figsize=(default_fig_size[0] * scale, default_fig_size[1] * scale)
    )
    ax.set_aspect("equal")

    site_markersize = 20
    ghost_markersize = 20
    site_fontsize = 10
    ghost_fontsize = 10

    def _cycled(i_sub: int):
        palette = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["C0"])
        return palette[(i_sub - 1) % len(palette)]

    ghost_site_registry: dict = {}
    dist_cache: dict[int, dict[int, int]] = {}

    def get_dists(i_site: int) -> dict[int, int]:
        if i_site not in dist_cache:
            dist_cache[i_site] = (
                {}
                if l.graph is None
                else l.graph.gdistances(i_site)
            )
        return dist_cache[i_site]

    # 1. Draw lattice background edges --------------------------------------
    if l.graph is not None:
        for (i, j) in l.graph.edges():
            c_i = site_crys[i - 1]
            c_j = site_crys[j - 1]
            Δc_raw = c_j - c_i
            Δc_wrapped = _wrap_Δ_crys(
                Δc_raw.copy(),
                sample_size=l.sample_size,
                pbc_indicator=l.pbc_indicator,
            )
            is_wrapped = np.linalg.norm(Δc_wrapped - Δc_raw) > 1e-10

            if not is_wrapped:
                ax.plot(
                    [site_cart[i - 1][0], site_cart[j - 1][0]],
                    [site_cart[i - 1][1], site_cart[j - 1][1]],
                    color="black",
                    linewidth=2,
                )
            else:
                ghost_j_crys = c_i + Δc_wrapped
                ghost_i_crys = c_j - Δc_wrapped
                ghost_j_cart = to_cart(ghost_j_crys)
                ghost_i_cart = to_cart(ghost_i_crys)
                (_, i_sub_i) = l.site_list[i - 1]
                (_, i_sub_j) = l.site_list[j - 1]

                ax.plot(
                    [site_cart[i - 1][0], ghost_j_cart[0]],
                    [site_cart[i - 1][1], ghost_j_cart[1]],
                    color=_cycled(i_sub_j),
                    alpha=0.28,
                    linewidth=2,
                    linestyle="--",
                )
                ax.plot(
                    [site_cart[j - 1][0], ghost_i_cart[0]],
                    [site_cart[j - 1][1], ghost_i_cart[1]],
                    color=_cycled(i_sub_i),
                    alpha=0.28,
                    linewidth=2,
                    linestyle="--",
                )
                key_j = (j, tuple(np.round(ghost_j_crys, 10)))
                key_i = (i, tuple(np.round(ghost_i_crys, 10)))
                ghost_site_registry[key_j] = (j, i_sub_j, ghost_j_cart)
                ghost_site_registry[key_i] = (i, i_sub_i, ghost_i_cart)

    # 2. Draw hopping arcs ---------------------------------------------------
    for ((site_from, site_to), amplitude) in tb_model.full_hopping_map.items():
        i = l.site_to_index_map[site_from]
        j = l.site_to_index_map[site_to]

        p_i = site_cart[i - 1]
        p_j = site_cart[j - 1]

        if i == j:
            # on-site: self-loop arc
            scale_len = float(np.linalg.norm(brav_vec[0]))
            offset1 = np.asarray([scale_len * 0.5, scale_len * 0.5])
            ax.annotate(
                f"{amplitude:.3g}",
                xy=p_i + offset1,
                color="darkgreen",
                fontsize=8,
                ha="center",
                va="center",
            )
            continue

        c_i = site_crys[i - 1]
        c_j = site_crys[j - 1]
        Δc_raw = c_j - c_i
        Δc_wrapped = _wrap_Δ_crys(
            Δc_raw.copy(),
            sample_size=l.sample_size,
            pbc_indicator=l.pbc_indicator,
        )
        is_wrapped = np.linalg.norm(Δc_wrapped - Δc_raw) > 1e-10

        graph_d = get_dists(i).get(j, 0)
        linestyle = "-" if np.imag(amplitude) == 0 else ":"
        alpha = 0.28 if is_wrapped else 0.64

        if is_wrapped:
            ghost_crys = c_i + Δc_wrapped
            ghost_cart = to_cart(ghost_crys)
            p_target = ghost_cart
            (_, i_sub_j) = l.site_list[j - 1]
            key_j = (j, tuple(np.round(ghost_crys, 10)))
            ghost_site_registry[key_j] = (j, i_sub_j, ghost_cart)
            color = "tomato"
        else:
            p_target = p_j
            palette = plt.rcParams["axes.prop_cycle"].by_key().get(
                "color", ["C0"]
            )
            color = palette[graph_d % len(palette)] if graph_d else "steelblue"

        ax.add_patch(
            FancyArrowPatch(
                p_i,
                p_target,
                arrowstyle="-|>",
                mutation_scale=12,
                connectionstyle="arc3,rad=0.16",
                color=color,
                alpha=alpha,
                linewidth=1.0,
                linestyle=linestyle,
                shrinkA=3.0,
                shrinkB=3.0,
            )
        )
        mid = 0.5 * (p_i + p_target)
        ax.annotate(
            f"{amplitude:.3g}",
            xy=mid,
            color=color,
            alpha=min(1.0, alpha + 0.3),
            fontsize=7,
            ha="center",
            va="center",
        )

    # 3. Draw ghost sites ----------------------------------------------------
    for (_, (i_site, i_sub, ghost_cart)) in ghost_site_registry.items():
        ax.scatter(
            [ghost_cart[0]],
            [ghost_cart[1]],
            color=_cycled(i_sub),
            alpha=0.28,
            s=ghost_markersize,
        )
        ax.text(
            ghost_cart[0],
            ghost_cart[1],
            f"{i_site}",
            color="white",
            fontsize=ghost_fontsize,
            ha="center",
            va="center",
            alpha=0.75,
        )

    # 4. Draw physical bulk sites -------------------------------------------
    for i_site in range(1, l.n_site + 1):
        (_, i_sub) = l.site_list[i_site - 1]
        x, y = site_cart[i_site - 1]
        ax.scatter([x], [y], color=_cycled(i_sub), alpha=0.82, s=site_markersize)
        ax.text(
            x,
            y,
            f"{i_site}",
            color="white",
            fontsize=site_fontsize,
            ha="center",
            va="center",
        )

    # --- unit cell & bravais arrows ----------------------------------------
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
        save_path = os.fspath(save_path)
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    if kwargs:
        raise TypeError(f"unexpected keyword arguments: {sorted(kwargs)}")
    return fig, ax


__all__ = [
    "HoppingTerm",
    "SitePair",
    "Real_Space_TightBinding_Model",
    "initialize_real_space_tightbinding_model",
    "add_hopping_term",
    "add_hopping_term_to_full_hopping_map",
    "add_hoppings_by_graph_distance",
    "haldane_nnn_hopping_amplitude",
    "plot_real_space_tightbinding_model",
]
