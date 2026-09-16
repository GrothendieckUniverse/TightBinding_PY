"""Fluxed, anisotropic dice (T3) lattice — a three-orbital example.

Physics
-------
The dice (T3) lattice is the honeycomb lattice with an extra *hub* sublattice
at the centre of each hexagon.  Its unit cell holds three sites: the hub
(``A1`` at the origin) and two rim sites ``A2``, ``A3`` at ``(1/3, 1/3)`` and
``(2/3, 2/3)`` (crystal coordinates).  The preset ``lattice_name="dice"``
ships the ``allowed_bonds=[(1, 2), (2, 3)]`` connectivity, so the graph links
the hub to the rim and the rim sites to each other.

This example attaches two ingredients (parameters preserved from the original
example, which was inspired by arXiv:2505.09009):

1. **flux** — the hub–rim (``1,2``) nearest-neighbour bonds carry Peierls
   phases ``± φ/3`` (``φ = 3 * flux_phase``), i.e. a complex hopping
   ``-t exp(± i φ/3)`` threading a flux through the hexagonal plaquettes;
2. **anisotropy** — the in-cell rim–ring (``2,3``) bond is rescaled by
   ``eta``, so the intra-cell and inter-cell rim hoppings differ.

A hub on-site detuning ``hub_detuning`` is added to break the particle-hole
symmetry of the rim sublattices.  The resulting three-band spectrum is gapped
and dispersive; this is a compact illustration of complex, direction-dependent
hoppings rather than a flat-band model.

Run from the package root::

    .venv/bin/python examples/dice_fluxed_anisotropic.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tightbinding_py import (
    add_hopping_term,
    build_Hk_crys,
    initialize_real_space_lattice,
    initialize_real_space_tightbinding_model,
    initialize_uniform_grids_from_lattice,
    plot_bands,
    plot_real_space_lattice,
)

FIG_DIR = Path("figures") / "examples"


def build_fluxed_anisotropic_dice(
    t: float = 1.0,
    eta: float = 0.7,
    flux_phase: float = 0.35,
    hub_detuning: float = 3.0,
):
    """Build the fluxed, anisotropic dice model on a PBC dice lattice.

    Args:
        t: overall hopping amplitude.
        eta: anisotropy of the in-cell rim–ring bond (``eta < 1`` weakens it).
        flux_phase: Peierls phase ``φ/3`` on the hub–rim bonds.
        hub_detuning: on-site potential on the hub sublattice.

    Returns:
        Real_Space_TightBinding_Model: the assembled model.
    """
    # The dice preset supplies bravais vectors, sublattices and the
    # allowed_bonds=[(1,2),(2,3)] connectivity (hub-rim and rim-ring).
    lat = initialize_real_space_lattice(
        lattice_name="dice", sample_size=[3, 3], pbc_indicator=[True, True]
    )
    tb = initialize_real_space_tightbinding_model(
        lat, model_name="fluxed anisotropic dice"
    )

    # --- hub on-site detuning ---------------------------------------------
    add_hopping_term(tb, ((((0, 0), 1), ((0, 0), 1)), hub_detuning), is_hermitian=False)

    # --- hub-rim (1,2) nearest-neighbour bonds with flux φ/3 --------------
    # The hub at (0,0) couples to rim site A2 in cells (0,0), (-1,0), (0,-1).
    # Alternating Peierls phases thread a flux around each hexagonal plaquette.
    add_hopping_term(tb, ((((0, 0), 1), ((0, 0), 2)), -t * np.exp(1j * flux_phase)))
    add_hopping_term(tb, ((((0, 0), 1), ((-1, 0), 2)), -t * np.exp(-1j * flux_phase)))
    add_hopping_term(tb, ((((0, 0), 1), ((0, -1), 2)), -t * np.exp(-1j * flux_phase)))

    # --- rim-ring (2,3) bonds with anisotropy -----------------------------
    # Rim A2 at (0,0) couples to rim A3 in cells (0,0), (-1,0), (0,-1).
    # The in-cell bond is rescaled by eta (anisotropic), the rest by 1.
    add_hopping_term(tb, ((((0, 0), 2), ((0, 0), 3)), -t * eta))
    add_hopping_term(tb, ((((0, 0), 2), ((-1, 0), 3)), -t))
    add_hopping_term(tb, ((((0, 0), 2), ((0, -1), 3)), -t))

    return tb


def main() -> None:
    tb = build_fluxed_anisotropic_dice(
        t=1.0, eta=0.7, flux_phase=0.35, hub_detuning=3.0
    )
    lat = tb.lattice

    # --- real-space lattice figure ----------------------------------------
    plot_real_space_lattice(lat, save_path=FIG_DIR / "dice_lattice.svg")

    # --- k-space Hamiltonian & band structure ------------------------------
    Hk_crys = build_Hk_crys(tb)
    k_data = initialize_uniform_grids_from_lattice(lat)

    k_path = [[0.0, 0.0], [2 / 3, 1 / 3], [1 / 2, 1 / 2], [0.0, 0.0]]  # Γ-K-M-Γ
    plot_bands(
        Hk_crys,
        k_data,
        k_path=k_path,
        k_path_name_list=["Γ", "K", "M", "Γ"],
        nband_range=range(1, lat.n_sub + 1),
        nk=30,
        save_path=FIG_DIR / "dice_fluxed_anisotropic.svg",
    )

    # --- report a few band energies ----------------------------------------
    for name, k in [("Γ", [0.0, 0.0]), ("K", [2 / 3, 1 / 3])]:
        ev = np.sort(np.linalg.eigvalsh(Hk_crys(np.asarray(k, dtype=float))))
        print(f"eigenvalues at {name} = {np.round(ev, 6)}")


if __name__ == "__main__":
    main()
