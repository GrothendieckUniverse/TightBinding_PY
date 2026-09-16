"""Kagome lattice with nearest-neighbour hopping — the flat band at E = +2.

Physics
-------
The kagome lattice (three sublattices on the triangular Bravais network) with
a uniform nearest-neighbour hopping ``t = -1`` hosts one **exactly flat** band
at energy ``E = +2``, sitting above two dispersive bands.  The flatness is not
a coincidence: it comes from a compact localised state — an alternating-sign
superposition on the six sites of a hexagon that cancels by destructive
interference on every site shared with a neighbouring hexagon.

The flat (highest) band touches the middle dispersive band at the zone centre
``Gamma`` (both at ``E = +2``), while the two lower bands touch at a **Dirac
point** at the zone corner ``K = (2/3, 1/3)`` (both at ``E = -1``).

This script builds the kagome lattice ``[3, 3]`` with PBC, adds the isotropic
nearest-neighbour hopping with :func:`add_hoppings_by_graph_distance`, plots
the band structure along ``Gamma-K-M-Gamma``, and verifies the flat band by
measuring the per-band energy spread over the whole Brillouin zone.

Run from the package root::

    .venv/bin/python examples/kagome_flat_band.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tightbinding_py import (
    add_hoppings_by_graph_distance,
    build_Hk_crys,
    initialize_real_space_lattice,
    initialize_real_space_tightbinding_model,
    initialize_uniform_grids_from_lattice,
    plot_bands,
    plot_real_space_lattice,
)

FIG_DIR = Path("figures") / "examples"


def main() -> None:
    lat = initialize_real_space_lattice(
        lattice_name="kagome", sample_size=[3, 3], pbc_indicator=[True, True]
    )
    tb = initialize_real_space_tightbinding_model(lat, model_name="kagome NN")

    # Isotropic nearest-neighbour hopping t = -1 on every graph edge.
    add_hoppings_by_graph_distance(tb, 1, -1.0)

    # --- real-space lattice figure ----------------------------------------
    plot_real_space_lattice(lat, save_path=FIG_DIR / "kagome_lattice.svg")

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
        save_path=FIG_DIR / "kagome_flat_band.svg",
    )

    # --- report the flat band and the Dirac point --------------------------
    for name, k in [("Γ", [0.0, 0.0]), ("K", [2 / 3, 1 / 3])]:
        ev = np.sort(np.linalg.eigvalsh(Hk_crys(np.asarray(k, dtype=float))))
        print(f"eigenvalues at {name} = {np.round(ev, 6)}")

    # Confirm the flat band by sampling energies across the whole BZ.
    ks = k_data.site_crys_list
    Es = np.array([np.linalg.eigvalsh(Hk_crys(k)) for k in ks])
    spread = np.ptp(Es, axis=0)
    print("per-band energy spread over the BZ:", np.round(spread, 10))
    print(f"flat band at E = {Es[0, 2]:.6f} (spread {spread[2]:.2e})")


if __name__ == "__main__":
    main()
