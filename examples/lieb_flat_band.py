"""Lieb lattice with nearest-neighbour hopping — the flat band at E = 0.

Physics
-------
The Lieb lattice (three sublattices on the square Bravais network, sublattice
``A1`` at the cell corner and ``A2``, ``A3`` at the edge midpoints) with a
uniform nearest-neighbour hopping ``t = -1`` hosts one **exactly flat** band
at energy ``E = 0``, flanked by two symmetric dispersive bands

    E(k) = ± |t| sqrt(4 + 2 cos(2 pi k1) + 2 cos(2 pi k2)).

The flat middle band is built from a compact localised state supported only on
the ``A2`` and ``A3`` sublattices, and it touches the dispersive bands at the
zone corner ``M = (1/2, 1/2)`` (all three bands at ``E = 0``).

This script builds the Lieb lattice ``[3, 3]`` with PBC, adds the isotropic
nearest-neighbour hopping with :func:`add_hoppings_by_graph_distance`, plots
the band structure along ``Gamma-X-M-Gamma``, and verifies the flat band by
measuring the per-band energy spread over the whole Brillouin zone.

Run from the package root::

    .venv/bin/python examples/lieb_flat_band.py
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
        lattice_name="Lieb", sample_size=[3, 3], pbc_indicator=[True, True]
    )
    tb = initialize_real_space_tightbinding_model(lat, model_name="Lieb NN")

    # Isotropic nearest-neighbour hopping t = -1 on every graph edge.
    add_hoppings_by_graph_distance(tb, 1, -1.0)

    # --- real-space lattice figure ----------------------------------------
    plot_real_space_lattice(lat, save_path=FIG_DIR / "lieb_lattice.svg")

    # --- k-space Hamiltonian & band structure ------------------------------
    Hk_crys = build_Hk_crys(tb)
    k_data = initialize_uniform_grids_from_lattice(lat)

    k_path = [[0.0, 0.0], [1 / 2, 0.0], [1 / 2, 1 / 2], [0.0, 0.0]]  # Γ-X-M-Γ
    plot_bands(
        Hk_crys,
        k_data,
        k_path=k_path,
        k_path_name_list=["Γ", "X", "M", "Γ"],
        nband_range=range(1, lat.n_sub + 1),
        nk=30,
        save_path=FIG_DIR / "lieb_flat_band.svg",
    )

    # --- report the flat band and the touching point ------------------------
    for name, k in [("Γ", [0.0, 0.0]), ("M", [1 / 2, 1 / 2])]:
        ev = np.sort(np.linalg.eigvalsh(Hk_crys(np.asarray(k, dtype=float))))
        print(f"eigenvalues at {name} = {np.round(ev, 6)}")

    # Confirm the flat (middle) band by sampling energies across the whole BZ.
    ks = k_data.site_crys_list
    Es = np.array([np.linalg.eigvalsh(Hk_crys(k)) for k in ks])
    spread = np.ptp(Es, axis=0)
    print("per-band energy spread over the BZ:", np.round(spread, 10))
    print(f"flat band at E = {Es[0, 1]:.6f} (spread {spread[1]:.2e})")


if __name__ == "__main__":
    main()
