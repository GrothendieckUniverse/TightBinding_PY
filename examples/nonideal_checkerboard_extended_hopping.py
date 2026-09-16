"""Checkerboard Chern insulator (Sun–Gu–Katsura–Sarma) — detuned (non-ideal).

Physics
-------
Same two-orbital checkerboard model as ``ideal_checkerboard_extended_hopping.py``
— complex nearest-neighbour hoppings threading a flux through each square
plaquette, plus real next-nearest-neighbour (NNN) and next-next-nearest-
neighbour (NNNN) hoppings — but with the NNN amplitudes **detuned** away from
the analytic flat-band values:

    t       = -1
    t1'     = -1.4 / (2 + sqrt(2))      (ideal: -1 / (2 + sqrt(2)))
    t2'     = +1.4 / (2 + sqrt(2))      (ideal: +1 / (2 + sqrt(2)))
    t''     = -1 / (2 + 2 sqrt(2))
    phi/2pi = 1/8

The 1.4× detuning broadens the lower band (it is no longer nearly flat) while
the model stays in the same topological phase: the lower band keeps Chern
number ``C = -1`` and the upper band ``C = +1``, and the band gap widens.

``lattice_name="checkerboard"`` is **not** a preset, so the Bravais vectors
and sublattice positions are passed explicitly to
:func:`initialize_real_space_lattice`.

Run from the package root::

    .venv/bin/python examples/nonideal_checkerboard_extended_hopping.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tightbinding_py import (
    Chern_number_Fukui_Hatsugai_Suzuki,
    add_hopping_term,
    build_Hk_crys,
    initialize_real_space_lattice,
    initialize_real_space_tightbinding_model,
    initialize_uniform_grids_from_lattice,
    plot_bands,
)

FIG_DIR = Path("figures") / "examples"

# Detuned (non-ideal) parameters: NNN amplitudes scaled by 1.4.
PARAMS_NONIDEAL = {
    "t": -1.0,
    "t1_prime": -1.4 / (2.0 + np.sqrt(2.0)),
    "t2_prime": 1.4 / (2.0 + np.sqrt(2.0)),
    "t_double_prime": -1.0 / (2.0 + 2.0 * np.sqrt(2.0)),
    "phi_over_2pi": 1.0 / 8.0,
}


def build_checkerboard_model(params: dict) -> object:
    """Build the SGKS checkerboard model from a parameter dictionary."""
    p = dict(PARAMS_NONIDEAL)
    p.update(params)

    t = p["t"]
    t1_prime = p["t1_prime"]
    t2_prime = p["t2_prime"]
    t_double_prime = p["t_double_prime"]
    phi = 2.0 * np.pi * p["phi_over_2pi"]

    # Explicit Bravais vectors + sublattices (checkerboard is NOT a preset).
    lat = initialize_real_space_lattice(
        brav_vec_list=[[1.0, 0.0], [0.0, 1.0]],
        sample_size=[3, 3],
        sub_crys_list=[[0.5, 0.0], [0.0, 0.5]],
        lattice_name="checkerboard",
        pbc_indicator=[True, True],
    )
    tb = initialize_real_space_tightbinding_model(
        lat, model_name="checkerboard_sgks"
    )

    # --- complex nearest-neighbour hoppings (flux through plaquettes) ------
    add_hopping_term(tb, ((((0, 0), 1), ((0, 0), 2)), -t * np.exp(-1j * phi)))
    add_hopping_term(tb, ((((0, 0), 1), ((1, 0), 2)), -t * np.exp(1j * phi)))
    add_hopping_term(tb, ((((0, 0), 2), ((0, 1), 1)), -t * np.exp(-1j * phi)))
    add_hopping_term(tb, ((((0, 0), 2), ((-1, 1), 1)), -t * np.exp(1j * phi)))

    # --- real next-nearest-neighbour hoppings (anisotropic) -----------------
    add_hopping_term(tb, ((((0, 0), 1), ((1, 0), 1)), -t1_prime))
    add_hopping_term(tb, ((((0, 0), 1), ((0, 1), 1)), -t2_prime))
    add_hopping_term(tb, ((((0, 0), 2), ((1, 0), 2)), -t2_prime))
    add_hopping_term(tb, ((((0, 0), 2), ((0, 1), 2)), -t1_prime))

    # --- real next-next-nearest-neighbour hoppings --------------------------
    add_hopping_term(tb, ((((0, 0), 1), ((1, 1), 1)), -t_double_prime))
    add_hopping_term(tb, ((((0, 0), 2), ((1, 1), 2)), -t_double_prime))
    add_hopping_term(tb, ((((0, 0), 2), ((-1, 1), 2)), -t_double_prime))
    add_hopping_term(tb, ((((0, 0), 1), ((1, -1), 1)), -t_double_prime))

    return tb


def report(tb, nk: int = 61) -> None:
    """Print band-gap and Chern-number information for a checkerboard model."""
    Hk_crys = build_Hk_crys(tb)

    # Band gap = min(second band) - max(first band) over a k-mesh.
    ks = [np.asarray([i / nk, j / nk], dtype=float) for i in range(nk) for j in range(nk)]
    Es = np.array([np.linalg.eigvalsh(Hk_crys(k)) for k in ks])
    E1, E2 = Es[:, 0], Es[:, 1]
    gap = E2.min() - E1.max()

    C1 = Chern_number_Fukui_Hatsugai_Suzuki(Hk_crys, band=1, nk=51)
    C2 = Chern_number_Fukui_Hatsugai_Suzuki(Hk_crys, band=2, nk=51)

    print(f"lower band: E in [{E1.min():.6f}, {E1.max():.6f}]  (width {E1.max() - E1.min():.6f})")
    print(f"band gap  : {gap:.6f}")
    print(f"Chern numbers: C_band1 = {C1:.6f},  C_band2 = {C2:.6f}")


def main() -> None:
    tb = build_checkerboard_model({})
    lat = tb.lattice

    Hk_crys = build_Hk_crys(tb)
    k_data = initialize_uniform_grids_from_lattice(lat)

    # Band structure along Γ-X-M-Γ.
    k_path = [[0.0, 0.0], [1 / 2, 0.0], [1 / 2, 1 / 2], [0.0, 0.0]]
    plot_bands(
        Hk_crys,
        k_data,
        k_path=k_path,
        k_path_name_list=["Γ", "X", "M", "Γ"],
        nband_range=range(1, lat.n_sub + 1),
        nk=30,
        save_path=FIG_DIR / "nonideal_checkerboard_extended_hopping.svg",
    )

    report(tb)


if __name__ == "__main__":
    main()
