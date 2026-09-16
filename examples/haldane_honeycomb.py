"""Haldane model on the honeycomb lattice — a Chern insulator.

Physics
-------
The Haldane model (F. D. M. Haldane, *Phys. Rev. Lett.* **61**, 2015 (1988))
is the minimal lattice realisation of the *quantum anomalous Hall effect*: a
band insulator with a nonzero Chern number and no net magnetic field.  It is
graphene (nearest-neighbour hopping ``t1``) augmented by two
time-reversal-breaking ingredients:

1. a **staggered sublattice potential** ``+M`` on sublattice ``A1`` and
   ``-M`` on ``A2`` (breaks inversion symmetry);
2. complex **next-nearest-neighbour (NNN) hoppings**
   ``t2 * exp(i * sgn * phi)`` whose phase sign ``sgn`` is fixed by the
   chirality of the two-hop path ``i -> k -> j`` (breaks time-reversal).

For ``|M| < 3*sqrt(3)*|t2*sin(phi)|`` the two Dirac cones acquire opposite
masses and the two bands carry Chern numbers ``C = -1`` (lower) and ``+1``
(upper).  The parameters used here (``t1=-1``, ``t2=-0.24``, ``phi=pi/2``,
``M=0.7``) lie inside the topological phase.

The model is built with :func:`add_hopping_term`, declaring one hopping
*template* per translationally-inequivalent bond; ``is_hermitian=True`` (the
default) adds the Hermitian conjugate automatically.  The complex NNN
amplitude pattern mirrors ``doc/design.ipynb`` (sublattice ``A1`` carries
``sgn=+1``, ``A2`` carries ``sgn=-1``), so the lower band has ``C=-1``.

Run from the package root::

    .venv/bin/python examples/haldane_honeycomb.py
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
    plot_real_space_lattice,
    plot_real_space_tightbinding_model,
)

FIG_DIR = Path("figures") / "examples"


def build_haldane_model(
    t1: float = -1.0,
    t2: float = -0.24,
    phi: float = np.pi / 2,
    mass: float = 0.7,
    sample_size: list[int] | None = None,
):
    """Build the Haldane model on a PBC honeycomb lattice.

    Args:
        t1: nearest-neighbour hopping amplitude.
        t2: next-nearest-neighbour hopping amplitude (complex phase ``phi``).
        phi: Haldane NNN phase.
        mass: staggered sublattice potential (``+mass`` on A1, ``-mass`` on A2).
        sample_size: number of unit cells along each Bravais vector
            (default ``[3, 3]``).

    Returns:
        Real_Space_TightBinding_Model: the assembled Haldane model.
    """
    lat = initialize_real_space_lattice(
        lattice_name="honeycomb",
        sample_size=sample_size if sample_size is not None else [3, 3],
        pbc_indicator=[True, True],
    )
    tb = initialize_real_space_tightbinding_model(lat, model_name="Haldane")

    # --- staggered sublattice potential (mass) ---------------------------
    # On-site terms: is_hermitian=False so they are not double-counted.
    add_hopping_term(tb, ((((0, 0), 1), ((0, 0), 1)), mass), is_hermitian=False)
    add_hopping_term(tb, ((((0, 0), 2), ((0, 0), 2)), -mass), is_hermitian=False)

    # --- nearest-neighbour hopping t1 (A1 -> A2) -------------------------
    add_hopping_term(tb, ((((0, 0), 1), ((0, 0), 2)), t1))
    add_hopping_term(tb, ((((0, 0), 1), ((0, -1), 2)), t1))
    add_hopping_term(tb, ((((0, 0), 1), ((-1, 0), 2)), t1))

    # --- complex next-nearest-neighbour hopping (Haldane) ----------------
    # Intra-sublattice NNN hoppings; sublattice A1 has chirality sign +1,
    # A2 has -1 (this convention gives C_lower = -1, C_upper = +1).
    for src in (1, 2):
        sgn = 1 if src == 1 else -1
        add_hopping_term(tb, ((((0, 0), src), ((1, 0), src)), t2 * np.exp(sgn * 1j * phi)))
        add_hopping_term(tb, ((((0, 0), src), ((0, 1), src)), t2 * np.exp(-sgn * 1j * phi)))
        add_hopping_term(tb, ((((0, 0), src), ((-1, 1), src)), t2 * np.exp(sgn * 1j * phi)))

    return tb


def main() -> None:
    tb = build_haldane_model(t1=-1.0, t2=-0.24, phi=np.pi / 2, mass=0.7)
    lat = tb.lattice

    # --- 1. real-space lattice + tight-binding model figures ---------------
    plot_real_space_lattice(lat, save_path=FIG_DIR / "haldane_honeycomb_lattice.svg")
    plot_real_space_tightbinding_model(tb, save_path=FIG_DIR / "haldane_honeycomb_model.svg")

    # --- 2. k-space Hamiltonian & band structure ---------------------------
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
        save_path=FIG_DIR / "haldane_honeycomb_bands.svg",
    )

    # --- 3. eigenvalues at high-symmetry points & Chern numbers ------------
    for name, k in [("Γ", [0.0, 0.0]), ("K", [2 / 3, 1 / 3])]:
        ev = np.linalg.eigvalsh(Hk_crys(np.asarray(k, dtype=float)))
        print(f"eigenvalues at {name} = {np.round(np.sort(ev), 6)}")

    C1 = Chern_number_Fukui_Hatsugai_Suzuki(Hk_crys, band=1, nk=51)
    C2 = Chern_number_Fukui_Hatsugai_Suzuki(Hk_crys, band=2, nk=51)
    print(f"Chern numbers: C_band1 = {C1:.6f},  C_band2 = {C2:.6f}")


if __name__ == "__main__":
    main()
