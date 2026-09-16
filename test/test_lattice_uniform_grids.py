"""Tests for lattice construction and uniform grids (tightbinding_py).

The reference numbers in this module were cross-checked against the Julia
package ``TightBinding.jl``.  All sublattice / site indices are **1-based**.
"""

from __future__ import annotations

import unittest

import numpy as np

from tightbinding_py import (
    add_hopping_term,
    add_hoppings_by_graph_distance,
    build_Hk_crys,
    Chern_number_Fukui_Hatsugai_Suzuki,
    generate_bilinear_terms,
    initialize_real_space_lattice,
    initialize_real_space_tightbinding_model,
    initialize_uniform_grids,
    initialize_uniform_grids_from_lattice,
)


def build_haldane_model(
    t1: float = -1.0,
    t2: float = -0.24,
    phi: float = np.pi / 2,
    mass: float = 0.7,
    sample_size=(3, 4),
):
    """Haldane model via add_hopping_term (the doc/design.ipynb convention).

    Sublattice ``A1`` (``src=1``) carries chirality sign ``+1``, ``A2``
    (``src=2``) carries ``-1``; with these phases the lower band has Chern
    number ``-1`` and the upper band ``+1``.
    """
    lat = initialize_real_space_lattice(
        lattice_name="honeycomb",
        sample_size=list(sample_size),
        pbc_indicator=[True, True],
    )
    tb = initialize_real_space_tightbinding_model(lat, model_name="Haldane")

    add_hopping_term(tb, ((((0, 0), 1), ((0, 0), 1)), mass), is_hermitian=False)
    add_hopping_term(tb, ((((0, 0), 2), ((0, 0), 2)), -mass), is_hermitian=False)

    add_hopping_term(tb, ((((0, 0), 1), ((0, 0), 2)), t1))
    add_hopping_term(tb, ((((0, 0), 1), ((0, -1), 2)), t1))
    add_hopping_term(tb, ((((0, 0), 1), ((-1, 0), 2)), t1))

    for src in (1, 2):
        sgn = 1 if src == 1 else -1
        add_hopping_term(tb, ((((0, 0), src), ((1, 0), src)), t2 * np.exp(sgn * 1j * phi)))
        add_hopping_term(tb, ((((0, 0), src), ((0, 1), src)), t2 * np.exp(-sgn * 1j * phi)))
        add_hopping_term(tb, ((((0, 0), src), ((-1, 1), src)), t2 * np.exp(sgn * 1j * phi)))

    return tb


class HoneycombLatticeTest(unittest.TestCase):
    """Reference numbers for the PBC honeycomb lattice [3, 4]."""

    def setUp(self):
        self.lat = initialize_real_space_lattice(
            lattice_name="honeycomb",
            sample_size=[3, 4],
            pbc_indicator=[True, True],
        )

    def test_counts_and_volume(self):
        self.assertEqual(self.lat.n_site, 24)
        self.assertEqual(self.lat.n_cell, 12)
        self.assertEqual(self.lat.n_sub, 2)
        self.assertAlmostEqual(self.lat.cell_volume, np.sqrt(3) / 2, places=14)

    def test_sub_names(self):
        self.assertEqual(self.lat.sub_name_list, ["A1", "A2"])

    def test_site_ordering_first_axis_fastest(self):
        expected = [
            ((0, 0), 1),
            ((0, 0), 2),
            ((1, 0), 1),
            ((1, 0), 2),
            ((2, 0), 1),
            ((2, 0), 2),
        ]
        self.assertEqual(self.lat.site_list[:6], expected)

    def test_site_to_index_map_is_1_based(self):
        self.assertEqual(self.lat.site_to_index_map[((0, 0), 1)], 1)
        self.assertEqual(self.lat.site_to_index_map[((0, 0), 2)], 2)

    def test_graph_edge_count(self):
        self.assertEqual(self.lat.graph.n_edges(), 36)

    def test_nearest_neighbor_distance(self):
        # The shortest graph edge length is the honeycomb NN distance 1/sqrt(3).
        brav = np.asarray(self.lat.brav_vec_list, dtype=float)
        distances = []
        for (i, j) in self.lat.graph.edges():
            d = self.lat.site_crys_list[j - 1] - self.lat.site_crys_list[i - 1]
            d = np.asarray(d, dtype=float)
            # wrap to the minimum-image convention
            for axis in range(self.lat.dim):
                if self.lat.pbc_indicator[axis]:
                    d[axis] -= round(d[axis] / self.lat.sample_size[axis]) * self.lat.sample_size[axis]
            distances.append(float(np.linalg.norm(d @ brav)))
        self.assertAlmostEqual(min(distances), 1 / np.sqrt(3), places=12)


class HaldaneModelTest(unittest.TestCase):
    """H(k) eigenvalues and Chern numbers of the Haldane model."""

    def setUp(self):
        # Store as *instance* attributes: a plain function stored as a class
        # attribute would be auto-bound as a method when read via `self`.
        self.tb = build_haldane_model()
        self.Hk = build_Hk_crys(self.tb)

    def _eigs(self, k):
        return np.sort(np.linalg.eigvalsh(self.Hk(np.asarray(k, dtype=float))))

    def test_eigenvalues_at_high_symmetry_points(self):
        cases = [
            ((0.0, 0.0), 3.080584360150),
            ((1 / 3, 1 / 3), 1.868154169227),
            ((2 / 3, 1 / 3), 0.547076581450),
            ((1 / 2, 1 / 2), 1.220655561573),
        ]
        for k, expected in cases:
            ev = self._eigs(k)
            self.assertAlmostEqual(ev[0], -expected, places=9)
            self.assertAlmostEqual(ev[1], +expected, places=9)

    def test_chern_numbers(self):
        C1 = Chern_number_Fukui_Hatsugai_Suzuki(self.Hk, band=1, nk=51)
        C2 = Chern_number_Fukui_Hatsugai_Suzuki(self.Hk, band=2, nk=51)
        self.assertAlmostEqual(C1, -1.0, places=6)
        self.assertAlmostEqual(C2, +1.0, places=6)


class TwistedPhaseValidationTest(unittest.TestCase):
    def test_twist_on_open_direction_raises(self):
        with self.assertRaises(ValueError):
            initialize_real_space_lattice(
                lattice_name="honeycomb",
                sample_size=[3, 4],
                pbc_indicator=[True, False],
                twisted_phases_over_2π=[0.0, 0.3],
            )

    def test_twist_on_periodic_direction_ok(self):
        lat = initialize_real_space_lattice(
            lattice_name="honeycomb",
            sample_size=[3, 4],
            pbc_indicator=[True, True],
            twisted_phases_over_2π=[0.3, 0.0],
        )
        self.assertEqual(lat.twisted_phases_over_2π, [0.3, 0.0])


class BilinearTermsTest(unittest.TestCase):
    def test_twisted_phase_adds_winding_factor(self):
        lat = initialize_real_space_lattice(
            lattice_name="honeycomb",
            sample_size=[3, 4],
            pbc_indicator=[True, True],
        )
        tb = initialize_real_space_tightbinding_model(lat)
        # Directed template with cell shift (1, 0): for the cell (2, y) the
        # destination (3, y) wraps to (0, y) with winding +1.
        add_hopping_term(tb, ((((0, 0), 1), ((1, 0), 1)), 1.0), is_hermitian=False)

        terms_0 = generate_bilinear_terms(tb, twisted_phases_over_2π=[0.0, 0.0])
        terms_t = generate_bilinear_terms(tb, twisted_phases_over_2π=[0.25, 0.0])
        self.assertEqual(len(terms_0), lat.n_cell)
        self.assertEqual(len(terms_t), lat.n_cell)

        # Reconstruct the amplitude at each source cell (keyed by i_site).
        amp_0 = {i: c for (i, j, c) in terms_0}
        amp_t = {i: c for (i, j, c) in terms_t}

        # Cells (0, y) and (1, y) do not wind: amplitudes unchanged.
        for y in range(4):
            i_00 = lat.site_to_index_map[((0, y), 1)]
            i_10 = lat.site_to_index_map[((1, y), 1)]
            self.assertAlmostEqual(amp_t[i_00], 1.0)
            self.assertAlmostEqual(amp_t[i_10], 1.0)

        # Cell (2, y) winds by +1: amplitude acquires exp(2 pi i * 0.25) = i.
        for y in range(4):
            i_20 = lat.site_to_index_map[((2, y), 1)]
            self.assertAlmostEqual(amp_t[i_20], np.exp(2j * np.pi * 0.25), places=12)


class GraphDistanceHoppingTest(unittest.TestCase):
    def test_graph_distance_one_adds_two_directed_entries_per_edge(self):
        lat = initialize_real_space_lattice(
            lattice_name="honeycomb",
            sample_size=[3, 4],
            pbc_indicator=[True, True],
        )
        tb = initialize_real_space_tightbinding_model(lat)
        add_hoppings_by_graph_distance(tb, 1, -1.0)
        # 36 undirected graph edges -> 36 * 2 directed full_hopping_map entries.
        self.assertEqual(len(tb.full_hopping_map), 36 * 2)


class UniformGridsTest(unittest.TestCase):
    def test_grid_from_lattice_counts_and_coordinates(self):
        lat = initialize_real_space_lattice(
            lattice_name="honeycomb",
            sample_size=[3, 4],
            pbc_indicator=[True, True],
            twisted_phases_over_2π=[0.5, 0.0],
        )
        grid = initialize_uniform_grids_from_lattice(lat)
        self.assertEqual(grid.nsite, lat.n_cell)
        # site_crys = (site_int + twist) / sample_size
        np.testing.assert_allclose(grid.site_crys_list[0], [0.5 / 3, 0.0])
        self.assertEqual(grid.name, "honeycomb")

    def test_standalone_grid(self):
        grid = initialize_uniform_grids(
            basis_vec_list=[[1.0, 0.0], [0.0, 1.0]],
            sample_size=[2, 3],
            twisted_phases_over_2π=[0.5, 0.0],
        )
        self.assertEqual(grid.nsite, 6)
        np.testing.assert_allclose(grid.site_crys_list[0], [0.25, 0.0])


if __name__ == "__main__":
    unittest.main()
