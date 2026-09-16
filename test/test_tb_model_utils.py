"""Tests for the tight-binding model builder and utility functions."""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

import numpy as np
import scipy.sparse as sp

from tightbinding_py import (
    add_hopping_term,
    build_Hk_crys,
    build_real_space_tb_Hamiltonain,
    dual_basis_vec_mat,
    initialize_real_space_lattice,
    initialize_real_space_tightbinding_model,
    initialize_uniform_grids_from_lattice,
    many_body_Chern_number_Fukui_Hatsugai_Suzuki,
)


def build_haldane_model(
    t1: float = -1.0,
    t2: float = -0.24,
    phi: float = np.pi / 2,
    mass: float = 0.7,
    sample_size=(3, 4),
):
    """Haldane model via add_hopping_term (the doc/design.ipynb convention)."""
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


class AddHoppingTermTest(unittest.TestCase):
    def setUp(self):
        self.lat = initialize_real_space_lattice(
            lattice_name="honeycomb",
            sample_size=[3, 4],
            pbc_indicator=[True, True],
        )
        self.tb = initialize_real_space_tightbinding_model(self.lat)

    def test_hermitian_expansion(self):
        amp = 1.0 + 2.0j
        add_hopping_term(self.tb, ((((0, 0), 1), ((0, 0), 2)), amp))
        key = (((0, 0), 1), ((0, 0), 2))
        hc_key = (((0, 0), 2), ((0, 0), 1))
        self.assertAlmostEqual(self.tb.input_hopping_map[key], amp)
        self.assertAlmostEqual(self.tb.input_hopping_map[hc_key], np.conj(amp))

    def test_non_hermitian_adds_only_forward(self):
        add_hopping_term(self.tb, ((((0, 0), 1), ((0, 0), 2)), 0.5), is_hermitian=False)
        self.assertEqual(len(self.tb.input_hopping_map), 1)

    def test_overwrite_warns_and_replaces(self):
        add_hopping_term(self.tb, ((((0, 0), 1), ((0, 0), 2)), 1.0))
        buf = io.StringIO()
        with redirect_stdout(buf):
            add_hopping_term(self.tb, ((((0, 0), 1), ((0, 0), 2)), 2.0))
        self.assertIn("WARNING", buf.getvalue())
        # The old value is overwritten, not added to.
        self.assertAlmostEqual(
            self.tb.input_hopping_map[(((0, 0), 1), ((0, 0), 2))], 2.0
        )

    def test_invalid_sublattice_index_raises(self):
        with self.assertRaises(ValueError):
            add_hopping_term(self.tb, ((((0, 0), 1), ((0, 0), 5)), 1.0))


class HamiltonianBuildTest(unittest.TestCase):
    def test_hermiticity_and_nnz(self):
        # Graphene: three NN hoppings on the honeycomb [3,4] torus.
        lat = initialize_real_space_lattice(
            lattice_name="honeycomb",
            sample_size=[3, 4],
            pbc_indicator=[True, True],
        )
        tb = initialize_real_space_tightbinding_model(lat)
        add_hopping_term(tb, ((((0, 0), 1), ((0, 0), 2)), -1.0))
        add_hopping_term(tb, ((((0, 0), 1), ((0, -1), 2)), -1.0))
        add_hopping_term(tb, ((((0, 0), 1), ((-1, 0), 2)), -1.0))

        H = build_real_space_tb_Hamiltonain(tb)
        self.assertIsInstance(H, sp.csc_matrix)
        Hd = H.toarray()
        self.assertEqual(H.shape, (24, 24))
        self.assertEqual(H.nnz, 72)
        np.testing.assert_allclose(Hd, Hd.conj().T, atol=1e-14)


class DualBasisTest(unittest.TestCase):
    def test_duality_relation(self):
        brav = [[1.0, 0.0], [0.5, np.sqrt(3) / 2]]
        dual = dual_basis_vec_mat(brav)
        # dual.T @ brav = 2 pi I (brav stored in columns).
        np.testing.assert_allclose(
            dual.T @ np.asarray(brav).T, 2 * np.pi * np.eye(2), atol=1e-12
        )


class UniformGridsTest(unittest.TestCase):
    def test_from_lattice_counts_and_crys(self):
        lat = initialize_real_space_lattice(
            lattice_name="honeycomb",
            sample_size=[3, 4],
            pbc_indicator=[True, True],
            twisted_phases_over_2π=[0.5, 0.0],
        )
        grid = initialize_uniform_grids_from_lattice(lat)
        self.assertEqual(grid.nsite, lat.n_cell)
        np.testing.assert_allclose(grid.site_crys_list[0], [0.5 / 3, 0.0])


class ManyBodyChernTest(unittest.TestCase):
    def test_half_filling_matches_lower_band_chern(self):
        tb = build_haldane_model()
        # n_occ = n_cell fills exactly the lower band (half filling); its
        # many-body Chern number equals the lower band's single-particle
        # Chern number, -1 for this convention.
        C = many_body_Chern_number_Fukui_Hatsugai_Suzuki(
            tb, n_occ=tb.lattice.n_cell, nθ=13
        )
        self.assertAlmostEqual(C, -1.0, places=5)


if __name__ == "__main__":
    unittest.main()
