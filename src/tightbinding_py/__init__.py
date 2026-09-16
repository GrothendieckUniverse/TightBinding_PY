"""tightbinding_py — faithful Python port of ``TightBinding.jl``.

Mirrors the Julia module ``TightBinding`` (``src/TightBinding.jl``) module
for module; every exported name below has a documented counterpart in
:mod:`tightbinding_py.lattice`, :mod:`tightbinding_py.tb_model`,
:mod:`tightbinding_py.uniform_grids`, :mod:`tightbinding_py.utils`, and
:mod:`tightbinding_py.band_plot`.

Note on one Julia quirk faithfully reproduced: the Julia module *exports*
``plot_band_counter`` (a typo) while *defining* ``plot_band_contour``.
The Python port provides the actual function
:func:`tightbinding_py.band_plot.plot_band_contour` (and no
``plot_band_counter``).
"""

from .lattice import (
    LatticeGraph,
    Real_Space_Lattice,
    initialize_real_space_lattice,
    plot_real_space_lattice,
)
from .tb_model import (
    Real_Space_TightBinding_Model,
    add_hopping_term,
    add_hopping_term_to_full_hopping_map,
    add_hoppings_by_graph_distance,
    haldane_nnn_hopping_amplitude,
    initialize_real_space_tightbinding_model,
    plot_real_space_tightbinding_model,
)
from .uniform_grids import (
    Uniform_Grids,
    initialize_uniform_grids,
    initialize_uniform_grids_from_lattice,
)
from .utils import (
    Chern_number_Fukui_Hatsugai_Suzuki,
    build_Hk_crys,
    build_real_space_tb_Hamiltonain,
    dual_basis_vec_list,
    dual_basis_vec_mat,
    generate_bilinear_terms,
    many_body_Chern_number_Fukui_Hatsugai_Suzuki,
)
from .band_plot import (
    find_1st_BZ_k_cart_list,
    plot_band_contour,
    plot_bands,
)

__all__ = [
    # lattice
    "Real_Space_Lattice",
    "LatticeGraph",
    "initialize_real_space_lattice",
    "plot_real_space_lattice",
    # tight-binding model
    "Real_Space_TightBinding_Model",
    "initialize_real_space_tightbinding_model",
    "add_hopping_term",
    "add_hopping_term_to_full_hopping_map",
    "add_hoppings_by_graph_distance",
    "haldane_nnn_hopping_amplitude",
    "plot_real_space_tightbinding_model",
    # uniform grids
    "Uniform_Grids",
    "initialize_uniform_grids",
    "initialize_uniform_grids_from_lattice",
    # utils
    "dual_basis_vec_mat",
    "dual_basis_vec_list",
    "build_Hk_crys",
    "Chern_number_Fukui_Hatsugai_Suzuki",
    "generate_bilinear_terms",
    "build_real_space_tb_Hamiltonain",
    "many_body_Chern_number_Fukui_Hatsugai_Suzuki",
    # band plot
    "plot_bands",
    "plot_band_contour",
    "find_1st_BZ_k_cart_list",
]
