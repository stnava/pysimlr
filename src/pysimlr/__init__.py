"""
PySiMLR: Auditable Multi-View Representation Learning for Biomedical Discovery.

PySiMLR implements Similarity-driven Multi-view Linear Reconstruction (`simlr`)
alongside deep variational (`lend_simr`, `ned_simr`, `ned_simr_shared_private`)
and invertible normalizing flow (`flow_simr`, `flow_simr_v`) extensions, all
governed by the Linear Entrypoint Principle (`Z_{m,lin} = X_m @ V_m`) and
Non-Negative Stiefel Approximating Flow (`nsa_flow`) geometry.
"""

import importlib
import os
import sys
from typing import Any

__version__ = "0.18.0"


def _configure_matplotlib_backend() -> None:
    """
    Select a non-interactive backend when there is no display.

    Called lazily, the first time a plotting entry point is actually used.
    Importing matplotlib at package import time cost roughly a second -- it
    defeated the point of deferring the other heavy scientific imports.
    """
    import matplotlib
    if "DISPLAY" not in os.environ or os.environ.get("MPLBACKEND") == "Agg":
        # Preserve inline plotting when executing within Jupyter/IPython notebooks
        if not ('IPython' in sys.modules or 'ipykernel' in sys.modules):
            matplotlib.use("Agg")

from .simlr import (
    simlr,
    predict_simlr,
    predict_shared_latent,
    reconstruct_from_learned_maps,
    estimate_rank,
    decompose_energy,
    simlr_perm,
    initialize_simlr,
    nsa_contrast_transform,
    nsa_nonnegative_transform,
)
from .nsa_backend import (
    load_nsa_backend,
    load_nsa_flow,
    load_polar_factor,
    load_nsa_estimator,
    load_stiefel_defect_normalised,
    load_consolidate_supports,
    load_varimax,
    backend_report,
)
from .optimizers import (
    create_optimizer,
    SimlrOptimizer
)
from .sparsification import (
    optimize_indicator_matrix,
    indicator_opt_both_ways,
    rank_based_matrix_segmentation,
    orthogonalize_and_q_sparsify,
    simlr_sparseness,
    project_to_orthonormal_nonnegative,
    project_to_partially_orthonormal_nonnegative
)
from .svd import (
    ba_svd,
    safe_pca,
    whiten_matrix,
    multiscale_svd,
    varimax,
    varimax_rotation,
)
from .regression import (
    smooth_matrix_prediction,
    smooth_regression,
    build_nsa_pipeline,
)
from .structural import (
    fit_structural_models,
    create_path_graph
)
from .paths import (
    simlr_path,
    permutation_test
)
from .deep import (
    lend_simr,
    ned_simr,
    ned_simr_shared_private,
    LENDNSAEncoder,
    ModalityDecoder,
    LENDSiMRModel,
    NEDSiMRModel,
    NEDSharedPrivateSiMRModel,
    ModalityEncoder,
    predict_deep
)
from .flows import (
    flow_simr,
    flow_simr_v,
    FlowSiMRModel,
    FlowSiMRVModel,
    FlowEncoderWrapper,
    FlowDecoderWrapper,
    FlowConditionalInference,
    NormalizingFlow,
)
from .consensus import (
    compute_shared_consensus
)
from .api import (
    fit,
    predict,
    method_names,
    describe_method,
    resolve_method,
    validate_result,
    parameter_summary,
    METHODS,
)
from .fused_ridge import fit_fused_ridge, predict_fused_ridge, fused_ridge_cv
from .relevance import feature_relevance
from .screening import apply_view_screen, screen_wide_views
from .loadings import (
    RANK_CRITERIA,
    loadings,
    feature_maps,
    FEATURE_MAP_KINDS,
    loading_inference,
    select_rank,
    structure_coefficients,
    olee_loadings,
)
from .interpretability import (
    summarize_basis_matrix,
    build_first_layer_contract,
    extract_first_layer_factors,
    analyze_first_layer_alignment,
    attribute_shared_to_first_layer,
    attribute_prediction_to_features,
    build_interpretability_report,
)
from .utils import (
    ConvergenceWarning,
    WideViewWarning,
    coupling_capacity_report,
    heldout_coupling,
    set_seed_based_on_time,
    multigrep,
    get_names_from_dataframe,
    map_asym_var,
    map_lr_average_var,
    rvcoef,
    adjusted_rvcoef,
    l1_normalize_features,
    orthogonality_defect,
    gradient_orthogonality_defect,
    angle_defect,
    gradient_angle_defect,
    write_simlr,
    read_simlr
)

__all__ = [
    'RANK_CRITERIA',
    'feature_table',
    'plot_basis_diagnostics',
    'plot_rank_selection',
    'plot_latent_coupling',
    'plot_modality_weights',
    'plot_flow_warping',
    'plot_top_features',
    'plot_feature_map_comparison',
    'plot_feature_map_shift',
    'cortical_region_name',
    'brain_rows',
    'plot_brain_map',
    'plot_feature_relevance',
    'map_colormap',
    'feature_map_values',
    'plot_feature_map_grid',
    'plot_feature_groups',
    'plot_variance_explained',
    'plot_fit_summary',
    'plot_benchmark_uncertainty',
    'plot_loadings',
    'plot_scores',
    'plot_convergence',
    'plot_view_correlations',
    'plot_latent_recovery',
    'plot_benchmark',
    'fit_fused_ridge',
    'predict_fused_ridge',
    'fused_ridge_cv',
    'plot_fused_coefficients',
    'plot_coefficient_paths',
    'plot_fused_ridge_cv',
    'plot_predictions',
    'ConvergenceWarning',
    'WideViewWarning',
    'coupling_capacity_report',
    'heldout_coupling',
    'screen_wide_views',
    'apply_view_screen',
    '__version__',
    'simlr',
    'predict_simlr',
    'loadings',
    'loading_inference',
    'feature_maps',
    'feature_relevance',
    'FEATURE_MAP_KINDS',
    'select_rank',
    'structure_coefficients',
    'olee_loadings',
    'predict_shared_latent',
    'reconstruct_from_learned_maps',
    'estimate_rank',
    'decompose_energy',
    'simlr_perm',
    'initialize_simlr',
    'nsa_contrast_transform',
    'nsa_nonnegative_transform',
    'load_nsa_backend',
    'load_nsa_flow',
    'load_polar_factor',
    'load_nsa_estimator',
    'load_stiefel_defect_normalised',
    'load_consolidate_supports',
    'load_varimax',
    'backend_report',
    'NSAFlow',
    'create_optimizer',
    'SimlrOptimizer',
    'optimize_indicator_matrix',
    'indicator_opt_both_ways',
    'rank_based_matrix_segmentation',
    'orthogonalize_and_q_sparsify',
    'simlr_sparseness',
    'project_to_orthonormal_nonnegative',
    'project_to_partially_orthonormal_nonnegative',
    'ba_svd',
    'safe_pca',
    'whiten_matrix',
    'multiscale_svd',
    'varimax',
    'varimax_rotation',
    'sparse_distance_matrix',
    'sparse_distance_matrix_xy',
    'sparse_dist',
    'SparseGraphResolvent',
    'create_grid_graph_laplacian',
    'create_graph_laplacian',
    'create_laplacian_resolvent_operator',
    'create_spatial_smoothing_operator',
    'create_smoothing_operator',
    'smooth_matrix_prediction',
    'smooth_regression',
    'build_nsa_pipeline',
    'MultiViewSiMR',
    'MultiViewPCA',
    'build_simlr_pipeline',
    'nnh_embed',
    'extend_simlr_embedding_with_new_modalities',
    'extend',
    'predict_views',
    'apply_simlr_matrices',
    'apply_simlr_matrices_dtfix',
    'fit_structural_models',
    'create_path_graph',
    'simlr_path',
    'permutation_test',
    'lend_simr',
    'ned_simr',
    'ned_simr_shared_private',
    'LENDNSAEncoder',
    'ModalityDecoder',
    'LENDSiMRModel',
    'NEDSiMRModel',
    'NEDSharedPrivateSiMRModel',
    'ModalityEncoder',
    'predict_deep',
    'flow_simr',
    'flow_simr_v',
    'FlowSiMRModel',
    'FlowSiMRVModel',
    'FlowEncoderWrapper',
    'FlowDecoderWrapper',
    'FlowConditionalInference',
    'NormalizingFlow',
    'compute_shared_consensus',
    'fit',
    'predict',
    'method_names',
    'describe_method',
    'resolve_method',
    'validate_result',
    'parameter_summary',
    'METHODS',
    'summarize_basis_matrix',
    'build_first_layer_contract',
    'extract_first_layer_factors',
    'analyze_first_layer_alignment',
    'attribute_shared_to_first_layer',
    'attribute_prediction_to_features',
    'build_interpretability_report',
    'set_seed_based_on_time',
    'multigrep',
    'get_names_from_dataframe',
    'map_asym_var',
    'map_lr_average_var',
    'rvcoef',
    'adjusted_rvcoef',
    'l1_normalize_features',
    'orthogonality_defect',
    'gradient_orthogonality_defect',
    'angle_defect',
    'gradient_angle_defect',
    'write_simlr',
    'read_simlr',
    'plot_lend_simr_architecture',
    'plot_ned_simr_architecture',
    'plot_ned_shared_private_architecture',
    'plot_nsa_flow_architecture',
    'plot_flow_simr_architecture',
    'plot_path_model',
    'generate_all_architecture_graphs',
    'benchmarks'
]

#: Attributes served on first access instead of at import time, because their
#: modules pull in matplotlib, scipy or pandas. Maps attribute name -> submodule.
_LAZY_ATTRS = {
    'plot_lend_simr_architecture': 'diagrams',
    'plot_ned_simr_architecture': 'diagrams',
    'plot_ned_shared_private_architecture': 'diagrams',
    'plot_nsa_flow_architecture': 'diagrams',
    'plot_flow_simr_architecture': 'diagrams',
    'plot_path_model': 'diagrams',
    'generate_all_architecture_graphs': 'diagrams',
    'plot_fused_coefficients': 'plotting',
    'plot_coefficient_paths': 'plotting',
    'plot_fused_ridge_cv': 'plotting',
    'plot_predictions': 'plotting',
    'feature_table': 'plotting',
    'plot_basis_diagnostics': 'plotting',
    'plot_rank_selection': 'plotting',
    'plot_latent_coupling': 'plotting',
    'plot_modality_weights': 'plotting',
    'plot_flow_warping': 'plotting',
    'plot_top_features': 'plotting',
    'plot_feature_map_comparison': 'plotting',
    'plot_feature_map_shift': 'plotting',
    'cortical_region_name': 'plotting',
    'brain_rows': 'plotting',
    'plot_brain_map': 'plotting',
    'plot_feature_relevance': 'plotting',
    'map_colormap': 'plotting',
    'feature_map_values': 'plotting',
    'plot_feature_map_grid': 'plotting',
    'plot_feature_groups': 'plotting',
    'plot_variance_explained': 'plotting',
    'plot_fit_summary': 'plotting',
    'plot_benchmark_uncertainty': 'plotting',
    'plot_loadings': 'plotting',
    'plot_scores': 'plotting',
    'plot_convergence': 'plotting',
    'plot_view_correlations': 'plotting',
    'plot_latent_recovery': 'plotting',
    'plot_benchmark': 'plotting',
    'plot_view_chord': 'plotting',
    'plot_component_radar': 'plotting',
    'plot_view_radar': 'plotting',
    'plot_feature_chord': 'plotting',
    # scipy and pandas: ~0.2 s each, needed only by these.
    'sparse_distance_matrix': 'sparse',
    'sparse_distance_matrix_xy': 'sparse',
    'sparse_dist': 'sparse',
    'SparseGraphResolvent': 'sparse',
    'create_grid_graph_laplacian': 'sparse',
    'create_graph_laplacian': 'sparse',
    'create_laplacian_resolvent_operator': 'sparse',
    'create_spatial_smoothing_operator': 'sparse',
    'create_smoothing_operator': 'sparse',
    'extend': 'extension',
    'predict_views': 'extension',
    'nnh_embed': 'nnh',
    'extend_simlr_embedding_with_new_modalities': 'nnh',
    'apply_simlr_matrices': 'nnh',
    'apply_simlr_matrices_dtfix': 'nnh',
}

#: Submodules exposed as attributes but not imported eagerly.
_LAZY_MODULES = ('diagrams', 'benchmarks', 'plotting')


def __getattr__(name: str) -> Any:
    """
    Resolve plotting entry points and heavy submodules on first access.

    Raises
    ------
    AttributeError
        If `name` is not a public attribute of this package.
    """
    if name in _LAZY_ATTRS:
        _configure_matplotlib_backend()
        module = importlib.import_module(f".{_LAZY_ATTRS[name]}", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _LAZY_MODULES:
        _configure_matplotlib_backend()
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    if name in _LAZY_SKLEARN:
        # scikit-learn wrappers resolve lazily (PEP 562): importing them
        # eagerly made `import pysimlr` import sklearn for every caller, which
        # test_heavy_optional_modules_are_not_imported_eagerly forbids.
        # nsa_flow 3.1.2 fixed the same bug on its side.
        from . import sklearn as _sk
        obj = getattr(_sk, name)
        globals()[name] = obj
        return obj
    if name == "NSAFlow":
        from .nsa_backend import load_nsa_estimator
        cls = load_nsa_estimator()
        if cls is None:
            raise AttributeError(
                "NSAFlow estimator is not available. Install nsa-flow via `pip install nsa-flow`."
            )
        globals()[name] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))


_LAZY_SKLEARN = ('MultiViewPCA', 'MultiViewSiMR', 'build_simlr_pipeline')
