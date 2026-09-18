import importlib
import os
import sys
from typing import Any

__version__ = "0.2.11"


def _configure_matplotlib_backend() -> None:
    """
    Select a non-interactive backend when there is no display.

    Called lazily, the first time a plotting entry point is actually used.
    Importing matplotlib at package import time cost roughly a second -- it
    arrives via pysimlr.benchmarks.plotting, which pulls in seaborn,
    scipy.stats and ipywidgets -- which defeated the point of deferring the
    other heavy scientific imports.
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
    calculate_u,
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
    multiscale_svd
)
from .sparse import (
    sparse_distance_matrix,
    sparse_distance_matrix_xy,
    sparse_dist,
    SparseGraphResolvent,
    create_grid_graph_laplacian,
    create_graph_laplacian,
    create_laplacian_resolvent_operator,
    create_spatial_smoothing_operator,
    create_smoothing_operator
)
from .regression import (
    smooth_matrix_prediction,
    smooth_regression,
    build_nsa_pipeline,
)
from .sklearn import (
    SiMLREstimator,
    SiMLRTransformer,
    SiMLR,
    LENDTransformer,
    NEDTransformer,
    FlowSiMLRTransformer,
    build_simlr_pipeline,
)
from .nnh import (
    nnh_embed,
    extend_simlr_embedding_with_new_modalities,
    apply_simlr_matrices,
    apply_simlr_matrices_dtfix
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
    deep_simr,
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
    FlowSiMLRModel,
    FlowSiMLRVModel,
    flow_simlr,
    flow_simlr_v,
    FlowEncoderWrapper,
    FlowDecoderWrapper,
    FlowConditionalInference,
    NormalizingFlow,
    FlowWhitener,
    flow_whiten_matrix,
    flow_whitener
)
from .consensus import (
    compute_shared_consensus
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
    set_seed_based_on_time,
    multigrep,
    get_names_from_dataframe,
    map_asym_var,
    map_lr_average_var,
    rvcoef,
    adjusted_rvcoef,
    l1_normalize_features,
    invariant_orthogonality_defect,
    gradient_invariant_orthogonality_defect,
    orthogonality_defect,
    gradient_orthogonality_defect,
    angle_defect,
    gradient_angle_defect,
    write_simlr,
    read_simlr
)

__all__ = [
    '__version__',
    'simlr',
    'predict_simlr',
    'predict_shared_latent',
    'reconstruct_from_learned_maps',
    'estimate_rank',
    'decompose_energy',
    'simlr_perm',
    'initialize_simlr',
    'calculate_u',
    'nsa_contrast_transform',
    'nsa_nonnegative_transform',
    'load_nsa_backend',
    'load_nsa_flow',
    'load_polar_factor',
    'load_nsa_estimator',
    'load_stiefel_defect_normalised',
    'load_consolidate_supports',
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
    'SiMLREstimator',
    'SiMLRTransformer',
    'SiMLR',
    'LENDTransformer',
    'NEDTransformer',
    'FlowSiMLRTransformer',
    'build_simlr_pipeline',
    'nnh_embed',
    'extend_simlr_embedding_with_new_modalities',
    'apply_simlr_matrices',
    'apply_simlr_matrices_dtfix',
    'fit_structural_models',
    'create_path_graph',
    'simlr_path',
    'permutation_test',
    'deep_simr',
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
    'FlowSiMLRModel',
    'FlowSiMLRVModel',
    'flow_simlr',
    'flow_simlr_v',
    'FlowEncoderWrapper',
    'FlowDecoderWrapper',
    'FlowConditionalInference',
    'NormalizingFlow',
    'FlowWhitener',
    'flow_whiten_matrix',
    'flow_whitener',
    'compute_shared_consensus',
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
    'invariant_orthogonality_defect',
    'gradient_invariant_orthogonality_defect',
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
#: modules pull in matplotlib/seaborn. Maps attribute name -> submodule.
_LAZY_ATTRS = {
    'plot_lend_simr_architecture': 'visualization',
    'plot_ned_simr_architecture': 'visualization',
    'plot_ned_shared_private_architecture': 'visualization',
    'plot_nsa_flow_architecture': 'visualization',
    'plot_flow_simr_architecture': 'visualization',
    'plot_path_model': 'visualization',
    'generate_all_architecture_graphs': 'visualization',
}

#: Submodules exposed as attributes but not imported eagerly.
_LAZY_MODULES = ('visualization', 'viz', 'benchmarks')


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
