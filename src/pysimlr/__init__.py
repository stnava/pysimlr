from .simlr import (
    simlr,
    predict_simlr,
    estimate_rank,
    decompose_energy,
    simlr_perm,
    initialize_simlr
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
    sparse_distance_matrix_xy
)
from .regression import (
    smooth_matrix_prediction,
    smooth_regression
)
from .nnh import (
    nnh_embed
)
from .paths import (
    simlr_path,
    permutation_test
)
from .deep import (
    deep_simr,
    lend_simr
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
    gradient_invariant_orthogonality_defect
)

__all__ = [
    'simlr',
    'predict_simlr',
    'estimate_rank',
    'decompose_energy',
    'simlr_perm',
    'initialize_simlr',
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
    'smooth_matrix_prediction',
    'smooth_regression',
    'nnh_embed',
    'simlr_path',
    'permutation_test',
    'deep_simr',
    'lend_simr',
    'set_seed_based_on_time',
    'multigrep',
    'get_names_from_dataframe',
    'map_asym_var',
    'map_lr_average_var',
    'rvcoef',
    'adjusted_rvcoef',
    'l1_normalize_features',
    'invariant_orthogonality_defect',
    'gradient_invariant_orthogonality_defect'
]
