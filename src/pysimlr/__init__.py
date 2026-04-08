from .svd import ba_svd, safe_pca, whiten_matrix, multiscale_svd
from .sparse import sparse_distance_matrix, sparse_distance_matrix_xy
from .sparsification import (
    optimize_indicator_matrix,
    indicator_opt_both_ways,
    rank_based_matrix_segmentation,
    orthogonalize_and_q_sparsify
)
from .utils import (
    set_seed_based_on_time,
    multigrep,
    get_names_from_dataframe,
    map_asym_var,
    map_lr_average_var,
    rvcoef,
    adjusted_rvcoef
)
from .regression import smooth_matrix_prediction, smooth_regression
from .simlr import simlr, predict_simlr, estimate_rank, decompose_energy
from .nnh import nnh_embed
from .paths import simlr_path, permutation_test

__version__ = "0.1.0"
