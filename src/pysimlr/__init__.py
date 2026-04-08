from .svd import ba_svd, whiten_matrix, multiscale_svd
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
    map_lr_average_var
)
from .regression import smooth_matrix_prediction, smooth_regression
from .simlr import simlr

__version__ = "0.0.1"
