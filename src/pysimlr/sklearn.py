"""
Scikit-learn compatible estimator and transformer interfaces for PySIMLR.

Provides full scikit-learn BaseEstimator and TransformerMixin compatibility
for SiMLR, LEND, NED, and Flow-SiMLR, including:
- Standard fit / transform / fit_transform / inverse_transform workflow
- Inspection via .components_ (shape [n_components, n_features]) and .loadings_ ([n_features, n_components])
- Direct drop-in usage within sklearn.pipeline.Pipeline, cross_val_score, GridSearchCV
- Seamless support for both single-view tabular data (X: [n, p]) and multi-view data (X: list of [n, p_m])
- High-performance default optimization via torch_lbfgs quasi-Newton solver
- Combinatorial support consolidation (consolidate=True) for strictly disjoint modules (zero crosstalk)
"""
from typing import Optional, Union, List, Tuple, Dict, Any
import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from .simlr import simlr, predict_shared_latent
from .deep import lend_simr, ned_simr, predict_deep
from .flows import flow_simr_v


def _to_numpy_2d(x: Any) -> np.ndarray:
    """Ensure array is 2D numpy float64/float32."""
    if hasattr(x, "to_numpy"):
        x = x.to_numpy()
    arr = np.asarray(x)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if not np.issubdtype(arr.dtype, np.floating):
        arr = arr.astype(np.float64)
    return arr


def _parse_input_views(X: Any) -> Tuple[List[torch.Tensor], bool, List[int]]:
    """
    Parse input into list of PyTorch float tensors.

    Returns
    -------
    views : List[torch.Tensor]
        List of 2D tensors.
    is_single_view : bool
        True if X was passed as a single 2D matrix rather than a list/tuple.
    feature_counts : List[int]
        Number of features per view.
    """
    if isinstance(X, (list, tuple)):
        is_single_view = False
        mats = [_to_numpy_2d(view) for view in X]
    else:
        is_single_view = True
        mats = [_to_numpy_2d(X)]

    feature_counts = [m.shape[1] for m in mats]
    tensors = [torch.from_numpy(m).float() for m in mats]
    return tensors, is_single_view, feature_counts


class SiMLREstimator(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer for Similarity-driven Multi-view Linear Representation (SiMLR).

    Can be used as a direct drop-in replacement for PCA, NMF, or FactorAnalysis in scikit-learn
    pipelines, supporting both standard single-view tables (X: [n, p]) and multi-modal lists
    (X: [X_1, X_2, ...]).

    Parameters
    ----------
    n_components : int, default=2
        Dimensionality of the shared latent space (k).
    iterations : int, default=50
        Maximum number of alternating optimization iterations.
    optimizer_type : str, default="torch_lbfgs"
        Optimizer for updating loading matrices. Defaults to "torch_lbfgs" for fast quasi-Newton convergence.
    energy_type : str, default="acc"
        Consensus similarity energy function ("acc", "regression", "logcosh", "nc").
    mixing_algorithm : str, default="newton"
        Algorithm for computing shared consensus ("newton", "svd", "pca", "ica").
    constraint : str, default="orthox0.1x1"
        Manifold constraint specification.
    sparseness_quantile : float, default=0.0
        Quantile threshold for sparsifying loading weights.
    positivity : {"either", "positive"}, default="either"
        Sign constraint on loadings. "positive" enforces non-negative loadings V >= 0.
    consolidate : bool, default=True
        Whether to enforce strictly disjoint module supports (zero crosstalk, zero lobe overlap).
    use_nsa : bool, default=True
        Whether to enable NSA-Flow manifold retractions during optimization.
    nsa_w : float, default=0.1 (the wrapped function's default)
        NSA-Flow trade-off stiffness weight between data fidelity and orthogonality.
    scale : bool, default=True
        Whether to standardize features prior to fitting.
    tol : float, default=1e-6
        Relative energy convergence tolerance.
    verbose : bool, default=False
        Whether to print optimization progress.
    random_state : int, optional
        Seed for reproducibility.
    """

    def __init__(
        self,
        n_components: int = 2,
        iterations: int = 50,
        optimizer_type: str = "torch_lbfgs",
        energy_type: str = "acc",
        mixing_algorithm: str = "newton",
        constraint: str = "orthox0.1x1",
        sparseness_quantile: float = 0.0,
        positivity: str = "either",
        consolidate: bool = True,
        use_nsa: bool = True,
        nsa_w: float = 0.1,
        scale: bool = True,
        tol: float = 1e-6,
        verbose: bool = False,
        random_state: Optional[int] = None,
    ):
        self.n_components = n_components
        self.iterations = iterations
        self.optimizer_type = optimizer_type
        self.energy_type = energy_type
        self.mixing_algorithm = mixing_algorithm
        self.constraint = constraint
        self.sparseness_quantile = sparseness_quantile
        self.positivity = positivity
        self.consolidate = consolidate
        self.use_nsa = use_nsa
        self.nsa_w = nsa_w
        self.scale = scale
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X: Any, y: Optional[Any] = None) -> "SiMLREstimator":
        """
        Fit the SiMLR model on input data X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of array-like
            Single-view data matrix or list of multi-view data matrices.
        y : Ignored
            Not used, present for scikit-learn API consistency.

        Returns
        -------
        self : SiMLREstimator
            Fitted estimator.
        """
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        tensors, is_single, feat_counts = _parse_input_views(X)
        self.is_single_view_ = is_single
        self.feature_counts_ = feat_counts
        self.n_features_in_ = feat_counts[0] if is_single else feat_counts

        scale_list = ["centerAndScale", "np"] if self.scale else ["none", "np"]

        res = simlr(
            tensors,
            k=self.n_components,
            iterations=self.iterations,
            optimizer_type=self.optimizer_type,
            energy_type=self.energy_type,
            constraint=self.constraint,
            mixing_algorithm=self.mixing_algorithm,
            sparseness_quantile=self.sparseness_quantile,
            positivity=self.positivity,
            consolidate=self.consolidate,
            scale_list=scale_list,
            tol=self.tol,
            verbose=self.verbose,
            use_nsa=self.use_nsa,
            nsa_w=self.nsa_w,
        )
        self.result_ = res

        # Convert learned loadings to numpy
        v_np = [v.detach().cpu().numpy() for v in res["v"]]
        u_val = res["u"]
        if isinstance(u_val, list):
            self.u_ = [u.detach().cpu().numpy() for u in u_val]
        else:
            self.u_ = u_val.detach().cpu().numpy()

        if is_single:
            self.loadings_ = v_np[0]
            # components_ follows scikit-learn convention: shape (n_components, n_features)
            self.components_ = v_np[0].T
        else:
            self.loadings_ = v_np
            self.components_ = [v.T for v in v_np]

        self.v_ = self.loadings_
        return self

    def transform(self, X: Any) -> np.ndarray:
        """
        Apply dimensionality reduction to X using learned basis.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of array-like
            New data to transform.

        Returns
        -------
        Z : np.ndarray of shape (n_samples, n_components)
            Projected latent scores.
        """
        if not hasattr(self, "result_"):
            raise ValueError("This SiMLREstimator instance is not fitted yet. Call 'fit' before using this method.")

        tensors, is_single, _ = _parse_input_views(X)
        if is_single and self.is_single_view_:
            # Fast direct projection for single view
            x_mat = tensors[0].numpy()
            return x_mat @ self.loadings_

        # Out-of-sample prediction via consensus anchor
        pred_u = predict_shared_latent(tensors, self.result_)
        return pred_u.detach().cpu().numpy()

    def fit_transform(self, X: Any, y: Optional[Any] = None) -> np.ndarray:
        """
        Fit model and transform X in a single pass.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or list of array-like
            Data to fit and transform.
        y : Ignored

        Returns
        -------
        Z : np.ndarray of shape (n_samples, n_components)
            Projected latent scores.
        """
        self.fit(X, y)
        if self.is_single_view_:
            return self.transform(X)
        if isinstance(self.u_, list):
            return self.u_[0]
        return self.u_

    def inverse_transform(self, Z: np.ndarray) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Transform data back to its original feature space.

        Parameters
        ----------
        Z : array-like of shape (n_samples, n_components)
            Latent coordinates.

        Returns
        -------
        X_reconstructed : np.ndarray or list of np.ndarray
            Reconstructed data in original feature space.
        """
        if not hasattr(self, "components_"):
            raise ValueError("This SiMLREstimator instance is not fitted yet.")

        Z_np = np.asarray(Z)
        if self.is_single_view_:
            return Z_np @ self.components_
        return [Z_np @ comp for comp in self.components_]


# Aliases for convenience
SiMLRTransformer = SiMLREstimator
SiMLR = SiMLREstimator


class LENDTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer for the LEND (Linear Encoder, Non-linear Decoder) architecture.

    Combines an interpretable linear first-layer encoder with a deep non-linear decoder.

    Parameters
    ----------
    n_components : int, default=2
        Dimensionality of the shared latent space.
    iterations : int, default=10
        Outer block-coordinate descent iterations.
    epochs : int, default=30
        Inner training epochs per iteration.
    energy_type : str, default="acc"
        Consensus similarity metric.
    mixing_algorithm : str, default="newton"
        Latent consensus formation algorithm.
    use_nsa : bool, default=True
        Whether to enforce Stiefel manifold constraints via NSA-Flow.
    nsa_w : float, default=0.1 (the wrapped function's default)
        NSA-Flow trade-off parameter.
    consolidate : bool, default=True
        Whether to consolidate supports for strictly disjoint modules.
    verbose : bool, default=False
        Whether to log progress.
    random_state : int, optional
        RNG seed.
    """

    def __init__(
        self,
        n_components: int = 2,
        iterations: int = 10,
        epochs: int = 30,
        energy_type: str = "acc",
        mixing_algorithm: str = "newton",
        use_nsa: bool = True,
        nsa_w: float = 0.1,
        consolidate: bool = True,
        verbose: bool = False,
        random_state: Optional[int] = None,
    ):
        self.n_components = n_components
        self.iterations = iterations
        self.epochs = epochs
        self.energy_type = energy_type
        self.mixing_algorithm = mixing_algorithm
        self.use_nsa = use_nsa
        self.nsa_w = nsa_w
        self.consolidate = consolidate
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X: Any, y: Optional[Any] = None) -> "LENDTransformer":
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        tensors, is_single, feat_counts = _parse_input_views(X)
        self.is_single_view_ = is_single
        self.feature_counts_ = feat_counts
        self.n_features_in_ = feat_counts[0] if is_single else feat_counts

        res = lend_simr(
            tensors,
            k=self.n_components,
            iterations=self.iterations,
            epochs=self.epochs,
            energy_type=self.energy_type,
            mixing_algorithm=self.mixing_algorithm,
            use_nsa=self.use_nsa,
            nsa_w=self.nsa_w,
            verbose=self.verbose,
        )
        self.result_ = res
        v_np = [v.detach().cpu().numpy() for v in res["v"]]

        if is_single:
            self.loadings_ = v_np[0]
            self.components_ = v_np[0].T
        else:
            self.loadings_ = v_np
            self.components_ = [v.T for v in v_np]

        self.v_ = self.loadings_
        return self

    def transform(self, X: Any) -> np.ndarray:
        if not hasattr(self, "result_"):
            raise ValueError("This LENDTransformer instance is not fitted yet.")
        tensors, is_single, _ = _parse_input_views(X)
        if is_single and self.is_single_view_:
            return tensors[0].numpy() @ self.loadings_
        pred = predict_deep(tensors, self.result_, device="cpu")
        return pred["u"].detach().cpu().numpy()


class NEDTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer for the NED (Non-linear Encoder, Non-linear Decoder) architecture.
    """

    def __init__(
        self,
        n_components: int = 2,
        iterations: int = 10,
        epochs: int = 30,
        energy_type: str = "acc",
        mixing_algorithm: str = "newton",
        use_nsa: bool = True,
        nsa_w: float = 0.1,
        verbose: bool = False,
        random_state: Optional[int] = None,
    ):
        self.n_components = n_components
        self.iterations = iterations
        self.epochs = epochs
        self.energy_type = energy_type
        self.mixing_algorithm = mixing_algorithm
        self.use_nsa = use_nsa
        self.nsa_w = nsa_w
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X: Any, y: Optional[Any] = None) -> "NEDTransformer":
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        tensors, is_single, feat_counts = _parse_input_views(X)
        self.is_single_view_ = is_single
        self.feature_counts_ = feat_counts
        self.n_features_in_ = feat_counts[0] if is_single else feat_counts

        res = ned_simr(
            tensors,
            k=self.n_components,
            iterations=self.iterations,
            epochs=self.epochs,
            energy_type=self.energy_type,
            mixing_algorithm=self.mixing_algorithm,
            use_nsa=self.use_nsa,
            nsa_w=self.nsa_w,
            verbose=self.verbose,
        )
        self.result_ = res
        v_np = [v.detach().cpu().numpy() for v in res["v"]]

        if is_single:
            self.loadings_ = v_np[0]
            self.components_ = v_np[0].T
        else:
            self.loadings_ = v_np
            self.components_ = [v.T for v in v_np]

        self.v_ = self.loadings_
        return self

    def transform(self, X: Any) -> np.ndarray:
        if not hasattr(self, "result_"):
            raise ValueError("This NEDTransformer instance is not fitted yet.")
        tensors, is_single, _ = _parse_input_views(X)
        if is_single and self.is_single_view_:
            return tensors[0].numpy() @ self.loadings_
        pred = predict_deep(tensors, self.result_, device="cpu")
        return pred["u"].detach().cpu().numpy()


class FlowSiMLRTransformer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer for the Flow-SiMLR-V bijective normalizing flow architecture.
    """

    def __init__(
        self,
        n_components: int = 2,
        epochs: int = 30,
        energy_type: str = "regression",
        mixing_algorithm: str = "newton",
        use_nsa: bool = True,
        nsa_w: float = 0.1,
        verbose: bool = False,
        random_state: Optional[int] = None,
    ):
        self.n_components = n_components
        self.epochs = epochs
        self.energy_type = energy_type
        self.mixing_algorithm = mixing_algorithm
        self.use_nsa = use_nsa
        self.nsa_w = nsa_w
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X: Any, y: Optional[Any] = None) -> "FlowSiMLRTransformer":
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        tensors, is_single, feat_counts = _parse_input_views(X)
        self.is_single_view_ = is_single
        self.feature_counts_ = feat_counts
        self.n_features_in_ = feat_counts[0] if is_single else feat_counts

        res = flow_simr_v(
            tensors,
            k=self.n_components,
            epochs=self.epochs,
            energy_type=self.energy_type,
            mixing_algorithm=self.mixing_algorithm,
            use_nsa=self.use_nsa,
            nsa_w=self.nsa_w,
            verbose=self.verbose,
        )
        self.result_ = res
        v_np = [v.detach().cpu().numpy() for v in res["v"]]

        if is_single:
            self.loadings_ = v_np[0]
            self.components_ = v_np[0].T
        else:
            self.loadings_ = v_np
            self.components_ = [v.T for v in v_np]

        self.v_ = self.loadings_
        return self

    def transform(self, X: Any) -> np.ndarray:
        if not hasattr(self, "result_"):
            raise ValueError("This FlowSiMLRTransformer instance is not fitted yet.")
        tensors, is_single, _ = _parse_input_views(X)
        if is_single and self.is_single_view_:
            return tensors[0].numpy() @ self.loadings_
        pred = predict_deep(tensors, self.result_, device="cpu")
        return pred["u"].detach().cpu().numpy()


def build_simlr_pipeline(
    n_components: int = 6,
    task: str = "classification",
    model: Optional[Any] = None,
    optimizer_type: str = "torch_lbfgs",
    consolidate: bool = True,
    scale: bool = True,
    **simlr_kwargs: Any,
) -> Pipeline:
    """
    Build a turnkey scikit-learn Pipeline with SiMLR dimensionality reduction.

    Adapts automatically to tabular data distribution and utilizes torch_lbfgs
    quasi-Newton optimization for fast, superlinear convergence.

    Parameters
    ----------
    n_components : int, default=6
        Number of latent components to extract.
    task : {"classification", "regression"}, default="classification"
        Task type used to pick default downstream predictor if `model` is None.
    model : estimator, optional
        Downstream scikit-learn estimator. If None, defaults to LogisticRegression(C=1.0, max_iter=500)
        for classification, or Ridge(alpha=1.0) for regression.
    optimizer_type : str, default="torch_lbfgs"
        Optimizer for basis learning (e.g. "torch_lbfgs" or "hybrid_adam").
    consolidate : bool, default=True
        Whether to enforce strictly disjoint supports with zero crosstalk.
    scale : bool, default=True
        Whether to prepend StandardScaler() to the pipeline.
    **simlr_kwargs : Any
        Additional keyword arguments forwarded to SiMLREstimator.

    Returns
    -------
    sklearn.pipeline.Pipeline
        The constructed turnkey pipeline.
    """
    steps = []
    if scale:
        steps.append(("scaler", StandardScaler()))

    kwargs = {
        "optimizer_type": optimizer_type,
        "consolidate": consolidate,
    }
    kwargs.update(simlr_kwargs)
    steps.append(("dim_reduction", SiMLREstimator(n_components=n_components, **kwargs)))

    if model is None:
        if task == "classification":
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(C=1.0, max_iter=500)
            steps.append(("classifier", model))
        elif task == "regression":
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=1.0)
            steps.append(("regressor", model))
        else:
            raise ValueError(f"Unknown task: {task}. Expected 'classification' or 'regression'.")
    else:
        step_name = "classifier" if task == "classification" else "regressor"
        steps.append((step_name, model))

    return Pipeline(steps)
