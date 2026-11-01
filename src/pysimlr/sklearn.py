"""
Scikit-learn interface to pysimlr.

`MultiViewSiMR` is one estimator for every registered method
(`pysimlr.method_names()`), selected by ``method``. Every parameter is
explicit, so `clone`, `get_params` and `GridSearchCV` work, and a grid over
``method`` compares methods::

    from sklearn.model_selection import GridSearchCV
    X = np.hstack([X1, X2])                       # sklearn CV slices rows of one array
    est = MultiViewSiMR(n_components=3, view_sizes=[X1.shape[1], X2.shape[1]])
    GridSearchCV(est, {"method": ["simlr", "lend", "flow_v"]}, cv=3).fit(X)

A parameter left at ``None`` takes the selected method's own default, so one
grid can span methods whose defaults differ. A parameter set to a value that
the selected method cannot use is recorded in ``parameters_ignored_`` and
warned about once (``strict=True`` raises instead); see `pysimlr.fit`.

`MultiViewPCA` (independent PCA per view) and `build_simlr_pipeline` are
also provided.
"""
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import api

#: `MultiViewSiMR` constructor arguments forwarded to `pysimlr.fit` when not
#: None (``n_components`` is passed as ``k``).
_FORWARDED = ("iterations", "energy_type", "mixing_algorithm", "positivity", "sparseness",
              "nsa_w", "constraint", "optimizer_type", "learning_rate", "topology", "varimax",
              "initialization_type", "hidden_dims", "dropout", "batch_size", "private_k",
              "num_layers", "hidden_dim", "device", "tol", "patience", "random_state")


def _split_views(X: Any, view_sizes: Optional[Sequence[int]]) -> List[torch.Tensor]:
    """A list of views from a list of matrices, or from one matrix and its column sizes."""
    if isinstance(X, (list, tuple)):
        if view_sizes is not None:
            raise ValueError("view_sizes splits a single matrix; X is already a list of views.")
        return api.as_views(X)
    full = api.as_views(X)[0]
    if view_sizes is None:
        return [full]
    sizes = [int(s) for s in view_sizes]
    if sum(sizes) != full.shape[1]:
        raise ValueError(f"view_sizes {sizes} sum to {sum(sizes)}, but X has {full.shape[1]} columns.")
    return list(torch.split(full, sizes, dim=1))


class MultiViewSiMR(BaseEstimator, TransformerMixin):
    """
    Scikit-learn transformer for every pysimlr multi-view method.

    Parameters
    ----------
    method : str, default="simlr"
        A name from `pysimlr.method_names()` or an alias.
    n_components : int, default=2
        Dimensionality ``k`` of the shared latent space.
    iterations, energy_type, mixing_algorithm, positivity, sparseness, nsa_w,
    constraint, optimizer_type, learning_rate, topology, varimax,
    initialization_type, hidden_dims, dropout, batch_size, private_k,
    num_layers, hidden_dim, device, tol, patience, random_state :
        Forwarded to `pysimlr.fit` when not ``None``; ``None`` means the
        selected method's default (`pysimlr.describe_method`).
        ``random_state`` seeds the fit on a private RNG stream; the global
        RNG state is not touched.
    scale : bool, optional
        Standardize each view with its training mean and standard deviation.
        ``None`` uses the method's default (all methods standardize by
        default). The deep and flow models always standardize; ``False`` is
        recorded as ignored for them.
    extra_params : dict, optional
        Any other parameter of the selected method.
    view_sizes : list of int, optional
        Column counts of the views when ``X`` is one concatenated matrix (the
        form scikit-learn's cross-validation can slice). ``None`` with a
        single matrix means one view.
    strict : bool, default=False
        Raise instead of recording a parameter the method cannot use.
    verbose : bool, default=False

    Attributes
    ----------
    result_ : dict
        The `pysimlr.fit` result.
    loadings_ : list of ndarray, or ndarray for single-view input
        Per-view first-layer bases ``V_m`` of shape ``(p_m, k)``.
    components_ : list of ndarray, or ndarray
        ``V_m.T``, scikit-learn's ``(k, p_m)`` convention.
    parameters_used_, parameters_ignored_ : dict
        What reached the method, and what did not and why.
    family_, budget_unit_ : str
    n_features_in_ : int
        Total columns across views.
    """

    def __init__(self, method: str = "simlr", n_components: int = 2,
                 iterations: Optional[int] = None, energy_type: Optional[str] = None,
                 mixing_algorithm: Optional[str] = None, positivity: Optional[str] = None,
                 sparseness: Optional[Union[float, List[Optional[float]]]] = None,
                 nsa_w: Optional[float] = None, constraint: Optional[str] = None,
                 optimizer_type: Optional[str] = None, learning_rate: Optional[Union[float, str]] = None,
                 topology: Optional[str] = None, varimax: Optional[bool] = None,
                 initialization_type: Optional[str] = None,
                 hidden_dims: Optional[Sequence[int]] = None, dropout: Optional[float] = None,
                 batch_size: Optional[int] = None, private_k: Optional[int] = None,
                 num_layers: Optional[int] = None, hidden_dim: Optional[int] = None,
                 device: Optional[str] = None, tol: Optional[float] = None,
                 patience: Optional[int] = None, random_state: Optional[int] = None,
                 scale: Optional[bool] = None,
                 extra_params: Optional[Dict[str, Any]] = None,
                 view_sizes: Optional[Sequence[int]] = None, strict: bool = False,
                 verbose: bool = False):
        self.method = method
        self.n_components = n_components
        self.iterations = iterations
        self.energy_type = energy_type
        self.mixing_algorithm = mixing_algorithm
        self.positivity = positivity
        self.sparseness = sparseness
        self.nsa_w = nsa_w
        self.constraint = constraint
        self.optimizer_type = optimizer_type
        self.learning_rate = learning_rate
        self.topology = topology
        self.varimax = varimax
        self.initialization_type = initialization_type
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.batch_size = batch_size
        self.private_k = private_k
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.device = device
        self.tol = tol
        self.patience = patience
        self.random_state = random_state
        self.scale = scale
        self.extra_params = extra_params
        self.view_sizes = view_sizes
        self.strict = strict
        self.verbose = verbose

    def _fit_params(self) -> Dict[str, Any]:
        params = {name: getattr(self, name) for name in _FORWARDED if getattr(self, name) is not None}
        if self.scale is not None:
            # `simlr` takes a preprocessing list, the joint baselines a flag;
            # the deep and flow models always standardize, so for them the
            # setting is recorded as ignored.
            accepts = api.resolve_method(self.method).accepts
            if "scale" in accepts:
                params["scale"] = bool(self.scale)
            else:
                params["scale_list"] = ["centerAndScale", "np"] if self.scale else ["none"]
        if self.verbose:
            params["verbose"] = True
        overlap = set(params) & set(self.extra_params or {})
        if overlap:
            raise ValueError(f"{sorted(overlap)} given both as constructor arguments and in extra_params.")
        params.update(self.extra_params or {})
        return params

    def fit(self, X: Any, y: Optional[Any] = None) -> "MultiViewSiMR":
        """Fit the selected method. ``y`` is ignored."""
        views = _split_views(X, self.view_sizes)
        res = api.fit(self.method, views, int(self.n_components), strict=self.strict,
                      **self._fit_params())
        self.result_ = res
        self.family_ = res["family"]
        self.budget_unit_ = res["parameters"]["budget_unit"]
        self.parameters_used_ = dict(res["parameters"]["used"])
        self.parameters_ignored_ = dict(res["parameters"]["ignored"])
        self.view_sizes_ = [int(v.shape[1]) for v in views]
        self.n_views_ = len(views)
        self.n_features_in_ = int(sum(self.view_sizes_))
        v_np = [v.detach().cpu().numpy() for v in res["v"]]
        self.input_was_array_ = not isinstance(X, (list, tuple))
        single = self.n_views_ == 1 and self.input_was_array_
        self.loadings_ = v_np[0] if single else v_np
        self.components_ = v_np[0].T if single else [v.T for v in v_np]
        return self

    def _check_fitted(self):
        if not hasattr(self, "result_"):
            raise ValueError(f"This {type(self).__name__} instance is not fitted yet; call 'fit' first.")

    def transform(self, X: Any) -> np.ndarray:
        """Shared latent ``u`` of shape ``(n, k)`` through the method's own predictor
        (the same preprocessing and consensus as at fit time)."""
        self._check_fitted()
        out = api.predict(self.result_, _split_views(X, self.view_sizes))
        return out["u"].detach().cpu().numpy()

    def score(self, X: Any, y: Optional[Any] = None) -> float:
        """Held-out reconstruction R^2, averaged over views. ``y`` is ignored.

        Per view, ``1 - ||X_m - X_hat_m||^2 / ||X_m||^2`` with ``X_m``
        standardized by the *training* mean and standard deviation
        (`pysimlr.predict`'s ``r2``): variance explained relative to the
        training mean, on the same per-feature scale for every method, so a
        grid over ``method`` compares like with like."""
        self._check_fitted()
        out = api.predict(self.result_, _split_views(X, self.view_sizes))
        return float(np.mean(out["r2"]))

    def inverse_transform(self, Z: np.ndarray) -> Union[np.ndarray, List[np.ndarray]]:
        """Reconstruct the input from latent coordinates, in input units.

        Defined for the linear methods (``X_m ~ Z W_m``) and the joint
        baselines (``X ~ Z V'``). The reconstruction is mapped back through
        the training preprocessing (`utils.invert_preprocessing`), so
        ``inverse_transform(transform(X))`` approximates ``X`` itself. Returns
        one array shaped like the input when the input was one array (a single
        view or a ``view_sizes`` concatenation), a list of arrays for list
        input. Deep and flow models reconstruct through their decoders in
        `pysimlr.predict`; there the latent alone is not the decoder's input.
        """
        self._check_fitted()
        res = self.result_
        z = torch.as_tensor(np.asarray(Z)).float()
        if self.family_ == "linear":
            from .simlr import reconstruct_from_learned_maps
            recs = reconstruct_from_learned_maps(z, res)
        elif self.family_ == "baseline":
            rec = z.double() @ res["v_tot"].double().T
            recs = list(torch.split(rec.float(), self.view_sizes_, dim=1))
        else:
            raise NotImplementedError(f"inverse_transform is defined for linear methods and the joint "
                                      f"baselines; {self.method!r} is {self.family_!r}.")
        from .utils import invert_preprocessing
        recs = [invert_preprocessing(r.float(), res["scale_list"], prov).detach().cpu().numpy()
                for r, prov in zip(recs, res["provenance_list"])]
        return np.hstack(recs) if self.input_was_array_ else recs


def build_simlr_pipeline(
    n_components: int = 6,
    task: str = "classification",
    model: Optional[Any] = None,
    method: str = "simlr",
    **estimator_params: Any,
) -> Pipeline:
    """
    A scikit-learn Pipeline: `MultiViewSiMR` followed by a downstream
    predictor (LogisticRegression(C=1.0, max_iter=500) for
    ``task="classification"``, Ridge(alpha=1.0) for ``"regression"``, or
    ``model``). ``estimator_params`` go to `MultiViewSiMR`, including
    ``scale`` and ``view_sizes``.

    There is no outer StandardScaler: `MultiViewSiMR` standardizes each view
    with its training statistics, and an outer scaler standardized the data
    a second time (and rejected list-of-views input).
    """
    if task not in ("classification", "regression"):
        raise ValueError(f"Unknown task: {task}. Expected 'classification' or 'regression'.")
    steps = [("dim_reduction", MultiViewSiMR(method=method, n_components=n_components,
                                              **estimator_params))]
    if model is None:
        if task == "classification":
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(C=1.0, max_iter=500)
        else:
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=1.0)
    steps.append(("classifier" if task == "classification" else "regressor", model))
    return Pipeline(steps)


class MultiViewPCA(BaseEstimator, TransformerMixin):
    """
    Multi-View Principal Component Analysis (Multi-View PCA).

    Extracts independent, orthonormal principal components per modality view.
    Unlike standard concatenated PCA (where large-scale or high-variance modalities
    can dominate the top eigenvectors), Multi-View PCA guarantees balanced representation
    by decomposing each view into its own orthogonal subspace:

        Z_m = (X_m - mu_m) @ V_m,   where V_m^T V_m = I_k

    Parameters
    ----------
    n_components : int or List[int], default=6
        Number of principal components to extract per view. If an integer, the same
        number of components (k) is extracted from every view. If a list of integers,
        specifies per-view component counts [k_1, k_2, ...].
    scale : bool, default=True
        Whether to center and standardize features per view prior to SVD.
    whiten : bool, default=False
        Whether to scale projections to unit variance.
    concat : bool, default=True
        Whether transform() returns a single concatenated 2D numpy array [Z_1, Z_2, ...]
        (shape: [N, sum(k_m)]) or a list of 2D numpy arrays.
    random_state : int, optional
        Seed for reproducible SVD / sign convention.
    view_sizes : list of int, optional
        Column counts of the views when ``X`` is one concatenated matrix (the
        form scikit-learn's cross-validation can slice).

    Attributes
    ----------
    loadings_ : List[np.ndarray]
        List of orthogonal loading matrices V_m of shape (p_m, k_m) per view.
    components_ : List[np.ndarray]
        List of component row vectors V_m.T of shape (k_m, p_m) per view.
    explained_variance_ : List[np.ndarray]
        Variance explained by each principal component per view.
    explained_variance_ratio_ : List[np.ndarray]
        Percentage of variance explained by each component per view.
    mean_ : List[np.ndarray]
        Per-feature empirical mean per view.
    scale_ : List[np.ndarray]
        Per-feature empirical standard deviation per view.
    """

    def __init__(
        self,
        n_components: Union[int, List[int]] = 6,
        scale: bool = True,
        whiten: bool = False,
        concat: bool = True,
        random_state: Optional[int] = None,
        view_sizes: Optional[Sequence[int]] = None,
    ) -> None:
        self.n_components = n_components
        self.scale = scale
        self.whiten = whiten
        self.concat = concat
        self.random_state = random_state
        self.view_sizes = view_sizes

    def fit(self, X: Any, y: Optional[Any] = None) -> "MultiViewPCA":
        """
        Fit Multi-View PCA on the input views.

        Parameters
        ----------
        X : list of array-like, or array-like
            If a list/tuple of arrays, each element is a view matrix of shape (n_samples, n_features_m).
            If a single 2D array, treated as a single view.
        y : ignored

        Returns
        -------
        self : MultiViewPCA
            Fitted estimator.
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        tensors = _split_views(X, self.view_sizes)
        is_single = len(tensors) == 1 and not isinstance(X, (list, tuple))
        feature_counts = [int(t.shape[1]) for t in tensors]
        self.is_single_view_ = is_single
        self.n_views_ = len(tensors)
        self.feature_counts_ = feature_counts

        if isinstance(self.n_components, (int, np.integer)):
            k_list = [int(self.n_components)] * self.n_views_
        else:
            k_list = [int(k) for k in self.n_components]
            if len(k_list) != self.n_views_:
                raise ValueError(
                    f"n_components list length ({len(k_list)}) must match number of views ({self.n_views_})."
                )
        self.k_list_ = k_list

        self.scalers_: List[Optional[StandardScaler]] = []
        self.pca_models_: List[PCA] = []
        self.loadings_: List[np.ndarray] = []
        self.components_: List[np.ndarray] = []
        self.explained_variance_: List[np.ndarray] = []
        self.explained_variance_ratio_: List[np.ndarray] = []
        self.mean_: List[np.ndarray] = []
        self.scale_: List[np.ndarray] = []

        for m_idx, (t_view, k) in enumerate(zip(tensors, k_list)):
            X_m = t_view.numpy()
            n_samples, n_features = X_m.shape
            k_eff = min(k, n_samples, n_features)

            if self.scale:
                scaler = StandardScaler()
                X_m_proc = scaler.fit_transform(X_m)
                self.scalers_.append(scaler)
                self.mean_.append(scaler.mean_)
                self.scale_.append(scaler.scale_)
            else:
                self.scalers_.append(None)
                self.mean_.append(np.mean(X_m, axis=0))
                self.scale_.append(np.std(X_m, axis=0))
                X_m_proc = X_m - self.mean_[-1]

            pca = PCA(
                n_components=k_eff,
                whiten=self.whiten,
                random_state=self.random_state,
            )
            pca.fit(X_m_proc)
            self.pca_models_.append(pca)

            # Resolve sign ambiguity deterministically: positive column sum
            v = pca.components_.T
            for col in range(v.shape[1]):
                if np.sum(v[:, col]) < 0:
                    v[:, col] = -v[:, col]
            
            self.loadings_.append(v)
            self.components_.append(v.T)
            self.explained_variance_.append(pca.explained_variance_)
            self.explained_variance_ratio_.append(pca.explained_variance_ratio_)

        return self

    def transform(self, X: Any) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Apply dimensionality reduction to X across all views.

        Parameters
        ----------
        X : list of array-like, or array-like
            Views to transform.

        Returns
        -------
        Z : np.ndarray or List[np.ndarray]
            If concat=True, concatenated latent matrix of shape (n_samples, sum(k_m)).
            If concat=False, list of projection matrices [Z_1, Z_2, ...].
        """
        if not hasattr(self, "pca_models_"):
            raise ValueError("MultiViewPCA instance is not fitted yet.")

        tensors = _split_views(X, self.view_sizes)
        if len(tensors) != self.n_views_:
            raise ValueError(
                f"Expected {self.n_views_} views, got {len(tensors)}."
            )

        projections = []
        for m_idx, (t_view, scaler, pca, v) in enumerate(
            zip(tensors, self.scalers_, self.pca_models_, self.loadings_)
        ):
            X_m = t_view.numpy()
            if scaler is not None:
                X_m_proc = scaler.transform(X_m)
            else:
                X_m_proc = X_m - self.mean_[m_idx]
            
            # Use deterministic sign-resolved loadings
            z_m = X_m_proc @ v
            if self.whiten:
                # Divide by singular value std
                std = np.std(z_m, axis=0, keepdims=True)
                std[std < 1e-10] = 1.0
                z_m = z_m / std
            projections.append(z_m)

        if self.concat:
            return np.hstack(projections)
        return projections

    def fit_transform(self, X: Any, y: Optional[Any] = None) -> Union[np.ndarray, List[np.ndarray]]:
        """Fit the model with X and apply the dimensionality reduction on X."""
        return self.fit(X, y).transform(X)

