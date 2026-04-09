
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ProductionICAOutput:
    sources: torch.Tensor
    projected: Optional[torch.Tensor] = None
    whitened: Optional[torch.Tensor] = None
    whitening_matrix: Optional[torch.Tensor] = None
    rotation_matrix: Optional[torch.Tensor] = None
    effective_unmixing: Optional[torch.Tensor] = None
    mean: Optional[torch.Tensor] = None
    covariance: Optional[torch.Tensor] = None


class FeatureQueue(nn.Module):
    """
    Fixed-capacity FIFO queue of projected features used only for statistics.
    """

    def __init__(self, feature_dim: int, capacity: int, dtype: torch.dtype = torch.float64) -> None:
        super().__init__()
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive")
        if capacity <= 1:
            raise ValueError("capacity must be > 1")

        self.feature_dim = int(feature_dim)
        self.capacity = int(capacity)
        self.dtype = dtype

        self.register_buffer("buffer", torch.zeros(capacity, feature_dim, dtype=dtype))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))
        self.register_buffer("write_ptr", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def enqueue(self, x: torch.Tensor) -> None:
        """
        x: [n, d]
        """
        if x.ndim != 2 or x.shape[1] != self.feature_dim:
            raise ValueError(f"Expected [n, {self.feature_dim}], got {tuple(x.shape)}")

        x = x.detach().to(dtype=self.dtype)
        n = x.shape[0]
        if n == 0:
            return

        if n >= self.capacity:
            self.buffer.copy_(x[-self.capacity :])
            self.count.fill_(self.capacity)
            self.write_ptr.zero_()
            return

        ptr = int(self.write_ptr.item())
        end = ptr + n

        if end <= self.capacity:
            self.buffer[ptr:end] = x
        else:
            first = self.capacity - ptr
            self.buffer[ptr:] = x[:first]
            self.buffer[: end - self.capacity] = x[first:]

        self.write_ptr.fill_(end % self.capacity)
        self.count.fill_(min(self.capacity, int(self.count.item()) + n))

    def get_all(self) -> torch.Tensor:
        count = int(self.count.item())
        if count == 0:
            return self.buffer[:0]
        if count < self.capacity:
            return self.buffer[:count]

        ptr = int(self.write_ptr.item())
        if ptr == 0:
            return self.buffer
        return torch.cat([self.buffer[ptr:], self.buffer[:ptr]], dim=0)


class WhiteningStats(nn.Module):
    """
    Statistics estimator backed by a queue memory bank and EMA.
    """

    def __init__(
        self,
        n_features: int,
        *,
        queue_capacity: int = 4096,
        momentum: float = 0.02,
        shrinkage: float = 0.1,
        eps: float = 1e-5,
        min_samples_for_update: int = 128,
        dtype_stats: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if n_features <= 0:
            raise ValueError("n_features must be positive")
        if not (0.0 <= shrinkage < 1.0):
            raise ValueError("shrinkage must be in [0,1)")
        if min_samples_for_update <= 1:
            raise ValueError("min_samples_for_update must be > 1")

        self.n_features = int(n_features)
        self.momentum = float(momentum)
        self.shrinkage = float(shrinkage)
        self.eps = float(eps)
        self.min_samples_for_update = int(min_samples_for_update)
        self.dtype_stats = dtype_stats

        self.queue = FeatureQueue(
            feature_dim=n_features,
            capacity=queue_capacity,
            dtype=dtype_stats,
        )

        self.register_buffer("running_mean", torch.zeros(n_features, dtype=dtype_stats))
        self.register_buffer("running_cov", torch.eye(n_features, dtype=dtype_stats))
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))

    @torch.no_grad()
    def update_queue(self, x: torch.Tensor) -> None:
        self.queue.enqueue(x)

    @torch.no_grad()
    def recompute_from_queue(self) -> bool:
        x = self.queue.get_all()
        n = x.shape[0]
        if n < self.min_samples_for_update:
            return False

        mean = x.mean(dim=0)
        xc = x - mean
        cov = (xc.T @ xc) / max(n - 1, 1)

        k = cov.shape[0]
        eye = torch.eye(k, device=cov.device, dtype=cov.dtype)
        avg_var = torch.trace(cov) / float(k)

        cov = (1.0 - self.shrinkage) * cov + self.shrinkage * avg_var * eye
        cov = cov + self.eps * eye

        if not bool(self.initialized.item()):
            self.running_mean.copy_(mean)
            self.running_cov.copy_(cov)
            self.initialized.fill_(True)
        else:
            m = self.momentum
            self.running_mean.mul_(1.0 - m).add_(m * mean)
            self.running_cov.mul_(1.0 - m).add_(m * cov)
        return True

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.running_mean, self.running_cov


class MatrixExponentialOrthogonal(nn.Module):
    """
    Orthogonal matrix parametrization:
        Q = exp(A - A^T)
    where A is unconstrained.

    This keeps the ICA rotation orthogonal by construction.
    """

    def __init__(self, n: int) -> None:
        super().__init__()
        if n <= 0:
            raise ValueError("n must be positive")
        self.n = int(n)
        self.raw = nn.Parameter(torch.zeros(n, n))

    def forward(self) -> torch.Tensor:
        skew = self.raw - self.raw.T
        return torch.matrix_exp(skew)


class ProductionStableICALayer(nn.Module):
    """
    Production-oriented ICA layer for small-batch settings.

    Pipeline:
        x -> projector -> whitening from memory-bank stats -> orthogonal ICA rotation

    Key design points:
    - queue-based statistics reduces dependence on current minibatch
    - whitening matrix can be refreshed only periodically
    - rotation is exactly orthogonal by construction
    - stats and eigendecomposition happen in float64
    """

    def __init__(
        self,
        in_features: int,
        n_components: int,
        *,
        projector_bias: bool = False,
        affine: bool = False,
        queue_capacity: int = 4096,
        stats_momentum: float = 0.02,
        whitening_shrinkage: float = 0.1,
        eps: float = 1e-5,
        min_samples_for_stats: int = 128,
        stats_update_interval: int = 8,
        whiten_refresh_interval: int = 8,
        freeze_whitener_steps: int = 0,
        detach_stats_input: bool = True,
        return_state: bool = False,
    ) -> None:
        super().__init__()

        if in_features <= 0 or n_components <= 0:
            raise ValueError("in_features and n_components must be positive")
        if n_components > in_features:
            raise ValueError("n_components must be <= in_features")
        if stats_update_interval <= 0:
            raise ValueError("stats_update_interval must be >= 1")
        if whiten_refresh_interval <= 0:
            raise ValueError("whiten_refresh_interval must be >= 1")
        if freeze_whitener_steps < 0:
            raise ValueError("freeze_whitener_steps must be >= 0")

        self.in_features = int(in_features)
        self.n_components = int(n_components)
        self.eps = float(eps)
        self.stats_update_interval = int(stats_update_interval)
        self.whiten_refresh_interval = int(whiten_refresh_interval)
        self.freeze_whitener_steps = int(freeze_whitener_steps)
        self.detach_stats_input = bool(detach_stats_input)
        self.return_state = bool(return_state)

        self.projector = nn.Linear(in_features, n_components, bias=projector_bias)
        self.rotation = MatrixExponentialOrthogonal(n_components)

        self.stats = WhiteningStats(
            n_features=n_components,
            queue_capacity=queue_capacity,
            momentum=stats_momentum,
            shrinkage=whitening_shrinkage,
            eps=eps,
            min_samples_for_update=min_samples_for_stats,
            dtype_stats=torch.float64,
        )

        if affine:
            self.weight = nn.Parameter(torch.ones(n_components))
            self.bias = nn.Parameter(torch.zeros(n_components))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.register_buffer("_cached_whitener", torch.eye(n_components, dtype=torch.float64))
        self.register_buffer("_step", torch.tensor(0, dtype=torch.long))
        self.register_buffer("_last_whitener_refresh_step", torch.tensor(-1, dtype=torch.long))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.projector.weight)
        if self.projector.bias is not None:
            nn.init.zeros_(self.projector.bias)
        if self.weight is not None:
            nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @torch.no_grad()
    def refresh_whitener(self) -> None:
        mean, cov = self.stats.get()
        evals, evecs = torch.linalg.eigh(cov.to(torch.float64))
        evals = torch.clamp(evals, min=self.eps)
        whitener = evecs @ torch.diag(evals.rsqrt()) @ evecs.T
        self._cached_whitener.copy_(whitener)
        self._last_whitener_refresh_step.copy_(self._step)

    def _should_update_stats(self) -> bool:
        return int(self._step.item()) % self.stats_update_interval == 0

    def _should_refresh_whitener(self) -> bool:
        step = int(self._step.item())
        last = int(self._last_whitener_refresh_step.item())

        if self.freeze_whitener_steps > 0 and last >= 0:
            if step - last < self.freeze_whitener_steps:
                return False

        return step % self.whiten_refresh_interval == 0

    def _maybe_update_statistics(self, z2d: torch.Tensor) -> None:
        if not self.training:
            return

        z_stats = z2d.detach() if self.detach_stats_input else z2d
        self.stats.update_queue(z_stats)

        if self._should_update_stats():
            updated = self.stats.recompute_from_queue()
            if updated and self._should_refresh_whitener():
                self.refresh_whitener()

    def _get_whitener(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self._should_refresh_whitener():
            self.refresh_whitener()
        return self._cached_whitener.to(dtype=dtype, device=device)

    def forward(self, x: torch.Tensor):
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected last dimension {self.in_features}, got {x.shape[-1]}"
            )

        orig_shape = x.shape
        x2d = x.reshape(-1, self.in_features)

        z2d = self.projector(x2d)
        self._maybe_update_statistics(z2d)

        mean, cov = self.stats.get()
        mean = mean.to(dtype=z2d.dtype, device=z2d.device)
        whitener = self._get_whitener(dtype=z2d.dtype, device=z2d.device)

        zw2d = (z2d - mean) @ whitener.T

        q = self.rotation().to(dtype=zw2d.dtype, device=zw2d.device)
        s2d = zw2d @ q.T

        if self.weight is not None:
            s2d = s2d * self.weight + self.bias

        s = s2d.reshape(*orig_shape[:-1], self.n_components)

        self._step += 1

        if not self.return_state:
            return s

        effective_unmixing = q @ whitener @ self.projector.weight
        return ProductionICAOutput(
            sources=s,
            projected=z2d.reshape(*orig_shape[:-1], self.n_components),
            whitened=zw2d.reshape(*orig_shape[:-1], self.n_components),
            whitening_matrix=whitener,
            rotation_matrix=q,
            effective_unmixing=effective_unmixing,
            mean=mean,
            covariance=cov.to(dtype=z2d.dtype, device=z2d.device),
        )

    def independence_loss(
        self,
        sources: torch.Tensor,
        *,
        lambda_offdiag: float = 1.0,
        lambda_var: float = 0.1,
        lambda_kurtosis: float = 0.05,
    ) -> torch.Tensor:
        """
        Auxiliary ICA-style regularizer.

        - off-diagonal covariance penalty
        - variance normalization penalty
        - mild non-Gaussianity encouragement
        """
        s2d = sources.reshape(-1, sources.shape[-1])
        s2d = s2d - s2d.mean(dim=0, keepdim=True)

        n = max(s2d.shape[0], 2)
        cov = (s2d.T @ s2d) / float(n - 1)

        diag = torch.diagonal(cov)
        offdiag = cov - torch.diag(diag)

        loss_offdiag = offdiag.pow(2).mean()
        loss_var = (diag - 1.0).pow(2).mean()

        std = torch.sqrt(torch.clamp(diag, min=self.eps))
        s_norm = s2d / std.unsqueeze(0)
        kurt = (s_norm.pow(4).mean(dim=0) - 3.0).abs().mean()

        return (
            lambda_offdiag * loss_offdiag
            + lambda_var * loss_var
            - lambda_kurtosis * kurt
        )
