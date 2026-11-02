import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Any, Optional, Union, Callable, Tuple, Sequence
import warnings
from abc import ABC, abstractmethod

#: Every hyperparameter any SiMLR optimizer reads, with its default. Keys absent
#: from this mapping are rejected with a warning by
#: :meth:`SimlrOptimizer.filter_params`, so it must stay in sync with the
#: optimizer implementations below.

#: Optimizers that never read `learning_rate` (L-BFGS-B has its own line
#: search). `simlr` skips the
#: learning-rate probes for them: every candidate scored identically, so
#: `learning_rate="auto"` cost seven wasted fits per call.
LEARNING_RATE_FREE_OPTIMIZERS = frozenset({"nsa_lbfgsb"})

SIMLR_OPTIMIZER_DEFAULTS: Dict[str, Any] = {
    'learning_rate': 0.001,
    'beta1': 0.9,          # Adam / Nadam / HybridAdam first moment
    'beta2': 0.999,        # Adam / Nadam / HybridAdam second moment
    'rmsprop_beta': 0.9,   # RMSProp squared-gradient decay (was 'beta', which collided with the flow models' NLL weight)
    'epsilon': 1e-8,
    'weight_decay': 0.0,
    'amsgrad': False,
    'momentum': 0.9,
    'decay_rate': 1e-3,    # LARS weight decay
    'lookahead_k': 5,      # Lookahead slow-weight period (was 'k', which collided with simlr's rank)
    'alpha': 0.5,          # Lookahead slow-weight interpolation
    'max_iter': 20,        # NSA L-BFGS-B maximum iterations per step
    'reparameterize': True, # Quadratic reparameterization for non-negativity
    'positivity': 'either', # Positivity constraint type
    'nonneg': None,         # Explicit nonneg flag for bound-constrained solvers
    'lbfgsb_tol': 1e-9,     # nsa_lbfgsb solver tolerance (was 'tol', which collided with simlr's convergence tol)
}

class SimlrOptimizer(ABC):
    """
    Abstract base class for all SiMLR-specific optimizers.

    Provides a common interface for updating modality-specific basis 
    matrices (V) during SiMLR optimization. Handles parameter filtering 
    and maintains optimizer state (e.g., momentum, second moments).

    Parameters
    ----------
    optimizer_type : str
        The name/type of the optimizer.
    v_mats : List[torch.Tensor]
        The initial basis matrices for each modality.
    **params : Dict[str, Any]
        Hyperparameters for the optimizer (e.g., learning_rate, beta1).

    Raises
    ------
    TypeError
        If inputs are of invalid types.
    """
    def __init__(self, optimizer_type: str, v_mats: List[torch.Tensor], **params):
        self.optimizer_type = optimizer_type
        self.params = self.filter_params(optimizer_type, params)
        self.state = []
        for v in v_mats:
            self.state.append({
                'm': torch.zeros_like(v),
                'v': torch.zeros_like(v),
                'v_max': torch.zeros_like(v),
                'iter': 0,
                'momentum': torch.zeros_like(v),
                # Seeded from the configured learning rate, not a constant.
                # It was a hardcoded 0.01, and because the key therefore always
                # exists the `state.get('last_step_size', torch.tensor(lr))`
                # fallbacks in the line-search optimizers never fired -- so
                # `learning_rate` was read only in their `full_energy_function
                # is None` branch, which `simlr` never takes because it always
                # passes one. Measured across lr from 1e-4 to 1.0, the fitted
                # basis was bit-identical for hybrid_adam, armijo_gradient,
                # bidirectional_armijo_gradient, lookahead and
                # bidirectional_lookahead: five optimizers whose documented
                # "initial step size for the line search" did nothing.
                'last_step_size': torch.tensor(
                    float(self.params.get('learning_rate', 0.01)) or 0.01)
            })

    def filter_params(self, optimizer_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge caller-supplied hyperparameters over the defaults.

        Parameters
        ----------
        optimizer_type : str
            The type of optimizer, used only for the warning message.
        params : Dict[str, Any]
            Caller-supplied hyperparameters.

        Returns
        -------
        Dict[str, Any]
            Every key in `SIMLR_OPTIMIZER_DEFAULTS`, overridden where the caller
            supplied a value.

        Raises
        ------
        TypeError
            If `params` contains a key no optimizer recognises -- most often a
            typo. These were first discarded in silence and then only warned
            about, so a misspelled hyperparameter had no effect on the fit.

        Notes
        -----
        Earlier revisions built the result solely from a `defaults` dict that
        omitted `decay_rate`, `beta`, `k` and `alpha`, and then pruned further
        by optimizer type. Because every value was taken from that dict, any
        key missing from it was dropped even when the caller passed it
        explicitly, which made LARS's `decay_rate`, RMSProp's `beta` and
        Lookahead's `k` and `alpha` permanently stuck at their fallbacks. The
        per-optimizer pruning served no purpose, since each `step` reads only
        the keys it needs.
        """
        filtered = dict(SIMLR_OPTIMIZER_DEFAULTS)
        unknown = sorted(set(params) - set(SIMLR_OPTIMIZER_DEFAULTS))
        if unknown:
            raise TypeError(
                f"Unrecognised optimizer parameter(s) {unknown} for "
                f"optimizer_type={optimizer_type!r}. "
                f"Known parameters: {sorted(SIMLR_OPTIMIZER_DEFAULTS)}."
            )
        filtered.update({k: v for k, v in params.items()
                         if k in SIMLR_OPTIMIZER_DEFAULTS})
        return filtered

    @abstractmethod
    def step(self, i: int, v_current: torch.Tensor, descent_gradient: torch.Tensor, 
             full_energy_function: Optional[Callable] = None) -> torch.Tensor:
        """
        Abstract method to perform a single optimization step.

        Parameters
        ----------
        i : int
            The index of the modality.
        v_current : torch.Tensor
            Current basis matrix for the modality.
        descent_gradient : torch.Tensor
            The gradient direction for the update.
        full_energy_function : Optional[Callable], optional
            Function to calculate the energy (used for line search).

        Returns
        -------
        torch.Tensor
            The updated basis matrix.

        Raises
        ------
        NotImplementedError
            If not implemented by a subclass.
        """
        pass

def backtracking_linesearch(v_current: torch.Tensor, 
                            descent_direction: torch.Tensor, 
                            ascent_gradient: torch.Tensor,
                            energy_function: Callable, 
                            initial_step_size: float = 1.0,
                            alpha: float = 1e-4, 
                            beta: float = 0.5, 
                            max_iter: int = 10,
                            min_step: float = 1e-12) -> float:
    """
    Find an optimal step size using the Armijo backtracking line search rule.

    Ensures that the step taken along the search direction results in a 
    sufficient decrease of the energy function relative to the gradient slope.

    Parameters
    ----------
    v_current : torch.Tensor
        Current value of the parameters (basis matrix).
    descent_direction : torch.Tensor
        The direction along which to search (e.g., negative gradient).
    ascent_gradient : torch.Tensor
        The gradient at the current position (used for slope calculation).
    energy_function : Callable
        Function that computes the energy or loss for a given parameter set.
    initial_step_size : float, default=1.0
        The first step size to try.
    alpha : float, default=1e-4
        Sufficient decrease constant (Armijo parameter).
    beta : float, default=0.5
        Reduction factor for the step size in each iteration.
    max_iter : int, default=10
        Maximum number of backtracking steps.
    min_step : float, default=1e-12
        Minimum allowable step size.

    Returns
    -------
    float
        The optimal step size found.

    Raises
    ------
    TypeError
        If inputs are of invalid types.
    """
    # The current iterate must be evaluable; a failure here is a bug upstream,
    # not a step size of zero (which is what it used to return, silently).
    initial_energy = energy_function(v_current)
        
    step_size = initial_step_size
    slope_term = torch.sum(ascent_gradient * descent_direction)
    if slope_term >= -1e-12:
        return 0.0
    
    for _ in range(max_iter):
        if step_size <= min_step:
            break
        v_candidate = v_current + step_size * descent_direction
        try:
            new_energy = energy_function(v_candidate)
        except (RuntimeError, ValueError, ArithmeticError):
            # A candidate the energy cannot be evaluated at (e.g. a degenerate
            # basis) is rejected like one that increases it.
            new_energy = float('inf')
        if new_energy <= initial_energy + alpha * step_size * slope_term:
            return step_size
        step_size *= beta
    return 0.0

def bidirectional_linesearch(v_current: torch.Tensor, 
                             descent_direction: torch.Tensor, 
                             ascent_gradient: torch.Tensor,
                             energy_function: Callable, 
                             initial_step_size: float = 1.0,
                             alpha: float = 1e-4, 
                             beta: float = 0.5, 
                             max_iter: int = 10,
                             min_step: float = 1e-12) -> Tuple[float, torch.Tensor]:
    """
    Perform a backtracking line search in both the positive and negative directions.

    Useful for complex energy landscapes where the initial descent direction 
    (from the gradient) might be misleading or when the search direction 
    is not strictly a descent direction.

    Parameters
    ----------
    v_current : torch.Tensor
        Current value of the parameters.
    descent_direction : torch.Tensor
        Primary search direction.
    ascent_gradient : torch.Tensor
        Gradient at current position.
    energy_function : Callable
        Function to minimize.
    initial_step_size : float, default=1.0
        Initial step to try in both directions.
    alpha : float, default=1e-4
        Armijo constant.
    beta : float, default=0.5
        Step reduction factor.
    max_iter : int, default=10
        Max steps per direction.
    min_step : float, default=1e-12
        Minimum step size.

    Returns
    -------
    Tuple[float, torch.Tensor]
        A tuple of (optimal_step_size, direction), where direction is 
        either `descent_direction` or `-descent_direction`. The direction is
        chosen by which candidate reaches the lower energy, not by which
        admits the larger step.

    Raises
    ------
    TypeError
        If inputs are of invalid types.
    """
    # Try positive direction
    pos_step = backtracking_linesearch(
        v_current, descent_direction, ascent_gradient, energy_function,
        initial_step_size, alpha, beta, max_iter, min_step
    )
    
    # Try negative direction
    neg_step = backtracking_linesearch(
        v_current, -descent_direction, -ascent_gradient, energy_function,
        initial_step_size, alpha, beta, max_iter, min_step
    )

    # Pick whichever candidate actually reaches the lower energy. Comparing the
    # two *step sizes* (as this previously did, via `pos_step >= neg_step`) says
    # nothing about which direction descends further: a larger admissible step
    # in one direction can easily land above a smaller step in the other.
    def _energy_at(step, direction):
        if step <= 0.0:
            return float('inf')
        try:
            return float(energy_function(v_current + step * direction))
        except (RuntimeError, ValueError, ArithmeticError):
            # Rejected like a candidate that increases the energy.
            return float('inf')

    pos_energy = _energy_at(pos_step, descent_direction)
    neg_energy = _energy_at(neg_step, -descent_direction)

    if pos_energy <= neg_energy and pos_step > 0:
        return pos_step, descent_direction
    if neg_step > 0 and neg_energy < float('inf'):
        return neg_step, -descent_direction
    if pos_step > 0:
        return pos_step, descent_direction
    return 0.0, descent_direction

class HybridAdam(SimlrOptimizer):
    """
    Hybrid optimizer combining Adam-style momentum with line search.

    Computes an Adam-like search direction (using max variance for stability) 
    and then performs a backtracking line search along that direction to 
    ensure a sufficient decrease in the energy function.

    Parameters
    ----------
    optimizer_type : str
        The type of optimizer.
    v_mats : List[torch.Tensor]
        Initial matrices.
    learning_rate : float, default=0.01
        Initial step size.
    beta1 : float, default=0.9
        Momentum decay.
    beta2 : float, default=0.999
        Variance decay.
    epsilon : float, default=1e-8
        Numerical stability constant.

    Raises
    ------
    TypeError
        If inputs are of invalid types.
    """
    def step(self, i: int, v_current: torch.Tensor, descent_gradient: torch.Tensor, 
             full_energy_function: Optional[Callable] = None) -> torch.Tensor:
        state = self.state[i]
        beta1 = self.params['beta1']
        beta2 = self.params['beta2']
        epsilon = self.params['epsilon']
        state['m'] = beta1 * state['m'] + (1 - beta1) * descent_gradient
        state['v'] = beta2 * state['v'] + (1 - beta2) * (descent_gradient**2)
        # Honour the advertised `amsgrad` flag; the running maximum used to be
        # applied unconditionally, so amsgrad=False had no effect.
        if self.params.get('amsgrad', False):
            state['v_max'] = torch.maximum(state['v_max'], state['v'])
            denom_sq = state['v_max']
        else:
            denom_sq = state['v']
        search_direction = state['m'] / (torch.sqrt(denom_sq) + epsilon)
        grad_norm = torch.norm(descent_gradient)
        if grad_norm > epsilon and torch.sum(-descent_gradient * search_direction) >= 0:
            state['m'] = descent_gradient.clone()
            search_direction = state['m'] / (torch.sqrt(denom_sq) + epsilon)
        if full_energy_function is not None:
            optimal_step_size = backtracking_linesearch(
                v_current=v_current,
                descent_direction=search_direction,
                ascent_gradient=-descent_gradient,
                energy_function=full_energy_function,
                initial_step_size=state['last_step_size'].item()
            )
        else:
            optimal_step_size = self.params['learning_rate']
        state['last_step_size'] = torch.tensor(optimal_step_size * 1.5 if optimal_step_size > 1e-9 else float(self.params['learning_rate']))
        return v_current + optimal_step_size * search_direction

class Adam(SimlrOptimizer):
    """
    Adam (Adaptive Moment Estimation) optimizer for SiMLR.

    Updates basis matrices using estimates of the first and second 
    moments of the gradients.

    Parameters
    ----------
    optimizer_type : str
        The type of optimizer.
    v_mats : List[torch.Tensor]
        Initial matrices.
    learning_rate : float, default=0.01
        Step size for updates.
    beta1 : float, default=0.9
        Exponential decay rate for the first moment estimates.
    beta2 : float, default=0.999
        Exponential decay rate for the second moment estimates.
    epsilon : float, default=1e-8
        Numerical stability constant.

    Raises
    ------
    TypeError
        If inputs are of invalid types.
    """
    def step(self, i: int, v_current: torch.Tensor, descent_gradient: torch.Tensor, 
             full_energy_function: Optional[Callable] = None) -> torch.Tensor:
        state = self.state[i]
        state['iter'] += 1
        beta1 = self.params['beta1']
        beta2 = self.params['beta2']
        epsilon = self.params['epsilon']
        lr = self.params['learning_rate']
        state['m'] = beta1 * state['m'] + (1 - beta1) * descent_gradient
        state['v'] = beta2 * state['v'] + (1 - beta2) * (descent_gradient**2)
        m_hat = state['m'] / (1 - beta1**state['iter'])
        v_hat = state['v'] / (1 - beta2**state['iter'])
        numerator = self._first_moment(m_hat, descent_gradient, beta1, state['iter'])
        return v_current + lr * (numerator / (torch.sqrt(v_hat) + epsilon))

    def _first_moment(self, m_hat, descent_gradient, beta1, t):
        """Bias-corrected first moment; `Nadam` adds the Nesterov look-ahead."""
        return m_hat

class Nadam(Adam):
    """
    Nadam (Nesterov-accelerated Adaptive Moment Estimation) optimizer for SiMLR.

    Combines Adam with Nesterov accelerated gradient (NAG) for potentially 
    faster convergence.

    Parameters
    ----------
    optimizer_type : str
        The type of optimizer.
    v_mats : List[torch.Tensor]
        Initial matrices.
    learning_rate : float, default=0.01
        Step size.
    beta1 : float, default=0.9
        Momentum decay.
    beta2 : float, default=0.999
        Variance decay.
    epsilon : float, default=1e-8
        Numerical stability constant.

    Raises
    ------
    TypeError
        If inputs are of invalid types.
    """
    def _first_moment(self, m_hat, descent_gradient, beta1, t):
        return beta1 * m_hat + ((1 - beta1) * descent_gradient) / (1 - beta1**t)

class ArmijoGradient(SimlrOptimizer):
    """
    Gradient descent optimizer with Armijo-style backtracking line search.

    Uses a line search to find an optimal step size that satisfies the 
    sufficient decrease condition (Armijo rule). Falls back to a constant 
    learning rate if no energy function is provided.

    Parameters
    ----------
    optimizer_type : str
        The type of optimizer.
    v_mats : List[torch.Tensor]
        Initial matrices.
    learning_rate : float, default=0.1
        Initial step size for the line search.
    epsilon : float, default=1e-8
        Numerical stability constant.

    Raises
    ------
    TypeError
        If inputs are of invalid types.
    """
    #: Restart the momentum from the raw direction when it stops being a
    #: descent direction. `BidirectionalArmijoGradient` keeps it, because its
    #: line search may move against the direction anyway.
    reset_non_descent = True

    def _line_search(self, v_current, direction, descent_gradient, energy_fn, initial_step):
        """Return ``(step_size, direction)``; a step of 0 means no decrease was found."""
        step = backtracking_linesearch(
            v_current=v_current, descent_direction=direction,
            ascent_gradient=-descent_gradient, energy_function=energy_fn,
            initial_step_size=initial_step)
        return step, direction

    def step(self, i: int, v_current: torch.Tensor, descent_gradient: torch.Tensor,
             full_energy_function: Optional[Callable] = None) -> torch.Tensor:
        state = self.state[i]
        epsilon = self.params.get('epsilon', 1e-8)
        lr = self.params['learning_rate']
        state['momentum'] = 0.9 * state['momentum'] + 0.1 * descent_gradient
        search_direction = state['momentum']
        if (self.reset_non_descent and torch.norm(descent_gradient) > epsilon
                and torch.sum(-descent_gradient * search_direction) >= 0):
            state['momentum'] = descent_gradient.clone()
            search_direction = state['momentum']
        dir_norm = torch.norm(search_direction)
        if dir_norm < epsilon:
            return v_current
        norm_direction = search_direction / dir_norm
        if full_energy_function is None:
            return v_current + lr * norm_direction
        initial_step = state.get('last_step_size', torch.tensor(lr)).item()
        step_size, direction = self._line_search(v_current, norm_direction, descent_gradient,
                                                 full_energy_function, initial_step)
        state['last_step_size'] = torch.tensor(step_size * 1.5 if step_size > 1e-10 else float(lr))
        return v_current + step_size * direction

class BidirectionalArmijoGradient(ArmijoGradient):
    """
    Gradient descent optimizer with bidirectional Armijo line search.

    Similar to `ArmijoGradient`, but the line search explores both the 
    descent and ascent directions to find the optimal step. This is 
    useful for complex energy surfaces where the sign of the gradient 
    might not immediately point towards the minimum.

    Parameters
    ----------
    optimizer_type : str
        The type of optimizer.
    v_mats : List[torch.Tensor]
        Initial matrices.
    learning_rate : float, default=0.1
        Initial step size for the line search.
    epsilon : float, default=1e-8
        Numerical stability constant.

    Raises
    ------
    TypeError
        If inputs are of invalid types.
    """
    reset_non_descent = False

    def _line_search(self, v_current, direction, descent_gradient, energy_fn, initial_step):
        return bidirectional_linesearch(
            v_current=v_current, descent_direction=direction,
            ascent_gradient=-descent_gradient, energy_function=energy_fn,
            initial_step_size=initial_step)

class Lookahead(SimlrOptimizer):
    """
    Lookahead optimizer using `HybridAdam` as the inner solver.

    Implements the "fast weights" and "slow weights" strategy to improve 
    convergence stability. Slow weights are updated every `k` steps 
    towards the fast weights.

    Parameters
    ----------
    optimizer_type : str
        The type of optimizer.
    v_mats : List[torch.Tensor]
        Initial matrices.
    k : int, default=5
        Frequency of slow weight updates.
    alpha : float, default=0.5
        Step size (interpolation factor) for slow weight updates.
    **params : Dict[str, Any]
        Additional parameters passed to the inner `HybridAdam` optimizer.

    Raises
    ------
    TypeError
        If inputs are of invalid types.
    """
    inner_class = HybridAdam

    def __init__(self, optimizer_type: str, v_mats: List[torch.Tensor], **params):
        super().__init__(optimizer_type, v_mats, **params)
        self.k = int(self.params['lookahead_k'])
        self.alpha = self.params.get('alpha', 0.5)
        self.slow_weights = [v.clone() for v in v_mats]
        self.inner_opt = self.inner_class(optimizer_type, v_mats, **params)

    def step(self, i: int, v_current: torch.Tensor, descent_gradient: torch.Tensor, 
             full_energy_function: Optional[Callable] = None) -> torch.Tensor:
        state = self.state[i]
        state['iter'] += 1
        v_next = self.inner_opt.step(i, v_current, descent_gradient, full_energy_function)
        if state['iter'] % self.k == 0:
            self.slow_weights[i] = self.slow_weights[i] + self.alpha * (v_next - self.slow_weights[i])
            return self.slow_weights[i].clone()
        return v_next

class BidirectionalLookahead(Lookahead):
    """
    Lookahead optimizer using `BidirectionalArmijoGradient` as the inner solver.

    Maintains a set of "slow weights" that are updated every `k` steps towards 
    the "fast weights" generated by the inner bidirectional Armijo optimizer. 
    This improves stability and convergence in complex SiMLR landscapes.

    Parameters
    ----------
    optimizer_type : str
        The type of optimizer.
    v_mats : List[torch.Tensor]
        Initial matrices.
    k : int, default=5
        Frequency of slow weight updates.
    alpha : float, default=0.5
        Step size (interpolation factor) for slow weight updates.
    **params : Dict[str, Any]
        Additional parameters passed to the inner optimizer.

    Raises
    ------
    TypeError
        If inputs are of invalid types.
    """
    inner_class = BidirectionalArmijoGradient


class RMSProp(SimlrOptimizer):
    """
    RMSProp (Root Mean Square Propagation) optimizer for SiMLR.

    Updates basis matrices using a moving average of squared gradients 
    to normalize the gradient magnitude.

    Parameters
    ----------
    optimizer_type : str
        The type of optimizer.
    v_mats : List[torch.Tensor]
        Initial matrices.
    learning_rate : float, default=0.01
        The step size for updates.
    beta : float, default=0.9
        Discounting factor for the history/coming gradient.
    epsilon : float, default=1e-8
        Numerical stability constant.

    Raises
    ------
    TypeError
        If inputs are of invalid types.
    """
    def step(self, i: int, v_current: torch.Tensor, descent_gradient: torch.Tensor, 
             full_energy_function: Optional[Callable] = None) -> torch.Tensor:
        state = self.state[i]
        beta = float(self.params['rmsprop_beta'])
        epsilon = self.params['epsilon']
        lr = self.params['learning_rate']
        state['v'] = beta * state['v'] + (1 - beta) * (descent_gradient**2)
        return v_current + lr * (descent_gradient / (torch.sqrt(state['v']) + epsilon))

class SGD(SimlrOptimizer):
    """
    Stochastic Gradient Descent (SGD) optimizer for SiMLR.

    Updates basis matrices using a constant learning rate.

    Parameters
    ----------
    optimizer_type : str
        The type of optimizer.
    v_mats : List[torch.Tensor]
        Initial matrices.
    learning_rate : float, default=0.01
        The step size for updates.

    Raises
    ------
    TypeError
        If inputs are of invalid types.
    """
    def step(self, i: int, v_current: torch.Tensor, descent_gradient: torch.Tensor, 
             full_energy_function: Optional[Callable] = None) -> torch.Tensor:
        lr = self.params['learning_rate']
        return v_current + lr * descent_gradient

class LARS(SimlrOptimizer):
    """
    Layer-wise Adaptive Rate Scaling (LARS) optimizer for SiMLR.

    Scales the learning rate based on the ratio of the weight norm to the 
    gradient norm, which is helpful for training with large gradients or 
    varying parameter scales.

    Parameters
    ----------
    optimizer_type : str
        The type of optimizer.
    v_mats : List[torch.Tensor]
        Initial matrices.
    learning_rate : float, default=0.01
        Global learning rate.
    decay_rate : float, default=1e-3
        Weight decay (L2 regularization) factor.

    Raises
    ------
    TypeError
        If inputs are of invalid types.
    """
    def step(self, i: int, v_current: torch.Tensor, descent_gradient: torch.Tensor, 
             full_energy_function: Optional[Callable] = None) -> torch.Tensor:
        lr = self.params['learning_rate']
        decay = self.params.get('decay_rate', 1e-3)
        v_norm = torch.norm(v_current)
        g_norm = torch.norm(descent_gradient)
        trust_ratio = v_norm / (g_norm + decay * v_norm + 1e-10)
        return v_current + lr * trust_ratio * (descent_gradient - decay * v_current)

class TorchNativeOptimizer(SimlrOptimizer):
    """
    Wrapper for using standard PyTorch optimizers within the SiMLR framework.

    Allows the use of first-order `torch.optim` algorithms (AdamW, Adagrad,
    NAdam) to update the basis matrices.

    Parameters
    ----------
    optimizer_type : str
        The type of PyTorch optimizer to use ('torch_adamw', 'torch_adagrad',
        'torch_nadam').
    v_mats : List[torch.Tensor]
        The initial basis matrices.
    **params : Dict[str, Any]
        Hyperparameters passed to the PyTorch optimizer.

    Raises
    ------
    TypeError
        If inputs are of invalid types.
    """
    def __init__(self, optimizer_type: str, v_mats: List[torch.Tensor], **params):
        super().__init__(optimizer_type, v_mats, **params)
        self.v_params = [nn.Parameter(v.clone()) for v in v_mats]
        lr = self.params['learning_rate']
        if optimizer_type == "torch_adamw":
            self.optimizers = [optim.AdamW([p], lr=lr) for p in self.v_params]
        elif optimizer_type == "torch_adagrad":
            self.optimizers = [optim.Adagrad([p], lr=lr) for p in self.v_params]
        elif optimizer_type == "torch_nadam":
            self.optimizers = [optim.NAdam([p], lr=lr) for p in self.v_params]
        else:
            self.optimizers = [optim.Adam([p], lr=lr) for p in self.v_params]

    def step(self, i: int, v_current: torch.Tensor, descent_gradient: torch.Tensor, 
             full_energy_function: Optional[Callable] = None) -> torch.Tensor:
        v_param = self.v_params[i]
        optimizer = self.optimizers[i]
        with torch.no_grad():
            v_param.copy_(v_current)
        v_param.grad = (-descent_gradient).contiguous()
        optimizer.step()
        return v_param.detach().clone()

class NSALBFGSB(SimlrOptimizer):
    r"""Bound-constrained L-BFGS-B, delegated to NSA-Flow's pure-torch solver.

    Why this rather than ``torch_lbfgs``
    ------------------------------------
    ``torch.optim.LBFGS`` is unconstrained. SiMLR's feasible set under a
    non-negative ``positivity`` is ``V >= 0``, and the existing path handles
    that by taking an unconstrained step and re-projecting afterwards. That is
    projected gradient wearing a quasi-Newton hat: the curvature information is
    built from iterates that keep being clipped, and the active set can only
    change by whatever the projection happens to do after the fact.

    L-BFGS-B builds the bound into the step. Its generalized Cauchy point
    minimises the quadratic model along the piecewise-linear projected steepest
    descent path, so it can activate or release many bounds in a single
    iteration, then minimises over the free variables with the limited-memory
    Hessian. On these objectives about half the coordinates sit at zero, which
    is exactly the regime where identifying the active set in one shot matters.

    NSA-Flow deprecated its own ``torch_lbfgs`` path on measured evidence when
    it introduced this solver; pysimlr's copy of that path additionally warns
    that its line search reuses a stale gradient unless a ``gradient_function``
    is supplied. Both point the same way.

    Notes
    -----
    Requires the NSA-Flow backend. `create_optimizer` raises a clear error when
    it is missing rather than silently substituting a different algorithm.

    SiMLR passes a *descent direction* (``descent_gradient``), and the sign
    convention here is that the step ascends it; the objective handed to
    L-BFGS-B is therefore ``full_energy_function`` with gradient
    ``-gradient_function``.
    """

    def __init__(self, optimizer_type: str, v_mats: List[torch.Tensor], **params):
        super().__init__(optimizer_type, v_mats, **params)
        from .nsa_backend import load_lbfgsb, load_gradient_mapping
        self._minimize = load_lbfgsb()
        self._grad_mapping = load_gradient_mapping()
        if self._minimize is None:
            raise ImportError(
                "optimizer_type='nsa_lbfgsb' needs the NSA-Flow backend "
                "(nsa_flow.lbfgsb.lbfgsb_minimize), which is not importable."
            )
        self.gradient_function: Optional[Callable] = None
        #: Smooth part of the energy, set by `simlr` with `gradient_function`;
        #: L-BFGS-B minimises this pair (a smooth function and its exact
        #: gradient), not the projected objective passed to `step`.
        self.smooth_energy_function: Optional[Callable] = None
        # Budget per outer sweep, in gradient evaluations. SiMLR calls `step`
        # once per modality per sweep, so this is an inner budget and is kept
        # small; the outer loop supplies the rest of the iteration.
        self.max_grad = int(self.params.get("max_iter", 20) or 20)
        if "nonneg" in params and params["nonneg"] is not None:
            self.nonneg = bool(params["nonneg"])
        else:
            positivity = params.get("positivity", None)
            if positivity == "either":
                self.nonneg = False
            elif positivity in {"positive", "hard", "nonnegative"}:
                self.nonneg = True
            else:
                self.nonneg = False

    def step(self, i: int, v_current: torch.Tensor, descent_gradient: torch.Tensor,
             full_energy_function: Optional[Callable] = None) -> torch.Tensor:
        grad_fn = self.gradient_function
        if self.smooth_energy_function is not None:
            full_energy_function = self.smooth_energy_function
        if full_energy_function is None or grad_fn is None:
            # Without both an energy and a gradient callable there is no
            # objective to minimise; fall back to the plain projected step
            # rather than pretending a quasi-Newton update happened.
            v_next = v_current + descent_gradient
            return torch.clamp(v_next, min=0.0) if self.nonneg else v_next

        # A line search probes points the caller never asked for, and under a
        # V >= 0 bound a long step lands exactly on the zero corner: the whole
        # basis becomes 0, which is feasible but rank-deficient, so the
        # projection inside the energy refuses it and the exception propagated
        # out of the solve. Captured at the point of failure the trial point
        # was `colnorms=[0.00e+00, 0.00e+00]`.
        #
        # An invalid trial point is scored +inf, which is the standard way to
        # tell a line search to back off. Reviving it instead would hand
        # L-BFGS-B a different point than the one it asked to evaluate, and its
        # curvature estimate is built from exactly those (point, value) pairs.
        INVALID = float("inf")

        def _degenerate(v):
            return not bool((v.detach().abs() > 1e-12).any(dim=0).all())

        def fun_grad(v):
            if _degenerate(v):
                return INVALID, torch.zeros_like(v)
            try:
                return float(full_energy_function(v)), (-grad_fn(v)).contiguous()
            except (RuntimeError, ValueError):
                return INVALID, torch.zeros_like(v)

        def fun(v):
            if _degenerate(v):
                return INVALID
            try:
                return float(full_energy_function(v))
            except (RuntimeError, ValueError):
                return INVALID

        certificate = None
        if self._grad_mapping is not None:
            proj = (lambda z: torch.clamp(z, min=0.0)) if self.nonneg else None
            certificate = lambda x, g: self._grad_mapping(x, g, proj)

        try:
            # float64: the curvature pairs (s, y) are differences of nearby
            # iterates and gradients, which float32 reduces to rounding noise.
            res = self._minimize(
                v_current.detach().double(),
                fun_grad,
                fun=fun,
                lower=0.0 if self.nonneg else None,
                max_grad=self.max_grad,
                tol=float(self.params["lbfgsb_tol"]),
                certificate=certificate,
            )
        except (ZeroDivisionError, FloatingPointError, RuntimeError) as exc:
            # The backend's Cauchy-point search divides by the curvature of the
            # quadratic model along the projected path, which is zero when that
            # path is flat -- reachable on these objectives, where the energy
            # moves by ~1e-5 over a whole solve. Observed as
            # ZeroDivisionError at nsa_flow/lbfgsb.py:248 on the 3-view case.
            # An upstream numerical edge case must not take down a benchmark
            # run, so this modality keeps its iterate and the sweep continues.
            warnings.warn(
                f"nsa_lbfgsb inner solve failed for modality {i} "
                f"({type(exc).__name__}: {exc}); keeping the current iterate "
                f"for this sweep.",
                RuntimeWarning, stacklevel=2,
            )
            self.last_result = None
            self.last_rejected = True
            return v_current.detach().clone()
        self.last_result = res
        v_next = res["x"].detach().clone().reshape(v_current.shape).to(v_current.dtype)

        # Reject a step that kills a component.
        #
        # The inner solve minimises this modality's energy with the consensus
        # `u` held fixed, so running it to high accuracy optimises against a
        # stale target -- the standard alternating-minimisation hazard. On the
        # 3-view case that shows up as a rank-deficient basis: at an inner
        # budget of 20 gradients one view's column went to zero and test
        # R-squared fell from 0.94 to 0.17, while budgets of 10 and 50 were
        # fine. The dependence on the budget is not monotone, so it is a
        # degenerate attractor rather than an under-solved step, and no choice
        # of `max_grad` avoids it in general.
        #
        # A basis with a dead column is not a feasible iterate for a rank-k
        # problem, so it is refused outright and the modality keeps the
        # iterate it came in with.
        live_before = (v_current.detach().abs() > 1e-12).any(dim=0)
        live_after = (v_next.abs() > 1e-12).any(dim=0)
        if bool((live_before & ~live_after).any()):
            self.last_rejected = True
            return v_current.detach().clone()
        self.last_rejected = False
        return v_next


#: Optimizer names that no longer exist, with the reason and the replacement.
REMOVED_OPTIMIZERS = {
    "torch_lbfgs": ("it rarely left its start and was once certified stationary there; "
                    "use 'nsa_lbfgsb'."),
    "nsa_flow": ("it was gradient descent with a fixed step followed by a second "
                 "NSA-Flow projection (the constraint is already applied after every "
                 "step) and fell back to an SVD polar factor without warning; use 'gd' "
                 "or any other optimizer."),
}


def create_optimizer(optimizer_type: str, v_mats: List[torch.Tensor], **params) -> SimlrOptimizer:
    """
    Factory function to instantiate a SiMLR optimizer by name.

    Parameters
    ----------
    optimizer_type : str
        The name of the optimizer to create. Supported values include:
        - 'hybrid_adam': `HybridAdam` (`simlr()`'s default, `defaults.DEFAULT_LINEAR_OPTIMIZER`)
        - 'adam': `Adam`
        - 'nadam': `Nadam`
        - 'rmsprop': `RMSProp`
        - 'gd': `SGD`
        - 'armijo_gradient': `ArmijoGradient` (`simlr()`'s default before 0.4.0)
        - 'bidirectional_armijo_gradient': `BidirectionalArmijoGradient`
        - 'lookahead': `Lookahead`
        - 'bidirectional_lookahead': `BidirectionalLookahead`
        - 'torch_adamw', 'torch_adagrad', 'torch_nadam': `TorchNativeOptimizer`
        - 'nsa_lbfgsb': `NSALBFGSB`, bound-constrained L-BFGS-B (needs NSA-Flow)
        - 'lars': `LARS`
    v_mats : List[torch.Tensor]
        Initial basis matrices for each modality.
    **params : Dict[str, Any]
        Hyperparameters passed to the optimizer constructor.

    Returns
    -------
    SimlrOptimizer
        An instance of the requested optimizer class.

    Raises
    ------
    ValueError
        If `optimizer_type` is not one of the supported names.
    TypeError
        If inputs are of invalid types.
    """
    mapping = {
        "hybrid_adam": HybridAdam,
        "adam": Adam,
        "nadam": Nadam,
        "rmsprop": RMSProp,
        "gd": SGD,
        "armijo_gradient": ArmijoGradient,
        "bidirectional_armijo_gradient": BidirectionalArmijoGradient,
        "lookahead": Lookahead,
        "bidirectional_lookahead": BidirectionalLookahead,
        "torch_adamw": TorchNativeOptimizer,
        "torch_adagrad": TorchNativeOptimizer,
        "torch_nadam": TorchNativeOptimizer,
        "nsa_lbfgsb": NSALBFGSB,
        "lars": LARS
    }
    if optimizer_type in REMOVED_OPTIMIZERS:
        raise ValueError(f"optimizer_type={optimizer_type!r} was removed in 0.4.0: "
                         f"{REMOVED_OPTIMIZERS[optimizer_type]}")
    if optimizer_type not in mapping:
        raise ValueError(
            f"Unknown optimizer_type {optimizer_type!r}. Choose one of: "
            f"{sorted(mapping)}. (This used to fall back to 'hybrid_adam' "
            f"silently, so a typo changed the algorithm without any notice.)"
        )
    opt_class = mapping[optimizer_type]
    return opt_class(optimizer_type, v_mats, **params)


#: Probe grid for `tune_learning_rate`, geometric and spanning the range over
#: which the SiMLR optimizers behave differently at all. It starts at 1.0
#: because that is the largest step with a defensible meaning: LARS moves ``V``
#: by exactly ``lr`` of its norm per sweep, so ``lr=1`` replaces the basis
#: every sweep and nothing above it is a search, it is a restart.
LR_PROBE_GRID = (1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001)


def tune_learning_rate(fit: Callable[..., Dict[str, Any]],
                       candidates: Sequence[float] = LR_PROBE_GRID,
                       probe_iterations: int = 3,
                       energy_key: Union[str, Sequence[str]] = ("best_energy", "energy", "loss_history"),
                       verbose: bool = False) -> Tuple[float, Dict[float, float]]:
    """Choose a learning rate by running short probes of the *real* fit.

    Parameters
    ----------
    fit : callable
        ``fit(learning_rate=lr, iterations=n)`` returning a result dict. Any
        SiMLR entry point works, which is what makes this method-agnostic.
    candidates : sequence of float
        Learning rates to probe, largest first.
    probe_iterations : int
        Sweeps per probe. Three is enough to separate a step that helps from
        one that overshoots, and cheap: the whole search costs about as much
        as a single ``len(candidates) * probe_iterations`` sweep fit.
    energy_key : str or sequence of str
        Which reported energy to compare; the first key present in the result
        is used. The default covers the linear path (``best_energy``) and the
        deep models (``loss_history``).

        A key holding a *history* is reduced to its **last** entry, not its
        minimum. `simlr` returns its best iterate, so ``best_energy`` already
        describes the basis the caller receives; the deep models return their
        final weights, so for them the last loss is the one that corresponds
        to the returned model. Scoring a history by its minimum would credit a
        run for an iterate it then threw away.

    Using it on a deep model
    ------------------------
    The deep entry points share the ``iterations`` budget name (counted in
    epochs) and are reached with a one-line adapter::

        tune_learning_rate(
            lambda learning_rate, iterations: lend_simr(
                mats, k=3, iterations=iterations, learning_rate=learning_rate))

    Returns
    -------
    (best_lr, {lr: energy}) -- the scores are returned so a caller can see
    whether the winner won clearly or by a rounding error.

    Why the probe runs the real model
    ---------------------------------
    The obvious implementation -- pick ``lr`` by evaluating
    :math:`f(v - lr\\,g)` -- describes plain gradient descent and is wrong for
    every optimizer here. LARS steps
    :math:`lr\\,(\\lVert v\\rVert/\\lVert g\\rVert)\\,\\hat g`, so its
    displacement is ``lr * ||v||`` *independently of the gradient*; Adam's is
    ~``lr`` per coordinate; only plain descent scales with ``||g||``. A single
    analytic probe therefore cannot serve them, and a tuner built on one
    returns a confident number that means nothing for the optimizer that
    consumes it.

    Running the model also keeps the projection in the loop. SiMLR's step is
    proximal: ``prox`` is applied after every update and can undo most of it,
    so the decrease that matters is in the *composite* objective, not in the
    smooth part the gradient describes.

    What it cannot do
    -----------------
    It compares energies, so it selects for optimisation quality, and on this
    model that is not the same as selecting for a good answer: support
    recovery has been measured to fall as the energy falls. It is a fix for
    an optimizer that does not move, not for an objective that points the
    wrong way. Candidates producing a non-finite or rank-deficient basis are
    rejected rather than scored.
    """
    scores: Dict[float, float] = {}
    first_error: Optional[BaseException] = None
    for lr in candidates:
        try:
            res = fit(learning_rate=float(lr), iterations=int(probe_iterations))
        except (ValueError, TypeError, KeyError):
            # A configuration error -- an unknown energy, an unknown optimizer,
            # a malformed argument -- is not a property of the step size and
            # must surface as itself. Swallowing it here turned every such
            # mistake into "no learning rate produced a usable fit", which
            # names the wrong thing and sends the reader to the tuner.
            raise
        except (RuntimeError, ArithmeticError) as exc:  # a step size that breaks the solve
            if first_error is None:
                first_error = exc
            continue                           # is not a candidate
        keys = (energy_key,) if isinstance(energy_key, str) else tuple(energy_key)
        e = next((res[kk] for kk in keys if res.get(kk) is not None), None)
        if isinstance(e, (list, tuple, np.ndarray)):
            e = e[-1] if len(e) else None          # last, not min: see `energy_key`
        vs = res.get("v", None)
        if e is None or vs is None or not np.isfinite(float(e)):
            continue
        if not all(torch.isfinite(v).all() and float(v.abs().sum()) > 0 for v in vs):
            continue
        scores[float(lr)] = float(e)
    if not scores:
        if first_error is not None:
            # Every candidate raised, so the failure is not a property of the
            # step size -- it is the fit itself. Re-raise the real exception
            # rather than a generic one about learning rates, which names the
            # wrong component and sends the reader to the tuner. A broken
            # backend surfaced as "no learning rate produced a usable fit".
            raise first_error
        raise RuntimeError(
            f"no learning rate in {list(candidates)} produced a usable fit; "
            f"every probe returned a non-finite energy or a degenerate basis."
        )
    best = min(scores, key=scores.get)
    if verbose:
        print(f"tune_learning_rate -> {best} from "
              + ", ".join(f"{k:g}:{v:.6g}" for k, v in sorted(scores.items())))
    return best, scores
