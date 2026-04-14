import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Any, Optional, Union, Callable, Tuple
from abc import ABC, abstractmethod

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
                'last_step_size': torch.tensor(0.01)
            })

    def filter_params(self, optimizer_type: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance parameter filtering to handle varying hyperparameters 
        (beta1, beta2, amsgrad) across optimizer types.
        """
        defaults = {
            'learning_rate': 0.001,
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-8,
            'weight_decay': 0.0,
            'amsgrad': False,
            'momentum': 0.9,
            'nsa_w': 0.1
        }
        
        # Merge defaults with provided params
        filtered = {k: params.get(k, v) for k, v in defaults.items()}
        
        # Optimizer-specific pruning if needed
        if optimizer_type == "gd":
            filtered = {k: v for k, v in filtered.items() if k in ['learning_rate', 'momentum', 'weight_decay']}
        elif optimizer_type in ["adam", "nadam", "hybrid_adam"]:
            filtered = {k: v for k, v in filtered.items() if k in ['learning_rate', 'beta1', 'beta2', 'epsilon', 'amsgrad']}
            
        return filtered

    @abstractmethod
    def step(self, i: int, v_current: torch.Tensor, descent_gradient: torch.Tensor, 
             full_energy_function: Optional[Callable] = None) -> torch.Tensor:
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
        Function that computes the energy\/loss for a given parameter set.
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
    """
    try:
        initial_energy = energy_function(v_current)
    except:
        return 0.0
        
    step_size = initial_step_size
    slope_term = torch.sum(ascent_gradient * descent_direction)
    
    for _ in range(max_iter):
        if step_size <= min_step:
            break
        v_candidate = v_current + step_size * descent_direction
        try:
            new_energy = energy_function(v_candidate)
        except:
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
        either `descent_direction` or `-descent_direction`.
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
    
    if pos_step >= neg_step and pos_step > 0:
        return pos_step, descent_direction
    elif neg_step > 0:
        return neg_step, -descent_direction
    else:
        return 0.0, descent_direction

class HybridAdam(SimlrOptimizer):
    """
    Hybrid optimizer combining Adam-style momentum with line search.

    Computes an Adam-like search direction (using max variance for stability) 
    and then performs a backtracking line search along that direction to 
    ensure a sufficient decrease in the energy function.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Initial step size.
    beta1 : float, default=0.9
        Momentum decay.
    beta2 : float, default=0.999
        Variance decay.
    epsilon : float, default=1e-8
        Numerical stability constant.
    """
    def step(self, i: int, v_current: torch.Tensor, descent_gradient: torch.Tensor, 
             full_energy_function: Optional[Callable] = None) -> torch.Tensor:
        state = self.state[i]
        beta1 = self.params['beta1']
        beta2 = self.params['beta2']
        epsilon = self.params['epsilon']
        state['m'] = beta1 * state['m'] + (1 - beta1) * descent_gradient
        state['v'] = beta2 * state['v'] + (1 - beta2) * (descent_gradient**2)
        state['v_max'] = torch.maximum(state['v_max'], state['v'])
        search_direction = state['m'] / (torch.sqrt(state['v_max']) + epsilon)
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
        state['last_step_size'] = torch.tensor(optimal_step_size * 1.5 if optimal_step_size > 1e-9 else 1.0)
        return v_current + optimal_step_size * search_direction

class Adam(SimlrOptimizer):
    """
    Adam (Adaptive Moment Estimation) optimizer for SiMLR.

    Updates basis matrices using estimates of the first and second 
    moments of the gradients.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size for updates.
    beta1 : float, default=0.9
        Exponential decay rate for the first moment estimates.
    beta2 : float, default=0.999
        Exponential decay rate for the second moment estimates.
    epsilon : float, default=1e-8
        Numerical stability constant.
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
        return v_current + lr * (m_hat / (torch.sqrt(v_hat) + epsilon))

class Nadam(SimlrOptimizer):
    """
    Nadam (Nesterov-accelerated Adaptive Moment Estimation) optimizer for SiMLR.

    Combines Adam with Nesterov accelerated gradient (NAG) for potentially 
    faster convergence.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size.
    beta1 : float, default=0.9
        Momentum decay.
    beta2 : float, default=0.999
        Variance decay.
    epsilon : float, default=1e-8
        Numerical stability constant.
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
        nesterov_m_hat = beta1 * m_hat + ((1 - beta1) * descent_gradient) / (1 - beta1**state['iter'])
        return v_current + lr * (nesterov_m_hat / (torch.sqrt(v_hat) + epsilon))

class ArmijoGradient(SimlrOptimizer):
    """
    Gradient descent optimizer with Armijo-style backtracking line search.

    Uses a line search to find an optimal step size that satisfies the 
    sufficient decrease condition (Armijo rule). Falls back to a constant 
    learning rate if no energy function is provided.

    Parameters
    ----------
    learning_rate : float, default=0.1
        Initial step size for the line search.
    epsilon : float, default=1e-8
        Numerical stability constant.
    """
    def step(self, i: int, v_current: torch.Tensor, descent_gradient: torch.Tensor, 
             full_energy_function: Optional[Callable] = None) -> torch.Tensor:
        state = self.state[i]
        epsilon = self.params['epsilon']
        state['momentum'] = 0.9 * state['momentum'] + 0.1 * descent_gradient
        if full_energy_function is not None:
            optimal_step_size = backtracking_linesearch(
                v_current=v_current,
                descent_direction=-descent_gradient + 0.1 * state['momentum'],
                ascent_gradient=-descent_gradient,
                energy_function=full_energy_function,
                initial_step_size=self.params['learning_rate']
            )
        else:
            optimal_step_size = self.params['learning_rate']
        return v_current - optimal_step_size * descent_gradient + 0.1 * state['momentum']

class BidirectionalArmijoGradient(SimlrOptimizer):
    """
    Gradient descent optimizer with bidirectional Armijo line search.

    Similar to `ArmijoGradient`, but the line search explores both the 
    descent and ascent directions to find the optimal step. This is 
    useful for complex energy surfaces where the sign of the gradient 
    might not immediately point towards the minimum.

    Parameters
    ----------
    learning_rate : float, default=0.1
        Initial step size for the line search.
    epsilon : float, default=1e-8
        Numerical stability constant.
    """
    def step(self, i: int, v_current: torch.Tensor, descent_gradient: torch.Tensor, 
             full_energy_function: Optional[Callable] = None) -> torch.Tensor:
        state = self.state[i]
        epsilon = self.params['epsilon']
        state['momentum'] = 0.9 * state['momentum'] + 0.1 * descent_gradient
        if full_energy_function is not None:
            optimal_step_size, dir = bidirectional_linesearch(
                v_current=v_current,
                descent_direction=-descent_gradient + 0.1 * state['momentum'],
                ascent_gradient=-descent_gradient,
                energy_function=full_energy_function,
                initial_step_size=self.params['learning_rate']
            )
            return v_current + optimal_step_size * dir
        else:
            return v_current - self.params['learning_rate'] * descent_gradient + 0.1 * state['momentum']

class Lookahead(SimlrOptimizer):
    """
    Lookahead optimizer using `HybridAdam` as the inner solver.

    Implements the "fast weights" and "slow weights" strategy to improve 
    convergence stability. Slow weights are updated every `k` steps 
    towards the fast weights.

    Parameters
    ----------
    k : int, default=5
        Frequency of slow weight updates.
    alpha : float, default=0.5
        Step size (interpolation factor) for slow weight updates.
    **params : Dict[str, Any]
        Additional parameters passed to the inner `HybridAdam` optimizer.
    """
    def __init__(self, optimizer_type: str, v_mats: List[torch.Tensor], **params):
        super().__init__(optimizer_type, v_mats, **params)
        self.k = self.params.get('k', 5)
        self.alpha = self.params.get('alpha', 0.5)
        self.slow_weights = [v.clone() for v in v_mats]
        self.inner_opt = HybridAdam(optimizer_type, v_mats, **params)

    def step(self, i: int, v_current: torch.Tensor, descent_gradient: torch.Tensor, 
             full_energy_function: Optional[Callable] = None) -> torch.Tensor:
        state = self.state[i]
        state['iter'] += 1
        v_next = self.inner_opt.step(i, v_current, descent_gradient, full_energy_function)
        if state['iter'] % self.k == 0:
            self.slow_weights[i] = self.slow_weights[i] + self.alpha * (v_next - self.slow_weights[i])
            return self.slow_weights[i].clone()
        return v_next

class BidirectionalLookahead(SimlrOptimizer):
    """
    Lookahead optimizer using `BidirectionalArmijoGradient` as the inner solver.

    Maintains a set of "slow weights" that are updated every `k` steps towards 
    the "fast weights" generated by the inner bidirectional Armijo optimizer. 
    This improves stability and convergence in complex SiMLR landscapes.

    Parameters
    ----------
    k : int, default=5
        Frequency of slow weight updates.
    alpha : float, default=0.5
        Step size (interpolation factor) for slow weight updates.
    **params : Dict[str, Any]
        Additional parameters passed to the inner optimizer.
    """
    def __init__(self, optimizer_type: str, v_mats: List[torch.Tensor], **params):
        super().__init__(optimizer_type, v_mats, **params)
        self.k = self.params.get('k', 5)
        self.alpha = self.params.get('alpha', 0.5)
        self.slow_weights = [v.clone() for v in v_mats]
        self.inner_opt = BidirectionalArmijoGradient(optimizer_type, v_mats, **params)

    def step(self, i: int, v_current: torch.Tensor, descent_gradient: torch.Tensor, 
             full_energy_function: Optional[Callable] = None) -> torch.Tensor:
        state = self.state[i]
        state['iter'] += 1
        v_next = self.inner_opt.step(i, v_current, descent_gradient, full_energy_function)
        if state['iter'] % self.k == 0:
            self.slow_weights[i] = self.slow_weights[i] + self.alpha * (v_next - self.slow_weights[i])
            return self.slow_weights[i].clone()
        return v_next

class RMSProp(SimlrOptimizer):
    """
    RMSProp (Root Mean Square Propagation) optimizer for SiMLR.

    Updates basis matrices using a moving average of squared gradients 
    to normalize the gradient magnitude.

    Parameters
    ----------
    learning_rate : float, default=0.01
        The step size for updates.
    beta : float, default=0.9
        Discounting factor for the history/coming gradient.
    epsilon : float, default=1e-8
        Numerical stability constant.
    """
    def step(self, i: int, v_current: torch.Tensor, descent_gradient: torch.Tensor, 
             full_energy_function: Optional[Callable] = None) -> torch.Tensor:
        state = self.state[i]
        beta = self.params.get('beta', 0.9)
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
    learning_rate : float, default=0.01
        The step size for updates.
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
    learning_rate : float, default=0.01
        Global learning rate.
    decay_rate : float, default=1e-3
        Weight decay (L2 regularization) factor.
    """
    def step(self, i: int, v_current: torch.Tensor, descent_gradient: torch.Tensor, 
             full_energy_function: Optional[Callable] = None) -> torch.Tensor:
        lr = self.params['learning_rate']
        decay = self.params.get('decay_rate', 1e-3)
        v_norm = torch.norm(v_current)
        g_norm = torch.norm(descent_gradient)
        trust_ratio = v_norm / (g_norm + decay * v_norm + 1e-10)
        return v_current + lr * trust_ratio * (descent_gradient - decay * v_current)

class NSAFlowOptimizer(SimlrOptimizer):
    """
    Optimizer using Non-Standard Analysis (NSA) Flow for Stiefel manifold updates.

    Performs a gradient step and then uses NSA Flow to retract the result 
    back onto the manifold of orthogonal matrices. Falls back to standard 
    SVD-based projection if the `nsa_flow` package is unavailable.

    Parameters
    ----------
    learning_rate : float, default=0.01
        The step size for gradient updates.
    nsa_w : float, default=0.1
        The retraction weight/step for NSA Flow.
    """
    def __init__(self, optimizer_type: str, v_mats: List[torch.Tensor], **params):
        super().__init__(optimizer_type, v_mats, **params)
        self.lr = self.params['learning_rate']
        self.w = self.params['nsa_w']
        try:
            from nsa_flow import nsa_flow_orth
            self.nsa_flow = nsa_flow_orth
        except ImportError:
            self.nsa_flow = None

    def step(self, i: int, v_current: torch.Tensor, descent_gradient: torch.Tensor, 
             full_energy_function: Optional[Callable] = None) -> torch.Tensor:
        v_next = v_current + self.lr * descent_gradient
        if self.nsa_flow:
            try:
                res = self.nsa_flow(v_next, w=self.w, max_iter=5)
                if res['Y'] is not None: return res['Y'].to(v_current.dtype)
            except: pass
        u, s, v_h = torch.linalg.svd(v_next, full_matrices=False)
        return u @ v_h

class TorchNativeOptimizer(SimlrOptimizer):
    """
    Wrapper for using standard PyTorch optimizers within the SiMLR framework.

    Allows the use of `torch.optim` algorithms (e.g., AdamW, Adagrad, LBFGS) 
    to update the basis matrices.

    Parameters
    ----------
    optimizer_type : str
        The type of PyTorch optimizer to use (e.g., 'torch_adamw', 
        'torch_lbfgs').
    v_mats : List[torch.Tensor]
        The initial basis matrices.
    **params : Dict[str, Any]
        Hyperparameters passed to the PyTorch optimizer.
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
        elif optimizer_type == "torch_lbfgs":
            self.optimizers = [optim.LBFGS([p], lr=lr) for p in self.v_params]
        else:
            self.optimizers = [optim.Adam([p], lr=lr) for p in self.v_params]

    def step(self, i: int, v_current: torch.Tensor, descent_gradient: torch.Tensor, 
             full_energy_function: Optional[Callable] = None) -> torch.Tensor:
        v_param = self.v_params[i]
        optimizer = self.optimizers[i]
        with torch.no_grad():
            v_param.copy_(v_current)
        v_param.grad = -descent_gradient
        if self.optimizer_type == "torch_lbfgs" and full_energy_function is not None:
            def closure():
                optimizer.zero_grad()
                v_param.grad = -descent_gradient
                return full_energy_function(v_param)
            optimizer.step(closure)
        else:
            optimizer.step()
        return v_param.detach().clone()

def create_optimizer(optimizer_type: str, v_mats: List[torch.Tensor], **params) -> SimlrOptimizer:
    """
    Factory function to instantiate a SiMLR optimizer by name.

    Parameters
    ----------
    optimizer_type : str
        The name of the optimizer to create. Supported values include:
        - 'hybrid_adam': `HybridAdam` (default)
        - 'adam': `Adam`
        - 'nadam': `Nadam`
        - 'rmsprop': `RMSProp`
        - 'gd': `SGD`
        - 'armijo_gradient': `ArmijoGradient`
        - 'bidirectional_armijo_gradient': `BidirectionalArmijoGradient`
        - 'lookahead': `Lookahead`
        - 'bidirectional_lookahead': `BidirectionalLookahead`
        - 'nsa_flow': `NSAFlowOptimizer`
        - 'torch_adamw', 'torch_adagrad', 'torch_nadam', 'torch_lbfgs': `TorchNativeOptimizer`
        - 'lars': `LARS`
    v_mats : List[torch.Tensor]
        Initial basis matrices for each modality.
    **params : Dict[str, Any]
        Hyperparameters passed to the optimizer constructor.

    Returns
    -------
    SimlrOptimizer
        An instance of the requested optimizer class.
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
        "nsa_flow": NSAFlowOptimizer,
        "torch_adamw": TorchNativeOptimizer,
        "torch_adagrad": TorchNativeOptimizer,
        "torch_nadam": TorchNativeOptimizer,
        "torch_lbfgs": TorchNativeOptimizer,
        "lars": LARS
    }
    opt_class = mapping.get(optimizer_type, HybridAdam)
    return opt_class(optimizer_type, v_mats, **params)
