import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Any, Optional, Union, Callable, Tuple
from abc import ABC, abstractmethod

class SimlrOptimizer(ABC):
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
            'nsa_w': 0.5
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
    def step(self, i: int, v_current: torch.Tensor, descent_gradient: torch.Tensor, 
             full_energy_function: Optional[Callable] = None) -> torch.Tensor:
        state = self.state[i]
        beta = self.params.get('beta', 0.9)
        epsilon = self.params['epsilon']
        lr = self.params['learning_rate']
        state['v'] = beta * state['v'] + (1 - beta) * (descent_gradient**2)
        return v_current + lr * (descent_gradient / (torch.sqrt(state['v']) + epsilon))

class SGD(SimlrOptimizer):
    def step(self, i: int, v_current: torch.Tensor, descent_gradient: torch.Tensor, 
             full_energy_function: Optional[Callable] = None) -> torch.Tensor:
        lr = self.params['learning_rate']
        return v_current + lr * descent_gradient

class LARS(SimlrOptimizer):
    def step(self, i: int, v_current: torch.Tensor, descent_gradient: torch.Tensor, 
             full_energy_function: Optional[Callable] = None) -> torch.Tensor:
        lr = self.params['learning_rate']
        decay = self.params.get('decay_rate', 1e-3)
        v_norm = torch.norm(v_current)
        g_norm = torch.norm(descent_gradient)
        trust_ratio = v_norm / (g_norm + decay * v_norm + 1e-10)
        return v_current + lr * trust_ratio * (descent_gradient - decay * v_current)

class NSAFlowOptimizer(SimlrOptimizer):
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
