import torch
from typing import List, Dict, Any, Optional, Callable, Tuple

class SimlrOptimizer:
    def __init__(self, optimizer_type: str, v_mats: List[torch.Tensor], **params):
        self.optimizer_type = optimizer_type
        self.params = params
        self.state = []
        for v in v_mats:
            self.state.append({
                'm': torch.zeros_like(v),
                'v': torch.zeros_like(v),
                'v_max': torch.zeros_like(v),
                'g_sum_sq': torch.zeros_like(v),
                'momentum': torch.zeros_like(v),
                'last_step_size': torch.tensor(1.0),
                'iter': 0
            })

    def step(self, i: int, v_current: torch.Tensor, descent_gradient: torch.Tensor, 
             full_energy_function: Optional[Callable] = None) -> torch.Tensor:
        raise NotImplementedError

def backtracking_linesearch(v_current: torch.Tensor, 
                            descent_direction: torch.Tensor, 
                            ascent_gradient: torch.Tensor,
                            energy_function: Callable, 
                            initial_step_size: float = 1.0,
                            alpha: float = 0.3, 
                            beta: float = 0.8, 
                            max_iter: int = 10) -> float:
    step_size = initial_step_size
    initial_energy = energy_function(v_current)
    
    # slope_term = sum(ascent_gradient * descent_direction)
    slope_term = torch.sum(ascent_gradient * descent_direction)
    
    if slope_term >= 0:
        # Not a descent direction
        return 0.0
        
    for _ in range(max_iter):
        v_candidate = v_current + step_size * descent_direction
        try:
            new_energy = energy_function(v_candidate)
        except:
            new_energy = float('inf')
            
        if new_energy <= initial_energy + alpha * step_size * slope_term:
            return step_size
        step_size *= beta
        
    return 0.0

class HybridAdam(SimlrOptimizer):
    def step(self, i: int, v_current: torch.Tensor, descent_gradient: torch.Tensor, 
             full_energy_function: Optional[Callable] = None) -> torch.Tensor:
        state = self.state[i]
        beta1 = self.params.get('beta1', 0.9)
        beta2 = self.params.get('beta2', 0.999)
        epsilon = self.params.get('epsilon', 1e-8)
        
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
            optimal_step_size = self.params.get('learning_rate', 0.001)
            
        state['last_step_size'] = torch.tensor(optimal_step_size * 1.5 if optimal_step_size > 1e-9 else 1.0)
        updated_v = v_current + optimal_step_size * search_direction
        return updated_v

class Adam(SimlrOptimizer):
    def step(self, i: int, v_current: torch.Tensor, descent_gradient: torch.Tensor, 
             full_energy_function: Optional[Callable] = None) -> torch.Tensor:
        state = self.state[i]
        state['iter'] += 1
        beta1 = self.params.get('beta1', 0.9)
        beta2 = self.params.get('beta2', 0.999)
        epsilon = self.params.get('epsilon', 1e-8)
        lr = self.params.get('learning_rate', 0.001)
        
        state['m'] = beta1 * state['m'] + (1 - beta1) * descent_gradient
        state['v'] = beta2 * state['v'] + (1 - beta2) * (descent_gradient**2)
        
        m_hat = state['m'] / (1 - beta1**state['iter'])
        v_hat = state['v'] / (1 - beta2**state['iter'])
        
        update_direction = m_hat / (torch.sqrt(v_hat) + epsilon)
        return v_current + lr * update_direction

class RMSProp(SimlrOptimizer):
    def step(self, i: int, v_current: torch.Tensor, descent_gradient: torch.Tensor, 
             full_energy_function: Optional[Callable] = None) -> torch.Tensor:
        state = self.state[i]
        beta = self.params.get('beta', 0.9)
        epsilon = self.params.get('epsilon', 1e-8)
        lr = self.params.get('learning_rate', 0.001)
        
        state['v'] = beta * state['v'] + (1 - beta) * (descent_gradient**2)
        update_direction = descent_gradient / (torch.sqrt(state['v']) + epsilon)
        return v_current + lr * update_direction

class SGD(SimlrOptimizer):
    def step(self, i: int, v_current: torch.Tensor, descent_gradient: torch.Tensor, 
             full_energy_function: Optional[Callable] = None) -> torch.Tensor:
        state = self.state[i]
        state['iter'] += 1
        lr = self.params.get('learning_rate', 1e-6)
        decay = self.params.get('decay_rate', 1e-3)
        
        effective_lr = lr * torch.exp(torch.tensor(-decay * state['iter']))
        return v_current + effective_lr * descent_gradient

def create_optimizer(optimizer_type: str, v_mats: List[torch.Tensor], **params) -> SimlrOptimizer:
    mapping = {
        "hybrid_adam": HybridAdam,
        "adam": Adam,
        "rmsprop": RMSProp,
        "gd": SGD,
    }
    opt_class = mapping.get(optimizer_type, HybridAdam)
    return opt_class(optimizer_type, v_mats, **params)
