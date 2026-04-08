import torch
from typing import List, Dict, Any, Optional, Callable, Tuple, Type
try:
    import nsa_flow as nsa
except ImportError:
    nsa = None

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
                'iter': 0,
                'v_slow': None,
                'step': 0
            })

    def step(self, i: int, v_current: torch.Tensor, descent_gradient: torch.Tensor, 
             full_energy_function: Optional[Callable] = None) -> torch.Tensor:
        raise NotImplementedError

def backtracking_linesearch(v_current: torch.Tensor, 
                            descent_direction: torch.Tensor, 
                            ascent_gradient: torch.Tensor,
                            energy_function: Callable, 
                            initial_step_size: float = 1.0,
                            alpha: float = 1e-4, 
                            beta: float = 0.5, 
                            max_iter: int = 10,
                            min_step: float = 1e-12) -> float:
    step_size = initial_step_size
    try:
        initial_energy = energy_function(v_current)
    except:
        initial_energy = float('inf')
        
    slope_term = torch.sum(ascent_gradient * descent_direction)
    if slope_term >= 0: 
        return 0.0
        
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
        return v_current + optimal_step_size * search_direction

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
        return v_current + lr * (m_hat / (torch.sqrt(v_hat) + epsilon))

class Nadam(SimlrOptimizer):
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
        nesterov_m_hat = beta1 * m_hat + ((1 - beta1) * descent_gradient) / (1 - beta1**state['iter'])
        return v_current + lr * (nesterov_m_hat / (torch.sqrt(v_hat) + epsilon))

class ArmijoGradient(SimlrOptimizer):
    def step(self, i: int, v_current: torch.Tensor, descent_gradient: torch.Tensor, 
             full_energy_function: Optional[Callable] = None) -> torch.Tensor:
        state = self.state[i]
        epsilon = self.params.get('epsilon', 1e-12)
        state['momentum'] = 0.9 * state['momentum'] + 0.1 * descent_gradient
        search_direction = state['momentum']
        dir_norm = torch.norm(search_direction)
        if dir_norm < epsilon:
            return v_current
        search_direction = search_direction / dir_norm
        
        if full_energy_function is not None:
            optimal_step_size = backtracking_linesearch(
                v_current=v_current,
                descent_direction=search_direction,
                ascent_gradient=-descent_gradient,
                energy_function=full_energy_function,
                initial_step_size=state['last_step_size'].item()
            )
        else:
            optimal_step_size = self.params.get('learning_rate', 0.01)
            
        state['last_step_size'] = torch.tensor(optimal_step_size * 1.5 if optimal_step_size > 1e-10 else 1.0)
        return v_current + optimal_step_size * search_direction

class BidirectionalArmijoGradient(SimlrOptimizer):
    def step(self, i: int, v_current: torch.Tensor, descent_gradient: torch.Tensor, 
             full_energy_function: Optional[Callable] = None) -> torch.Tensor:
        state = self.state[i]
        epsilon = self.params.get('epsilon', 1e-12)
        dir_norm = torch.norm(descent_gradient)
        if dir_norm < epsilon:
            return v_current
        norm_direction = descent_gradient / dir_norm
        
        if full_energy_function is not None:
            optimal_step_size, selected_direction = bidirectional_linesearch(
                v_current=v_current,
                descent_direction=norm_direction,
                ascent_gradient=-descent_gradient,
                energy_function=full_energy_function,
                initial_step_size=state['last_step_size'].item()
            )
        else:
            optimal_step_size = self.params.get('learning_rate', 0.01)
            selected_direction = norm_direction
            
        state['last_step_size'] = torch.tensor(optimal_step_size * 1.5 if optimal_step_size > 1e-9 else 1.0)
        return v_current + optimal_step_size * selected_direction

class Lookahead(SimlrOptimizer):
    def __init__(self, optimizer_type: str, v_mats: List[torch.Tensor], **params):
        super().__init__(optimizer_type, v_mats, **params)
        self.base_optimizer = Adam(optimizer_type, v_mats, **params)

    def step(self, i: int, v_current: torch.Tensor, descent_gradient: torch.Tensor, 
             full_energy_function: Optional[Callable] = None) -> torch.Tensor:
        state = self.state[i]
        if state['v_slow'] is None:
            state['v_slow'] = v_current.clone()
        
        v_fast = self.base_optimizer.step(i, v_current, descent_gradient, full_energy_function)
        
        state['step'] += 1
        k = self.params.get('k', 5)
        alpha = self.params.get('alpha', 0.5)
        
        if state['step'] % k == 0:
            state['v_slow'] = state['v_slow'] + alpha * (v_fast - state['v_slow'])
            v_fast = state['v_slow']
            
        return v_fast

class BidirectionalLookahead(SimlrOptimizer):
    def step(self, i: int, v_current: torch.Tensor, descent_gradient: torch.Tensor, 
             full_energy_function: Optional[Callable] = None) -> torch.Tensor:
        state = self.state[i]
        if state['v_slow'] is None:
            state['v_slow'] = v_current.clone()
            
        beta1 = self.params.get('beta1', 0.9)
        beta2 = self.params.get('beta2', 0.999)
        epsilon = self.params.get('epsilon', 1e-8)
        state['iter'] += 1
        
        state['m'] = beta1 * state['m'] + (1 - beta1) * descent_gradient
        state['v'] = beta2 * state['v'] + (1 - beta2) * (descent_gradient**2)
        state['v_max'] = torch.maximum(state['v_max'], state['v'])
        
        m_hat = state['m'] / (1 - beta1**state['iter'])
        v_hat = state['v_max'] / (1 - beta2**state['iter'])
        search_direction = m_hat / (torch.sqrt(v_hat) + epsilon)
        
        if full_energy_function is not None:
            optimal_step_size, selected_direction = bidirectional_linesearch(
                v_current=v_current,
                descent_direction=search_direction,
                ascent_gradient=-descent_gradient,
                energy_function=full_energy_function,
                initial_step_size=state['last_step_size'].item()
            )
        else:
            optimal_step_size = self.params.get('learning_rate', 0.001)
            selected_direction = search_direction
            
        v_fast = v_current + optimal_step_size * selected_direction
        
        state['step'] += 1
        k = self.params.get('k', 5)
        alpha = self.params.get('alpha', 0.5)
        if state['step'] % k == 0:
            state['v_slow'] = state['v_slow'] + alpha * (v_fast - state['v_slow'])
            v_fast = state['v_slow']
            
        state['last_step_size'] = torch.tensor(optimal_step_size * 1.5 if optimal_step_size > 1e-9 else 1.0)
        return v_fast

class RMSProp(SimlrOptimizer):
    def step(self, i: int, v_current: torch.Tensor, descent_gradient: torch.Tensor, 
             full_energy_function: Optional[Callable] = None) -> torch.Tensor:
        state = self.state[i]
        beta = self.params.get('beta', 0.9)
        epsilon = self.params.get('epsilon', 1e-8)
        lr = self.params.get('learning_rate', 0.001)
        state['v'] = beta * state['v'] + (1 - beta) * (descent_gradient**2)
        return v_current + lr * (descent_gradient / (torch.sqrt(state['v']) + epsilon))

class SGD(SimlrOptimizer):
    def step(self, i: int, v_current: torch.Tensor, descent_gradient: torch.Tensor, 
             full_energy_function: Optional[Callable] = None) -> torch.Tensor:
        state = self.state[i]
        state['iter'] += 1
        lr = self.params.get('learning_rate', 1e-6)
        decay = self.params.get('decay_rate', 1e-3)
        effective_lr = lr * torch.exp(torch.tensor(-decay * state['iter']))
        return v_current + effective_lr * descent_gradient

class NSAFlowOptimizer(SimlrOptimizer):
    """
    Bridge to nsa-flow package for Riemannian optimization.
    """
    def __init__(self, optimizer_type: str, v_mats: List[torch.Tensor], **params):
        super().__init__(optimizer_type, v_mats, **params)
        if nsa is None:
            raise ImportError("nsa-flow package not installed.")
        self.lr = params.get('learning_rate', 0.01)
        self.w = params.get('nsa_w', 0.5) # Weight for orthogonality

    def step(self, i: int, v_current: torch.Tensor, descent_gradient: torch.Tensor, 
             full_energy_function: Optional[Callable] = None) -> torch.Tensor:
        v_cand = v_current + self.lr * descent_gradient
        try:
            res_nsa = nsa.nsa_flow_orth(
                v_cand,
                w=self.w,
                retraction="soft_polar",
                optimizer="asgd",
                max_iter=500,
                tol=1e-5,
                verbose=False
            )
            v_next = res_nsa['Y']
            if torch.isnan(v_next).any():
                u_svd, _, v_svd = torch.linalg.svd(v_cand, full_matrices=False)
                v_next = u_svd @ v_svd
        except:
            u_svd, _, v_svd = torch.linalg.svd(v_cand, full_matrices=False)
            v_next = u_svd @ v_svd
            
        return v_next

class TorchNativeOptimizer(SimlrOptimizer):
    def __init__(self, optimizer_type: str, v_mats: List[torch.Tensor], **params):
        super().__init__(optimizer_type, v_mats, **params)
        self.optimizers = []
        self.v_params = []
        opt_map = {
            "torch_adamw": torch.optim.AdamW,
            "torch_adagrad": torch.optim.Adagrad,
            "torch_nadam": torch.optim.NAdam,
            "torch_lbfgs": torch.optim.LBFGS
        }
        opt_class = opt_map.get(optimizer_type, torch.optim.AdamW)
        lr = params.get('learning_rate', 0.01)
        for v in v_mats:
            v_param = v.detach().clone().requires_grad_(True)
            self.v_params.append(v_param)
            self.optimizers.append(opt_class([v_param], lr=lr))

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
        "torch_lbfgs": TorchNativeOptimizer
    }
    opt_class = mapping.get(optimizer_type, HybridAdam)
    return opt_class(optimizer_type, v_mats, **params)
