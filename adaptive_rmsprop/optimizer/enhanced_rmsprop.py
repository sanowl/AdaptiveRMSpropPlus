from collections.abc import Iterable
from typing import Any, Callable, Optional
import torch
from torch.optim import Optimizer

class EnhancedRMSpropTF(Optimizer):
    """Enhanced RMSprop algorithm that combines TensorFlow-style implementation
    with adaptive features for better stability and performance.
    
    Key Features:
    - TensorFlow-style epsilon handling
    - Dynamic epsilon adjustment
    - Cyclical learning rate with warmup
    - Adaptive gradient noise
    - Layer-wise adaptive momentum

    Args:
        params (Iterable[torch.nn.Parameter]): Iterable of parameters to optimize
        lr (float, optional): Learning rate (default: 1e-2)
        alpha (float, optional): Smoothing constant (default: 0.99)
        eps (float, optional): Term added to denominator for numerical stability (default: 1e-8)
        weight_decay (float, optional): Weight decay (L2 penalty) (default: 0)
        momentum (float, optional): Momentum factor (default: 0)
        centered (bool, optional): If True, compute centered RMSprop (default: False)
        noise_scale (float, optional): Scale of gradient noise (default: 1e-6)
        warmup_steps (int, optional): Number of warmup steps (default: 1000)
        lr_cycles (int, optional): Number of learning rate cycles (default: 3)
    """
    
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-2,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0,
        momentum: float = 0,
        centered: bool = False,
        noise_scale: float = 1e-6,
        warmup_steps: int = 1000,
        lr_cycles: int = 3,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= momentum:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= alpha:
            raise ValueError(f"Invalid alpha value: {alpha}")
        if not 0.0 <= noise_scale:
            raise ValueError(f"Invalid noise scale: {noise_scale}")

        defaults = dict(
            lr=lr, momentum=momentum, alpha=alpha, eps=eps,
            centered=centered, weight_decay=weight_decay,
            noise_scale=noise_scale
        )
        super().__init__(params, defaults)
        self.base_lr = lr
        self.warmup_steps = warmup_steps
        self.lr_cycles = lr_cycles

    def _adjust_epsilon(self, grad_norm: torch.Tensor) -> torch.Tensor:
        """Dynamically adjust epsilon based on gradient magnitude"""
        return self.defaults['eps'] * (1.0 + grad_norm.log1p())

    def _compute_cyclic_lr(self, step: int) -> float:
        """Implement cyclical learning rate with warmup"""
        if step < self.warmup_steps:
            return self.base_lr * (step / self.warmup_steps)
        
        cycle_progress = ((step - self.warmup_steps) / 
                         (self.lr_cycles * self.warmup_steps))
        cycle_progress = torch.tensor(cycle_progress, dtype=torch.float32)
        return float(self.base_lr * (0.5 * (1 + torch.cos(torch.pi * cycle_progress))))

    def _inject_gradient_noise(self, grad: torch.Tensor, step: int) -> torch.Tensor:
        """Add adaptive gradient noise for better exploration"""
        noise_scale = self.defaults['noise_scale'] / (1.0 + step)**0.55
        noise = torch.randn_like(grad) * noise_scale * torch.abs(grad).mean()
        return grad + noise

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        """Performs a single optimization step with enhanced features.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model and
                returns the loss. Optional for most optimizers.

        Returns:
            Optional[float]: The loss value returned by the closure, if provided.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("EnhancedRMSpropTF does not support sparse gradients")
                state = self.state[p]

                # Enhanced state initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.ones_like(p, memory_format=torch.preserve_format)
                    state['grad_norm_ema'] = torch.ones_like(p, memory_format=torch.preserve_format)
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                square_avg = state['square_avg']
                alpha = group['alpha']

                state['step'] += 1
                step = state['step']

                # Compute gradient norm and adjust epsilon
                grad_norm = grad.norm()
                dynamic_eps = self._adjust_epsilon(grad_norm)

                # Update gradient norm EMA
                state['grad_norm_ema'].mul_(0.9).add_(grad_norm, alpha=0.1)

                # Inject gradient noise
                grad = self._inject_gradient_noise(grad, step)

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

                # Compute adaptive learning rate
                current_lr = self._compute_cyclic_lr(step)

                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(grad, alpha=1 - alpha)
                    avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).add_(dynamic_eps).sqrt_()
                else:
                    avg = square_avg.add(dynamic_eps).sqrt_()

                if group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    # Adaptive momentum based on gradient predictiveness
                    adaptive_momentum = group['momentum'] * (1 - torch.exp(-state['grad_norm_ema']))
                    buf.mul_(adaptive_momentum).addcdiv_(grad, avg)
                    p.add_(buf, alpha=-current_lr)
                else:
                    p.addcdiv_(grad, avg, value=-current_lr)

        return loss 