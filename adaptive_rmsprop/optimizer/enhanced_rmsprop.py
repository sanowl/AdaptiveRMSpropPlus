import math
from collections.abc import Iterable
from typing import Callable, Optional

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
    - Optional gradient clipping, decoupled weight decay, and maximize mode

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
        min_lr_ratio (float, optional): Fraction of lr used as the cosine floor
            after warmup (default: 0.0)
        gradient_clip (float, optional): Maximum gradient norm per parameter before
            RMSprop statistics are updated (default: None)
        decoupled_weight_decay (bool, optional): Apply AdamW-style decay directly
            to parameters instead of adding L2 penalty to gradients (default: False)
        maximize (bool, optional): Optimize parameters to maximize the objective
            instead of minimizing it (default: False)
        noise_decay (float, optional): Power-law decay applied to gradient noise
            over optimization steps (default: 0.55)
        grad_norm_beta (float, optional): EMA factor used by adaptive momentum
            (default: 0.9)
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
        min_lr_ratio: float = 0.0,
        gradient_clip: Optional[float] = None,
        decoupled_weight_decay: bool = False,
        maximize: bool = False,
        noise_decay: float = 0.55,
        grad_norm_beta: float = 0.9,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= alpha < 1.0:
            raise ValueError(f"Invalid alpha value: {alpha}")
        if not 0.0 <= noise_scale:
            raise ValueError(f"Invalid noise scale: {noise_scale}")
        if not isinstance(warmup_steps, int) or warmup_steps < 0:
            raise ValueError(f"Invalid warmup_steps value: {warmup_steps}")
        if not isinstance(lr_cycles, int) or lr_cycles <= 0:
            raise ValueError(f"Invalid lr_cycles value: {lr_cycles}")
        if not 0.0 <= min_lr_ratio <= 1.0:
            raise ValueError(f"Invalid min_lr_ratio value: {min_lr_ratio}")
        if gradient_clip is not None and gradient_clip <= 0.0:
            raise ValueError(f"Invalid gradient_clip value: {gradient_clip}")
        if not 0.0 <= noise_decay:
            raise ValueError(f"Invalid noise_decay value: {noise_decay}")
        if not 0.0 <= grad_norm_beta < 1.0:
            raise ValueError(f"Invalid grad_norm_beta value: {grad_norm_beta}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            alpha=alpha,
            eps=eps,
            centered=centered,
            weight_decay=weight_decay,
            noise_scale=noise_scale,
            min_lr_ratio=min_lr_ratio,
            gradient_clip=gradient_clip,
            decoupled_weight_decay=decoupled_weight_decay,
            maximize=maximize,
            noise_decay=noise_decay,
            grad_norm_beta=grad_norm_beta,
        )
        super().__init__(params, defaults)
        self.base_lr = lr
        self.warmup_steps = warmup_steps
        self.lr_cycles = lr_cycles

    def _adjust_epsilon(
        self,
        grad_norm: torch.Tensor,
        eps: Optional[float] = None,
    ) -> torch.Tensor:
        """Dynamically adjust epsilon based on gradient magnitude"""
        base_eps = self.defaults["eps"] if eps is None else eps
        return base_eps * (1.0 + grad_norm.clamp_min(0).log1p())

    def _compute_cyclic_lr(
        self,
        step: int,
        base_lr: Optional[float] = None,
        min_lr_ratio: Optional[float] = None,
    ) -> float:
        """Implement cyclical learning rate with warmup"""
        lr = self.base_lr if base_lr is None else base_lr
        lr_floor = self.defaults["min_lr_ratio"] if min_lr_ratio is None else min_lr_ratio

        if self.warmup_steps > 0 and step < self.warmup_steps:
            return lr * (step / self.warmup_steps)

        cycle_length = max(1, self.lr_cycles * max(1, self.warmup_steps))
        cycle_step = step - self.warmup_steps
        if self.warmup_steps == 0:
            cycle_step = max(0, step - 1)

        cycle_progress = (cycle_step % cycle_length) / cycle_length
        cosine_scale = 0.5 * (1.0 + math.cos(math.pi * cycle_progress))
        return lr * (lr_floor + (1.0 - lr_floor) * cosine_scale)

    def _inject_gradient_noise(
        self,
        grad: torch.Tensor,
        step: int,
        noise_scale: Optional[float] = None,
        noise_decay: Optional[float] = None,
    ) -> torch.Tensor:
        """Add adaptive gradient noise for better exploration"""
        scale = self.defaults["noise_scale"] if noise_scale is None else noise_scale
        decay = self.defaults["noise_decay"] if noise_decay is None else noise_decay
        if scale == 0:
            return grad

        scaled_noise = scale / (1.0 + step) ** decay
        noise = torch.randn_like(grad) * scaled_noise * grad.abs().mean()
        return grad + noise

    def _clip_gradient(self, grad: torch.Tensor, max_norm: Optional[float]) -> torch.Tensor:
        """Clip a gradient tensor by norm without mutating p.grad."""
        if max_norm is None:
            return grad

        grad_norm = grad.norm()
        if grad_norm <= max_norm:
            return grad

        return grad * (max_norm / (grad_norm + 1e-12))

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
                if group["maximize"]:
                    grad = -grad

                state = self.state[p]

                # Enhanced state initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["square_avg"] = torch.ones_like(p, memory_format=torch.preserve_format)
                    state["grad_norm_ema"] = torch.zeros((), device=p.device, dtype=p.dtype)
                    if group["momentum"] > 0:
                        state["momentum_buffer"] = torch.zeros_like(
                            p,
                            memory_format=torch.preserve_format,
                        )
                    if group["centered"]:
                        state["grad_avg"] = torch.zeros_like(
                            p,
                            memory_format=torch.preserve_format,
                        )

                square_avg = state["square_avg"]
                alpha = group["alpha"]

                state["step"] += 1
                step = state["step"]

                grad = self._clip_gradient(grad, group["gradient_clip"])

                # Compute gradient norm and adjust epsilon
                grad_norm = grad.norm()
                dynamic_eps = self._adjust_epsilon(grad_norm, group["eps"])

                # Update gradient norm EMA
                grad_norm_beta = group["grad_norm_beta"]
                state["grad_norm_ema"].mul_(grad_norm_beta).add_(
                    grad_norm,
                    alpha=1 - grad_norm_beta,
                )

                # Inject gradient noise
                grad = self._inject_gradient_noise(
                    grad,
                    step,
                    group["noise_scale"],
                    group["noise_decay"],
                )

                if group["weight_decay"] != 0 and not group["decoupled_weight_decay"]:
                    grad = grad.add(p, alpha=group["weight_decay"])

                square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

                # Compute adaptive learning rate
                current_lr = self._compute_cyclic_lr(
                    step,
                    group["lr"],
                    group["min_lr_ratio"],
                )

                if group["weight_decay"] != 0 and group["decoupled_weight_decay"]:
                    p.mul_(1 - current_lr * group["weight_decay"])

                if group["centered"]:
                    grad_avg = state["grad_avg"]
                    grad_avg.mul_(alpha).add_(grad, alpha=1 - alpha)
                    avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).add_(dynamic_eps).sqrt_()
                else:
                    avg = square_avg.add(dynamic_eps).sqrt_()

                if group["momentum"] > 0:
                    buf = state["momentum_buffer"]
                    # Adaptive momentum based on gradient predictiveness
                    adaptive_momentum = group["momentum"] * (
                        1 - torch.exp(-state["grad_norm_ema"])
                    )
                    buf.mul_(adaptive_momentum).addcdiv_(grad, avg)
                    p.add_(buf, alpha=-current_lr)
                else:
                    p.addcdiv_(grad, avg, value=-current_lr)

        return loss 