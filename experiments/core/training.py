import os
import sys
import time
import logging
from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import asdict, dataclass

import torch
from tqdm import tqdm

sys.path.append(os.getcwd())

from experiments.models.linear_transformer import in_context_loss

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Serialized state returned by the generic training loop.

    Parameters:
        history: Snapshot tensor with shape `(num_checkpoints, L, H, 2, d, d)`.
        history_iterations: Iteration numbers aligned with the saved snapshots.
        train_losses: Per-iteration training losses.
        grad_norms: Per-iteration gradient norms.
        elapsed_time: Total wall-clock runtime in seconds.
    """

    history: torch.Tensor
    history_iterations: List[int]
    train_losses: List[float]
    grad_norms: List[float]
    elapsed_time: float


def build_optimizer(
    optimizer_name: str,
    model: torch.nn.Module,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.9,
    momentum: float = 0.9,
    lbfgs_history_size: int = 100,
    lbfgs_max_iter: int = 4
):
    """Create an optimizer compatible with the reference experiments.

    Parameters:
        optimizer_name: Optimizer identifier.
        model: Model whose parameters should be optimized.
        learning_rate: Initial learning rate.
        beta1: First Adam-style coefficient.
        beta2: Second Adam-style coefficient.
        momentum: SGD momentum factor.
        lbfgs_history_size: History size for LBFGS.
        lbfgs_max_iter: Number of LBFGS inner iterations.
    """

    if optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr = learning_rate, betas = (beta1, beta2), weight_decay = 0.0)
    if optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr = learning_rate, momentum = momentum, weight_decay = 0.0)
    if optimizer_name == "lbfgs":
        return torch.optim.LBFGS(
            model.parameters(),
            history_size = lbfgs_history_size,
            max_iter = lbfgs_max_iter,
            lr = learning_rate
        )
    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def clip_gradients(
    allparam: torch.nn.Parameter,
    clip_value: Optional[float],
    clip_mode: str
) -> Optional[float]:
    """Clip gradients according to the requested policy.

    Parameters:
        allparam: Parameter tensor that stores all transformer matrices.
        clip_value: Gradient clipping threshold, if enabled.
        clip_mode: Clipping strategy (`overall` or `per_matrix`).
    """

    if allparam.grad is None:
        return None

    grad_tensor = allparam.grad
    if clip_value is None:
        return float(grad_tensor.norm().item())

    if clip_mode == "overall":
        norm_value = float(grad_tensor.norm().item())
        if norm_value > clip_value:
            grad_tensor.mul_(clip_value / norm_value)
        return norm_value

    if clip_mode == "per_matrix":
        last_norm = 0.0
        for layer_index in range(grad_tensor.shape[0]):
            for head_index in range(grad_tensor.shape[1]):
                for matrix_index in range(grad_tensor.shape[2]):
                    matrix_grad = grad_tensor[layer_index, head_index, matrix_index, :, :]
                    last_norm = float(matrix_grad.norm().item())
                    if last_norm > clip_value:
                        matrix_grad.mul_(clip_value / last_norm)
        return last_norm

    raise ValueError(f"Unsupported clip mode: {clip_mode}")


def apply_lr_decay(
    iteration: int,
    optimizer,
    decay_steps: List[int],
    decay_factor: float,
    visited_steps: set
) -> None:
    """Apply piecewise learning-rate decay at configured iterations.

    Parameters:
        iteration: Current outer-loop iteration.
        optimizer: Optimizer whose learning rate will be updated.
        decay_steps: Iterations at which to apply the multiplicative decay.
        decay_factor: Multiplicative decay factor.
        visited_steps: Mutable set used to guard against repeated decays.
    """

    if iteration in decay_steps and iteration not in visited_steps:
        optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] * decay_factor
        visited_steps.add(iteration)


def run_training_loop(
    model: torch.nn.Module,
    optimizer,
    optimizer_name: str,
    max_iters: int,
    checkpoint_stride: int,
    log_stride: int,
    batch_fetcher: Callable[[int], Tuple[torch.Tensor, torch.Tensor]],
    clip_value: Optional[float] = None,
    clip_mode: str = "overall",
    zero_p_after_step: bool = False,
    lr_decay_steps: Optional[List[int]] = None,
    lr_decay_factor: float = 0.5,
    progress_label: str = ""
) -> TrainingResult:
    """Execute a shared training loop for all experiment runners.

    Parameters:
        model: Transformer model to optimize.
        optimizer: Optimizer instance.
        optimizer_name: Optimizer identifier for branching logic.
        max_iters: Number of training iterations.
        checkpoint_stride: Iteration interval for history checkpoints.
        log_stride: Iteration interval for text logging.
        batch_fetcher: Callable that returns `(Z, y)` for a given iteration.
        clip_value: Gradient clipping threshold, if enabled.
        clip_mode: Clipping strategy (`overall` or `per_matrix`).
        zero_p_after_step: Whether to zero the `P` matrices after each update.
        lr_decay_steps: Optional iteration list for learning-rate decays.
        lr_decay_factor: Multiplicative decay factor.
        progress_label: Human-readable progress label shown by `tqdm`.
    """

    lr_decay_steps = lr_decay_steps or []
    history_tensors = []
    history_iterations = []
    train_losses = []
    grad_norms = []
    decay_markers = set()
    start_time = time.time()

    logger.info("=" * 80)
    logger.info("Training loop started: %s", progress_label or "unnamed_run")
    logger.info(
        "max_iters = %s | checkpoint_stride = %s | log_stride = %s | optimizer = %s",
        max_iters,
        checkpoint_stride,
        log_stride,
        optimizer_name
    )
    logger.info("=" * 80)

    progress_bar = tqdm(
        range(max_iters),
        desc = progress_label,
        leave = False,
        dynamic_ncols = True
    )
    for iteration in progress_bar:
        apply_lr_decay(iteration, optimizer, lr_decay_steps, lr_decay_factor, decay_markers)
        Z, y = batch_fetcher(iteration)

        if iteration % checkpoint_stride == 0:
            history_tensors.append(model.allparam.detach().clone().cpu())
            history_iterations.append(iteration)

        if optimizer_name == "lbfgs":
            def closure():
                """Closure required by the LBFGS optimizer.

                Parameters:
                    None: The closure captures the tensors from the outer scope.
                """

                optimizer.zero_grad()
                loss_value = in_context_loss(model, Z, y)
                loss_value.backward()
                return loss_value

            loss = in_context_loss(model, Z, y)
            loss.backward()
            grad_norm = float(model.allparam.grad.norm().item()) if model.allparam.grad is not None else 0.0
            optimizer.step(closure)
            optimizer.zero_grad()
        else:
            loss = in_context_loss(model, Z, y)
            loss.backward()
            clipped_norm = clip_gradients(model.allparam, clip_value, clip_mode)
            grad_norm = float(clipped_norm) if clipped_norm is not None else 0.0
            optimizer.step()
            optimizer.zero_grad()

        if zero_p_after_step:
            model.zero_p()

        train_losses.append(float(loss.item()))
        grad_norms.append(grad_norm)
        if iteration % log_stride == 0 or iteration < 5:
            logger.info(
                "iter %s | loss %.6f | gradnorm %.6f | lr %.6f",
                iteration,
                loss.item(),
                grad_norm,
                optimizer.param_groups[0]["lr"]
            )
        progress_bar.set_postfix(
            loss = f"{loss.item():.3e}",
            grad = f"{grad_norm:.3e}",
            lr = f"{optimizer.param_groups[0]['lr']:.3e}"
        )

    if not history_iterations or history_iterations[-1] != max_iters - 1:
        history_tensors.append(model.allparam.detach().clone().cpu())
        history_iterations.append(max_iters - 1)

    elapsed_time = time.time() - start_time
    logger.info("-" * 80)
    logger.info(
        "Training loop finished: %s | elapsed_time = %.2fs | final_loss = %.6f",
        progress_label or "unnamed_run",
        elapsed_time,
        train_losses[-1]
    )
    logger.info("-" * 80)
    return TrainingResult(
        history = torch.stack(history_tensors),
        history_iterations = history_iterations,
        train_losses = train_losses,
        grad_norms = grad_norms,
        elapsed_time = elapsed_time
    )
