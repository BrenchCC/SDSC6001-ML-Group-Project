import os
import sys
import logging
from typing import Callable, Dict, List, Optional, Sequence

import torch

sys.path.append(os.getcwd())

logger = logging.getLogger(__name__)


def compute_dist_identity(matrix: torch.Tensor) -> float:
    """Compute the scale-invariant distance to the identity matrix.

    Parameters:
        matrix: Matrix whose distance to a scaled identity is measured.
    """

    scale = torch.sum(torch.diagonal(matrix)) / matrix.shape[0]
    ideal_identity = scale * torch.eye(matrix.shape[0], device = matrix.device, dtype = matrix.dtype)
    difference = matrix - ideal_identity
    return float((torch.norm(difference, p = "fro") / torch.norm(matrix, p = "fro")).item())


def rotate_q_matrix(matrix: torch.Tensor, U: torch.Tensor, D: torch.Tensor) -> torch.Tensor:
    """Rotate a learned `Q` matrix into the covariance-normalized basis.

    Parameters:
        matrix: Matrix to rotate.
        U: Orthogonal covariance factor.
        D: Diagonal covariance factor.
    """

    UD = torch.mm(U, D)
    return torch.mm(torch.mm(UD.t(), matrix), UD)


def evaluate_history_losses(
    history: torch.Tensor,
    eval_Z: torch.Tensor,
    eval_y: torch.Tensor,
    model_builder: Callable[[], torch.nn.Module],
    device: torch.device
) -> List[float]:
    """Evaluate every checkpoint in a saved history on a fixed dataset.

    Parameters:
        history: Checkpoint tensor stack with shape `(num_checkpoints, ...)`.
        eval_Z: Evaluation prompt tensor.
        eval_y: Evaluation query labels.
        model_builder: Callable that returns a fresh model instance.
        device: Device used for the forward pass.
    """

    losses = []
    eval_Z = eval_Z.to(device)
    eval_y = eval_y.to(device)
    model = model_builder().to(device)

    from experiments.models.linear_transformer import in_context_loss

    for checkpoint in history:
        with torch.no_grad():
            model.allparam.copy_(checkpoint.to(device))
        loss_value = in_context_loss(model, eval_Z, eval_y)
        losses.append(float(loss_value.item()))
    return losses


def select_best_checkpoint(losses: Sequence[float], tail_window: int) -> int:
    """Select the best checkpoint index from the tail of a loss sequence.

    Parameters:
        losses: Sequence of checkpoint losses.
        tail_window: Number of final checkpoints to search.
    """

    start_index = max(0, len(losses) - tail_window)
    tail_losses = list(losses[start_index:])
    best_offset = tail_losses.index(min(tail_losses))
    return start_index + best_offset


def compute_distance_curves(
    history: torch.Tensor,
    n_layer: int,
    include_p: bool,
    rotate_q: bool = False,
    U: Optional[torch.Tensor] = None,
    D: Optional[torch.Tensor] = None
) -> Dict[str, List[float]]:
    """Compute distance-to-identity curves for all relevant matrices.

    Parameters:
        history: Checkpoint tensor stack.
        n_layer: Number of transformer layers.
        include_p: Whether to include the `P` matrices.
        rotate_q: Whether to rotate `Q` matrices into the covariance basis.
        U: Optional orthogonal covariance factor.
        D: Optional diagonal covariance factor.
    """

    curves: Dict[str, List[float]] = {}
    for layer_index in range(n_layer):
        for matrix_index in range(2):
            if matrix_index == 0 and (not include_p or layer_index == n_layer - 1):
                continue
            label = f"B{layer_index}" if matrix_index == 0 else f"A{layer_index}"
            curves[label] = []
            for checkpoint in history:
                matrix = checkpoint[layer_index, 0, matrix_index, :, :].clone()
                if matrix_index == 1 and rotate_q:
                    if U is None or D is None:
                        raise ValueError("U and D must be provided when rotate_q is enabled.")
                    matrix = rotate_q_matrix(matrix, U, D)
                curves[label].append(compute_dist_identity(matrix))
    return curves


def extract_final_matrices(
    history: torch.Tensor,
    n_layer: int,
    include_p: bool,
    rotate_q: bool = False,
    U: Optional[torch.Tensor] = None,
    D: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    """Extract the final learned matrices for visualization.

    Parameters:
        history: Checkpoint tensor stack.
        n_layer: Number of transformer layers.
        include_p: Whether to include the `P` matrices.
        rotate_q: Whether to rotate `Q` matrices into the covariance basis.
        U: Optional orthogonal covariance factor.
        D: Optional diagonal covariance factor.
    """

    final_state = history[-1]
    matrices: Dict[str, torch.Tensor] = {}
    for layer_index in range(n_layer):
        for matrix_index in range(2):
            if matrix_index == 0 and (not include_p or layer_index == n_layer - 1):
                continue
            label = f"B{layer_index}" if matrix_index == 0 else f"A{layer_index}"
            matrix = final_state[layer_index, 0, matrix_index, :, :].clone()
            if matrix_index == 1 and rotate_q:
                if U is None or D is None:
                    raise ValueError("U and D must be provided when rotate_q is enabled.")
                matrix = rotate_q_matrix(matrix, U, D)
            matrices[label] = matrix
    return matrices


def mean_std_from_seed_curves(seed_curves: List[List[float]]) -> Dict[str, List[float]]:
    """Aggregate multiple per-seed curves into mean and standard deviation.

    Parameters:
        seed_curves: List of equally sized numeric curves.
    """

    curve_tensor = torch.tensor(seed_curves, dtype = torch.float32)
    if curve_tensor.shape[0] == 1:
        zero_std = torch.zeros_like(curve_tensor[0])
        return {
            "mean": curve_tensor.mean(dim = 0).tolist(),
            "std": zero_std.tolist()
        }
    return {
        "mean": curve_tensor.mean(dim = 0).tolist(),
        "std": curve_tensor.std(dim = 0).tolist()
    }
