import os
import sys
import logging
from typing import Callable, List, Optional, Tuple

import torch
from tqdm import tqdm

sys.path.append(os.getcwd())

logger = logging.getLogger(__name__)


def do_gd(Z: torch.Tensor, eta: float, numstep: int, dimension: int, device: torch.device) -> torch.Tensor:
    """Run a fixed number of vanilla GD steps on a single prompt instance.

    Parameters:
        Z: Prompt tensor for one task.
        eta: Step size.
        numstep: Number of GD updates.
        dimension: Covariate dimension.
        device: Device used for linear algebra.
    """

    context_length = Z.shape[0] - 1
    X = Z[0:context_length, 0:dimension].to(device)
    Y = Z[0:context_length, dimension].to(device)
    weights = torch.zeros(X.shape[1], device = device)
    for _ in range(numstep):
        XTXw = torch.einsum("ik,ij,j->k", X, X, weights)
        XTY = torch.einsum("ik,i->k", X, Y)
        grad = XTXw - XTY
        weights = weights - eta * grad
    return weights


def do_preconditioned_gd(
    Z: torch.Tensor,
    eta: float,
    numstep: int,
    dimension: int,
    U: torch.Tensor,
    D: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """Run a fixed number of preconditioned GD steps on a single prompt instance.

    Parameters:
        Z: Prompt tensor for one task.
        eta: Step size.
        numstep: Number of updates.
        dimension: Covariate dimension.
        U: Orthogonal covariance factor.
        D: Diagonal covariance factor.
        device: Device used for linear algebra.
    """

    context_length = Z.shape[0] - 1
    X = Z[0:context_length, 0:dimension].to(device)
    Y = Z[0:context_length, dimension].to(device)
    weights = torch.zeros(X.shape[1], device = device)
    X = torch.einsum("ij,jk,Nk->Ni", torch.inverse(D), U.t(), X)
    for _ in range(numstep):
        XTXw = torch.einsum("ik,ij,j->k", X, X, weights)
        XTY = torch.einsum("ik,i->k", X, Y)
        grad = XTXw - XTY
        weights = weights - eta * grad
    return weights


def do_ols(
    Z: torch.Tensor,
    dimension: int,
    U: torch.Tensor,
    D: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """Compute the OLS predictor in the covariance-normalized basis.

    Parameters:
        Z: Prompt tensor for one task.
        dimension: Covariate dimension.
        U: Orthogonal covariance factor.
        D: Diagonal covariance factor.
        device: Device used for linear algebra.
    """

    context_length = Z.shape[0] - 1
    X = Z[0:context_length, 0:dimension].to(device)
    Y = Z[0:context_length, dimension].to(device)
    X = torch.einsum("ij,jk,Nk->Ni", torch.inverse(D), U.t(), X)
    XTX = torch.einsum("ik,ij->kj", X, X)
    XTY = torch.einsum("ik,i->k", X, Y)
    return torch.einsum("ik,k->i", torch.linalg.pinv(XTX), XTY)


def eval_prediction(
    Z: torch.Tensor,
    y_test: torch.Tensor,
    weights: torch.Tensor,
    dimension: int,
    device: torch.device,
    U: Optional[torch.Tensor] = None,
    D: Optional[torch.Tensor] = None
) -> float:
    """Evaluate a predictor on the query example of one prompt instance.

    Parameters:
        Z: Prompt tensor for one task.
        y_test: Ground-truth query label.
        weights: Predictor weights.
        dimension: Covariate dimension.
        device: Device used for linear algebra.
        U: Optional orthogonal covariance factor.
        D: Optional diagonal covariance factor.
    """

    X_test = Z[-1, 0:dimension].to(device)
    if U is not None and D is not None:
        X_test = torch.einsum("ij,jk,k->i", torch.inverse(D), U.t(), X_test)
    prediction = torch.einsum("i,i->", weights, X_test)
    return float(((y_test.to(device) - prediction) ** 2).item())


def evaluate_baseline_loss(
    Z: torch.Tensor,
    y: torch.Tensor,
    method_name: str,
    dimension: int,
    numstep: int,
    eta: Optional[float],
    device: torch.device,
    U: Optional[torch.Tensor] = None,
    D: Optional[torch.Tensor] = None,
    max_samples: Optional[int] = None
) -> float:
    """Estimate the mean loss of a baseline method on a dataset.

    Parameters:
        Z: Prompt batch.
        y: Query-label batch.
        method_name: Baseline identifier (`gd`, `pgd`, or `ols`).
        dimension: Covariate dimension.
        numstep: Number of optimizer steps for iterative baselines.
        eta: Step size for iterative baselines.
        device: Device used for linear algebra.
        U: Optional orthogonal covariance factor.
        D: Optional diagonal covariance factor.
        max_samples: Optional truncation of the dataset size.
    """

    total_loss = 0.0
    sample_count = min(Z.shape[0], max_samples) if max_samples is not None else Z.shape[0]
    for sample_index in range(sample_count):
        Zi = Z[sample_index, :, :]
        yi = y[sample_index]
        if method_name == "gd":
            if eta is None:
                raise ValueError("eta must be provided for GD.")
            weights = do_gd(Zi, eta, numstep, dimension, device)
            total_loss += eval_prediction(Zi, yi, weights, dimension, device)
        elif method_name == "pgd":
            if eta is None or U is None or D is None:
                raise ValueError("eta, U, and D must be provided for preconditioned GD.")
            weights = do_preconditioned_gd(Zi, eta, numstep, dimension, U, D, device)
            total_loss += eval_prediction(Zi, yi, weights, dimension, device, U, D)
        elif method_name == "ols":
            if U is None or D is None:
                raise ValueError("U and D must be provided for OLS.")
            weights = do_ols(Zi, dimension, U, D, device)
            total_loss += eval_prediction(Zi, yi, weights, dimension, device, U, D)
        else:
            raise ValueError(f"Unsupported baseline method: {method_name}")
    return total_loss / sample_count


def find_best_eta(
    Z: torch.Tensor,
    y: torch.Tensor,
    method_name: str,
    eta_grid: List[float],
    dimension: int,
    numstep: int,
    device: torch.device,
    U: Optional[torch.Tensor] = None,
    D: Optional[torch.Tensor] = None,
    max_samples: Optional[int] = None
) -> Tuple[float, float]:
    """Grid-search the best step size for an iterative baseline.

    Parameters:
        Z: Validation prompt batch.
        y: Validation query-label batch.
        method_name: Baseline identifier (`gd` or `pgd`).
        eta_grid: Candidate step-size grid.
        dimension: Covariate dimension.
        numstep: Number of optimizer steps.
        device: Device used for linear algebra.
        U: Optional orthogonal covariance factor.
        D: Optional diagonal covariance factor.
        max_samples: Optional number of validation samples.
    """

    best_eta = eta_grid[0]
    best_loss = float("inf")
    for eta in eta_grid:
        loss_value = evaluate_baseline_loss(
            Z = Z,
            y = y,
            method_name = method_name,
            dimension = dimension,
            numstep = numstep,
            eta = eta,
            device = device,
            U = U,
            D = D,
            max_samples = max_samples
        )
        if loss_value < best_loss:
            best_eta = eta
            best_loss = loss_value
    return best_eta, best_loss
