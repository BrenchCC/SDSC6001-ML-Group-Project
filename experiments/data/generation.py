import math
import os
import sys
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.append(os.getcwd())

logger = logging.getLogger(__name__)


def build_rotation_bundle(
    dimension: int,
    diagonal_values: List[float],
    seed: int,
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """Construct the rotated covariance factors used by the paper experiments.

    Parameters:
        dimension: Covariate dimension.
        diagonal_values: Diagonal entries used for the square-root covariance matrix.
        seed: Random seed used to sample the orthogonal basis.
        device: Torch device for generated tensors.
    """

    logger.info(
        "Generating rotation bundle: seed = %s | dimension = %s | device = %s",
        seed,
        dimension,
        device
    )
    torch.manual_seed(seed)
    gaussian = torch.FloatTensor(dimension, dimension).uniform_(-1.0, 1.0).to(device)
    U = torch.linalg.svd(gaussian)[0].to(device)
    D = torch.diag(torch.tensor(diagonal_values, dtype = torch.float32, device = device))
    return {"U": U, "D": D}


def generate_data(
    mode: str = "normal",
    N: int = 20,
    d: int = 1,
    B: int = 1000,
    shape_k: float = 0.1,
    U: Optional[torch.Tensor] = None,
    D: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate a batch of synthetic linear-regression prompts.

    Parameters:
        mode: Supported data mode (`normal`, `sphere`, or `gamma`).
        N: Number of in-context examples.
        d: Covariate dimension.
        B: Batch size.
        shape_k: Shape parameter used by the gamma generator.
        U: Optional orthogonal matrix for rotated covariance settings.
        D: Optional diagonal matrix for rotated covariance settings.
        device: Device on which the tensors are created.
    """

    if device is None:
        device = torch.device("cpu")

    logger.info(
        "Generating synthetic batch: mode = %s | N = %s | d = %s | B = %s | rotated = %s | device = %s",
        mode,
        N,
        d,
        B,
        U is not None and D is not None,
        device
    )
    W = torch.FloatTensor(B, d).normal_(0.0, 1.0).to(device)
    X = torch.FloatTensor(B, N, d).normal_(0.0, 1.0).to(device)
    X_test = torch.FloatTensor(B, 1, d).normal_(0.0, 1.0).to(device)

    if U is not None and D is not None:
        U = U.to(device)
        D = D.to(device)
        W = torch.FloatTensor(B, d).normal_(0.0, 1.0).to(device)
        W = torch.mm(W, torch.inverse(D))
        W = torch.mm(W, U.t())

    if mode == "sphere":
        X.div_(X.norm(p = 2, dim = 2)[:, :, None])
        X_test.div_(X_test.norm(p = 2, dim = 2)[:, :, None])
    elif mode == "gamma":
        gamma_scales = np.random.gamma(shape = shape_k, scale = (10.0 / shape_k) ** 0.5, size = [B, N])
        gamma_test_scales = np.random.gamma(shape = shape_k, scale = (10.0 / shape_k) ** 0.5, size = [B, 1])
        gamma_scales = torch.tensor(gamma_scales, dtype = torch.float32, device = device).sqrt()
        gamma_test_scales = torch.tensor(gamma_test_scales, dtype = torch.float32, device = device).sqrt()
        X.div_(X.norm(p = 2, dim = 2)[:, :, None])
        X_test.div_(X_test.norm(p = 2, dim = 2)[:, :, None])
        X.mul_(gamma_scales[:, :, None])
        X_test.mul_(gamma_test_scales[:, :, None])
    elif mode != "normal":
        raise ValueError(f"Unsupported data mode: {mode}")

    if U is not None and D is not None:
        X = torch.einsum("ij,jk,BNk->BNi", U, D, X)
        X_test = torch.einsum("ij,jk,BNk->BNi", U, D, X_test)

    y = torch.einsum("bi,bni->bn", W, X).unsqueeze(2)
    y_zero = torch.zeros(B, 1, 1, device = device)
    y_test = torch.einsum("bi,bni->bn", W, X_test).squeeze(1)
    X_combined = torch.cat([X, X_test], dim = 1)
    y_combined = torch.cat([y, y_zero], dim = 1)
    Z = torch.cat([X_combined, y_combined], dim = 2)
    return Z.to(device), y_test.to(device)


def generate_data_inplace(
    Z: torch.Tensor,
    U: Optional[torch.Tensor] = None,
    D: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Refresh a prompt tensor in place while preserving its shape.

    Parameters:
        Z: Existing prompt tensor to be overwritten.
        U: Optional orthogonal matrix for rotated covariance settings.
        D: Optional diagonal matrix for rotated covariance settings.
    """

    batch_size = Z.shape[0]
    dimension = Z.shape[2] - 1
    device = Z.device
    logger.debug(
        "Refreshing synthetic batch in place: batch_size = %s | dimension = %s | rotated = %s | device = %s",
        batch_size,
        dimension,
        U is not None and D is not None,
        device
    )
    X = Z[:, :, 0:-1]
    X.normal_(0.0, 1.0)
    W = torch.FloatTensor(batch_size, dimension).normal_(0.0, 1.0).to(device)

    if U is not None and D is not None:
        U = U.to(device)
        D = D.to(device)
        W = torch.mm(W, torch.inverse(D))
        W = torch.mm(W, U.t())
        Z[:, :, 0:-1] = torch.einsum("ij,jk,BNk->BNi", U, D, X)

    Z[:, :, -1] = torch.einsum("bi,bni->bn", W, Z[:, :, 0:-1])
    y_test = Z[:, -1, -1].detach().clone()
    Z[:, -1, -1].zero_()
    return Z.to(device), y_test.to(device)


def build_eval_bundle(
    mode: str,
    context_length: int,
    dimension: int,
    batch_size: int,
    shape_k: float,
    seed: int,
    device: torch.device,
    U: Optional[torch.Tensor] = None,
    D: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    """Create a reproducible evaluation bundle for later reuse.

    Parameters:
        mode: Synthetic data mode.
        context_length: Number of in-context examples.
        dimension: Covariate dimension.
        batch_size: Evaluation batch size.
        shape_k: Gamma shape parameter.
        seed: Random seed for reproducibility.
        device: Device on which to create the tensors.
        U: Optional orthogonal covariance factor.
        D: Optional diagonal covariance factor.
    """

    logger.info(
        "Building evaluation bundle: seed = %s | context_length = %s | dimension = %s | batch_size = %s",
        seed,
        context_length,
        dimension,
        batch_size
    )
    np.random.seed(seed)
    torch.manual_seed(seed)
    Z, y = generate_data(
        mode = mode,
        N = context_length,
        d = dimension,
        B = batch_size,
        shape_k = shape_k,
        U = U,
        D = D,
        device = device
    )
    return {"Z": Z.detach().cpu(), "y": y.detach().cpu()}
