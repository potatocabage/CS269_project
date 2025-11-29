import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple


def numpy_sdf_penalty(samples: np.ndarray, obstacles: List[Tuple[float, float, float]], margin: float = 0.2, alpha: float = 10.0) -> np.ndarray:
    """
    Compute an SDF-based softplus penalty for numpy samples.

    Args:
        samples: (B, T, 2) array of trajectories or (N, 2) points.
        obstacles: list of (cx, cy, r).
        margin: safety margin (m) used in softplus(m - d_signed).
        alpha: sharpness of softplus.

    Returns:
        If input is (B, T, 2) -> returns (B,) penalty averaged over time and obstacles.
        If input is (N, 2) -> returns (N,) penalty averaged over obstacles.
    """
    if len(obstacles) == 0:
        if samples.ndim == 3:
            return np.zeros((samples.shape[0],), dtype=float)
        elif samples.ndim == 2:
            return np.zeros((samples.shape[0],), dtype=float)

    obs_np = np.array(obstacles, dtype=float)  # (K,3)
    centers = obs_np[:, :2]  # (K,2)
    radii = obs_np[:, 2]  # (K,)

    if samples.ndim == 3:
        B, T, _ = samples.shape
        pts = samples.reshape(-1, 2)  # (B*T, 2)
        # compute (B*T, K) distances
        dists = np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=2)
        d_signed = dists - radii[None, :]
        obs_pen = np.log1p(np.exp(alpha * (margin - d_signed)))
        obs_pen_mean = obs_pen.mean(axis=1).reshape(B, T)
        return obs_pen_mean.mean(axis=1)

    elif samples.ndim == 2:
        pts = samples  # (N,2)
        dists = np.linalg.norm(pts[:, None, :] - centers[None, :, :], axis=2)
        d_signed = dists - radii[None, :]
        obs_pen = np.log1p(np.exp(alpha * (margin - d_signed)))
        return obs_pen.mean(axis=1)
    else:
        raise ValueError("Unsupported samples ndim for numpy_sdf_penalty")


def torch_sdf_penalty(actions: torch.Tensor, obstacles: List[Tuple[float, float, float]], margin: float = 0.2, alpha: float = 10.0) -> torch.Tensor | None:
    """
    Compute SDF penalty for torch tensors. Returns (B,) penalty averaged over time and obstacles.
    If no obstacles provided, returns None.
    """
    if obstacles is None or len(obstacles) == 0:
        return None

    device = actions.device
    dtype = actions.dtype
    obs_np = np.array(obstacles, dtype=float)
    centers = torch.tensor(obs_np[:, :2], dtype=dtype, device=device)  # (K,2)
    radii = torch.tensor(obs_np[:, 2], dtype=dtype, device=device)  # (K,)

    # actions: (B, T, 2)
    pts = actions.unsqueeze(2)  # (B, T, 1, 2)
    ctr = centers.unsqueeze(0).unsqueeze(0)  # (1,1,K,2)
    dists = torch.linalg.norm(pts - ctr, dim=-1)  # (B, T, K)
    d_signed = dists - radii.unsqueeze(0).unsqueeze(0)  # (B, T, K)
    obs_pen = F.softplus(alpha * (margin - d_signed))
    obs_term = obs_pen.mean(dim=(1, 2))  # (B,)
    return obs_term
