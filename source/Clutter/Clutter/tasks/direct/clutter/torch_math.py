"""Small PyTorch math helpers used by the migrated Clutter grasp task.

旧 DemoGrasp 通过 `isaacgymenvs.utils.torch_jit_utils` 获得这些函数。Clutter
迁到 IsaacLab 后不再依赖 IsaacGymEnvs，因此这里提供一组轻量替代实现。

注意：本模块统一使用 IsaacLab 的四元数顺序 `wxyz`。旧 DemoGrasp 的 pkl/IsaacGym
数据若是 `xyzw`，应在加载时先转换，再调用这里的函数。
"""

from __future__ import annotations

from typing import Any

import torch


def to_torch(
    value: Any,
    dtype: torch.dtype = torch.float,
    device: str | torch.device | None = None,
    requires_grad: bool = False,
) -> torch.Tensor:
    """Convert Python/NumPy data to a torch tensor on the requested device.

    这是 IsaacGymEnvs `to_torch()` 的兼容替代。旧代码大量用它把配置里的 list、
    NumPy 数组或标量搬到 GPU；迁移后保留这个名字可以降低旧逻辑复用成本。
    """

    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    tensor.requires_grad_(requires_grad)
    return tensor


def torch_rand_float(
    lower: float | torch.Tensor,
    upper: float | torch.Tensor,
    shape: tuple[int, ...] | torch.Size,
    device: str | torch.device,
) -> torch.Tensor:
    """Sample uniform random floats in `[lower, upper]`.

    旧 reset/randomization 代码用它批量采样位置、角度、桌面高度等随机量。
    `lower` 和 `upper` 可以是标量，也可以是能广播到 `shape` 的 tensor。
    """

    lower_t = torch.as_tensor(lower, dtype=torch.float, device=device)
    upper_t = torch.as_tensor(upper, dtype=torch.float, device=device)
    return (upper_t - lower_t) * torch.rand(shape, dtype=torch.float, device=device) + lower_t


def scale(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Map normalized values from `[-1, 1]` to `[lower, upper]`.

    常用于把策略输出动作还原成真实关节目标。
    """

    return 0.5 * (x + 1.0) * (upper - lower) + lower


def unscale(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Map values from `[lower, upper]` back to the normalized `[-1, 1]` range.

    常用于把关节位置编码成策略观测，或把参考轨迹转成动作。
    """

    return 2.0 * (x - lower) / (upper - lower + 1.0e-6) - 1.0


def tensor_clamp(x: torch.Tensor, lower: torch.Tensor | float, upper: torch.Tensor | float) -> torch.Tensor:
    """Clamp a tensor with scalar or tensor lower/upper bounds."""

    lower_t = torch.as_tensor(lower, dtype=x.dtype, device=x.device)
    upper_t = torch.as_tensor(upper, dtype=x.dtype, device=x.device)
    return torch.maximum(torch.minimum(x, upper_t), lower_t)


def normalize_quat(quat: torch.Tensor) -> torch.Tensor:
    """Normalize `wxyz` quaternions and avoid division by zero."""

    return quat / torch.clamp(torch.linalg.norm(quat, dim=-1, keepdim=True), min=1.0e-6)


def quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    """Return the conjugate of a `wxyz` quaternion."""

    quat = normalize_quat(quat)
    return torch.cat((quat[..., 0:1], -quat[..., 1:4]), dim=-1)


def quat_mul(q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    """Multiply two `wxyz` quaternions.

    四元数乘法用于组合旋转，也可配合共轭旋转点或计算姿态误差。
    """

    w0, x0, y0, z0 = q0.unbind(dim=-1)
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    return torch.stack(
        (
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ),
        dim=-1,
    )


def quat_apply(quat: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Rotate 3-D points/vectors by `wxyz` quaternions.

    支持 `quat` 的 batch 维广播到 `points`，例如 `[B, 1, 4]` 旋转 `[B, N, 3]`。
    """

    quat = normalize_quat(quat)
    q_vec = quat[..., 1:4]
    q_w = quat[..., 0:1]
    if q_vec.shape[:-1] != points.shape[:-1]:
        q_vec = torch.broadcast_to(q_vec, points.shape[:-1] + (3,))
        q_w = torch.broadcast_to(q_w, points.shape[:-1] + (1,))
    uv = torch.cross(q_vec, points, dim=-1)
    uuv = torch.cross(q_vec, uv, dim=-1)
    return points + 2.0 * (q_w * uv + uuv)


def quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Create `wxyz` quaternions from angle-axis rotations."""

    if angle.ndim == 0:
        # 标量角度配单个 axis 时，扩展成和 axis batch 维一致的形状。
        angle = angle.expand(axis.shape[:-1])
    elif angle.ndim == axis.ndim and angle.shape[-1] == 1:
        # 只移除显式的最后一维 `[B, 1]`，避免把 `[1]` 误压成 0 维标量。
        angle = angle.squeeze(-1)
    axis = axis / torch.clamp(torch.linalg.norm(axis, dim=-1, keepdim=True), min=1.0e-6)
    half_angle = 0.5 * angle
    quat = torch.cat((torch.cos(half_angle).unsqueeze(-1), axis * torch.sin(half_angle).unsqueeze(-1)), dim=-1)
    return normalize_quat(quat)


def quat_diff_rad(q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    """Return the shortest angular distance between two `wxyz` quaternions in radians."""

    q0 = normalize_quat(q0)
    q1 = normalize_quat(q1)
    dot = torch.abs(torch.sum(q0 * q1, dim=-1))
    dot = torch.clamp(dot, -1.0, 1.0)
    return 2.0 * torch.acos(dot)


def slerp(q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Spherical linear interpolation between two `wxyz` quaternions.

    `t` 可以是标量、`[B]`、`[B, 1]` 或能广播到四元数 batch 维的 tensor。
    当两姿态非常接近时，退化成线性插值以避免除以很小的 `sin(theta)`。
    """

    q0 = normalize_quat(q0)
    q1 = normalize_quat(q1)
    while t.ndim < q0.ndim:
        t = t.unsqueeze(-1)

    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = torch.abs(dot).clamp(-1.0, 1.0)

    close = dot > 0.9995
    linear = normalize_quat(q0 + t * (q1 - q0))

    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta = theta_0 * t
    sin_theta = torch.sin(theta)
    s0 = torch.cos(theta) - dot * sin_theta / torch.clamp(sin_theta_0, min=1.0e-6)
    s1 = sin_theta / torch.clamp(sin_theta_0, min=1.0e-6)
    spherical = normalize_quat(s0 * q0 + s1 * q1)

    return torch.where(close, linear, spherical)
