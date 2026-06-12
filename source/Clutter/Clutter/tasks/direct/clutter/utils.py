"""Utility functions for the migrated Clutter grasp task.

本文件从旧 DemoGrasp 的 `tasks/utils.py` 迁移而来，保留位姿插值、颜色表、
点云加载、点云变换、最远点采样和索引等工具。旧文件依赖
`isaacgymenvs.utils.torch_jit_utils`，迁移后改为依赖同目录的 `torch_math.py`。

注意：Clutter/IsaacLab 任务内部统一使用 `wxyz` 四元数顺序。
"""

from __future__ import annotations

import os

import numpy as np
import torch

from .torch_math import quat_apply, quat_diff_rad, slerp


def batch_linear_interpolate_poses(
    pose1: torch.Tensor,
    pose2: torch.Tensor,
    max_trans_step: float,
    max_rot_step: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch interpolate between two sets of poses.

    Args:
        pose1: 起始位姿，形状 `[B, 7]`，内容为 `xyz + wxyz`。
        pose2: 目标位姿，形状 `[B, 7]`，与 `pose1` 一一对应。
        max_trans_step: 相邻插值点之间允许的最大平移距离。
        max_rot_step: 相邻插值点之间允许的最大旋转角度，单位为弧度。

    Returns:
        interp_poses: 插值后的位姿序列，形状 `[B, T_max + 1, 7]`，包含起点和终点。
        timesteps: 每个 batch 样本自己的有效插值步数，形状 `[B]`。
    """

    if pose1.shape != pose2.shape or pose1.shape[-1] != 7:
        raise ValueError(f"pose1/pose2 must both have shape [B, 7], got {pose1.shape} and {pose2.shape}.")

    batch_size = pose1.shape[0]
    device = pose1.device

    # 将 pose 拆成位置和姿态；姿态使用 IsaacLab 的 wxyz 顺序。
    pos1, quat1 = pose1[:, :3], pose1[:, 3:]
    pos2, quat2 = pose2[:, :3], pose2[:, 3:]

    # 按平移距离和旋转角距离分别估计所需步数，再取更保守的较大值。
    trans_dist = torch.norm(pos2 - pos1, dim=-1)
    max_trans_step = max(float(max_trans_step), 1.0e-6)
    max_rot_step = max(float(max_rot_step), 1.0e-6)
    trans_steps = torch.ceil(trans_dist / max_trans_step).long().clamp(min=1)
    rot_steps = torch.ceil(quat_diff_rad(quat1, quat2) / max_rot_step).long().clamp(min=1)
    timesteps = torch.maximum(trans_steps, rot_steps)
    max_steps = int(timesteps.max().item())

    # 不同样本可能需要不同步数，这里用同一个 T_max 承载，并用 mask 清理无效段。
    step_idx = torch.arange(max_steps + 1, device=device).expand(batch_size, -1)
    valid_mask = step_idx <= timesteps.unsqueeze(1)
    t = step_idx.float() / timesteps.unsqueeze(1).clamp(min=1)
    t = t * valid_mask.float()

    # 位置用线性插值，姿态用 SLERP，保证到目标姿态的旋转速度更平滑。
    interp_pos = pos1.unsqueeze(1) + t.unsqueeze(-1) * (pos2 - pos1).unsqueeze(1)
    interp_quat = slerp(
        quat1.unsqueeze(1).expand(-1, max_steps + 1, -1),
        quat2.unsqueeze(1).expand(-1, max_steps + 1, -1),
        t.unsqueeze(-1),
    )

    return torch.cat((interp_pos, interp_quat), dim=-1), timesteps


# 颜色名称到 RGB 浮点值的映射，用于渲染随机化和语言指令生成。
COLORS_DICT = {
    "red": [1.0, 0.0, 0.0],
    "green": [0.0, 1.0, 0.0],
    "blue": [0.0, 0.0, 1.0],
    "yellow": [1.0, 1.0, 0.0],
    "cyan": [0.0, 1.0, 1.0],
    "magenta": [1.0, 0.0, 1.0],
    "white": [1.0, 1.0, 1.0],
    "black": [0.0, 0.0, 0.0],
    "gray": [0.5, 0.5, 0.5],
    "light_gray": [0.75, 0.75, 0.75],
    "dark_gray": [0.25, 0.25, 0.25],
    "orange": [1.0, 0.65, 0.0],
    "purple": [0.5, 0.0, 0.5],
    "pink": [1.0, 0.75, 0.8],
    "brown": [0.65, 0.16, 0.16],
    "olive": [0.5, 0.5, 0.0],
    "teal": [0.0, 0.5, 0.5],
    "navy": [0.0, 0.0, 0.5],
    "maroon": [0.5, 0.0, 0.0],
    "lime": [0.75, 1.0, 0.0],
    "gold": [1.0, 0.84, 0.0],
    "silver": [0.75, 0.75, 0.75],
    "bronze": [0.8, 0.5, 0.2],
    "sky_blue": [0.53, 0.81, 0.92],
    "forest_green": [0.13, 0.55, 0.13],
    "violet": [0.93, 0.51, 0.93],
    "coral": [1.0, 0.5, 0.31],
    "salmon": [0.98, 0.5, 0.45],
    "turquoise": [0.25, 0.88, 0.82],
    "indigo": [0.29, 0.0, 0.51],
    "beige": [0.96, 0.96, 0.86],
    "ivory": [1.0, 1.0, 0.94],
}


def load_object_point_clouds(object_files: list[str], asset_root: str) -> list[np.ndarray]:
    """Load object point clouds inferred from URDF paths.

    旧数据集约定：`ObjDatasetName/urdf/name.urdf` 对应
    `ObjDatasetName/pointclouds/name.npy`。这里保留该路径推导规则。
    """

    point_clouds = []
    for object_file in object_files:
        path_parts = object_file.split("/")
        if len(path_parts) != 3:
            raise ValueError(f"Filename should be ObjDatasetName/urdf/name.urdf, got {object_file!r}.")
        point_cloud_file = os.path.join(path_parts[0], "pointclouds", path_parts[-1].replace(".urdf", ".npy"))
        full_path = os.path.join(asset_root, point_cloud_file)
        print(f"object file: {object_file} -> pcl file: {point_cloud_file}")
        point_clouds.append(np.load(full_path))
    return point_clouds


def transform_points(quat: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Rotate points by a batch of `wxyz` quaternions.

    Args:
        quat: 旋转四元数，形状可为 `[B, 4]`、`[B, 1, 4]` 或可广播形状。
        points: 点或点云，最后一维可以是 xyz，也可以是旧代码常见的 xyz0。

    Returns:
        旋转后的 xyz 点，形状与输入点的前三维一致。
    """

    xyz = points[..., :3]
    if quat.ndim == 2 and xyz.ndim == 3:
        quat = quat.unsqueeze(1)
    return quat_apply(quat.to(dtype=xyz.dtype, device=xyz.device), xyz)


def farthest_point_sample(
    xyz: torch.Tensor,
    npoint: int,
    device: str | torch.device | None = None,
    init: torch.Tensor | list[int] | None = None,
) -> torch.Tensor:
    """Farthest point sampling for batched point clouds.

    Args:
        xyz: 输入点云，形状 `[B, N, 3]`。
        npoint: 需要采样的点数。
        device: 输出索引所在设备；默认跟随 `xyz.device`。
        init: 可选初始点索引，形状可广播到 `[B]`。

    Returns:
        采样点索引，形状 `[B, npoint]`。
    """

    device = xyz.device if device is None else device
    batch_size, num_points, channels = xyz.size()
    centroids = torch.zeros(batch_size, npoint, dtype=torch.long, device=device)
    distance = torch.ones(batch_size, num_points, dtype=xyz.dtype, device=device) * 1.0e10

    if init is not None:
        farthest = torch.as_tensor(init, dtype=torch.long, device=device).reshape(batch_size)
    else:
        farthest = torch.randint(0, num_points, (batch_size,), dtype=torch.long, device=device)

    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)
    xyz = xyz.to(device)

    # 每轮选择当前“离已选点集合最远”的点，使采样点在物体表面分布更均匀。
    for sample_id in range(npoint):
        centroids[:, sample_id] = farthest
        centroid = xyz[batch_indices, farthest, :].view(batch_size, 1, channels)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.where(dist < distance, dist, distance)
        farthest = torch.max(distance, dim=-1).indices
    return centroids


def index_points(points: torch.Tensor, idx: torch.Tensor, device: str | torch.device | None = None) -> torch.Tensor:
    """Gather batched points/features by batched indices.

    Args:
        points: 输入点或点特征，形状 `[B, N, C]`。
        idx: 索引 tensor，常见形状为 `[B, S]`，也支持 `[B, S, K]`。
        device: 构造 batch 索引的设备；默认跟随 `points.device`。
    """

    device = points.device if device is None else device
    batch_size = points.size(0)
    idx = idx.to(device)
    view_shape = list(idx.size())
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.size())
    repeat_shape[0] = 1
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    return points.to(device)[batch_indices, idx, :]
