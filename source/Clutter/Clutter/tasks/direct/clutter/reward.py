"""Pure PyTorch reward functions for the Clutter grasping task.

这个文件从旧 DemoGrasp 的 `tasks/reward.py` 迁移而来。奖励函数只依赖
PyTorch tensor，不直接调用 IsaacGym/IsaacLab API，因此可以被 IsaacLab 的
`ClutterEnv.compute_reward()` 直接复用。函数返回旧版约定的多项 buffer，
环境侧会把其中的 reward、success 统计和日志接回 IsaacLab。
"""

from __future__ import annotations

import torch


def reward_binary(
    reset_buf,
    progress_buf,
    successes,
    current_successes,
    has_hit_table,
    max_episode_length: float,
    table_heights,
    object_pos,
    palm_pos,
    fingertip_pos,
    num_fingers: int,
    object_init_states,
    **kwargs,
):
    """Compute the old DemoGrasp binary grasp success reward.

    奖励语义非常朴素：只有在 episode 结算时，如果物体被抬高且手掌/指尖确实
    靠近物体，就给 1，否则给 0。中间步骤主要输出诊断指标，便于观察策略是否
    在接近物体、抬升物体，以及是否碰到桌面。
    """

    # info 中的 tensor 会被环境写入 extras/log，用于训练曲线和调试。
    info = {}

    # 物体相对 reset 时的高度变化，是判断“是否抓起来”的核心量。
    object_delta_z = object_pos[:, 2] - object_init_states[:, 2]

    # 手掌中心到物体中心的距离；截断到 0.5 是为了让日志尺度稳定。
    palm_object_dist = torch.norm(object_pos - palm_pos, dim=-1)
    palm_object_dist = torch.where(palm_object_dist >= 0.5, 0.5, palm_object_dist)

    # 物体水平偏移只作为辅助指标，帮助观察物体是否被推远。
    horizontal_offset = torch.norm(object_pos[:, 0:2], dim=-1)

    # 累加每个指尖到物体中心的距离；旧奖励用这个总距离判断手是否接近物体。
    fingertips_object_dist = torch.zeros_like(object_delta_z)
    for finger_id in range(fingertip_pos.shape[-2]):
        fingertips_object_dist += torch.norm(fingertip_pos[:, finger_id, :] - object_pos, dim=-1)
    fingertips_object_dist = torch.where(fingertips_object_dist >= 3.0, 3.0, fingertips_object_dist)

    # 如果配置中没有找到指尖 body，不能让 0 个指尖距离误判为“已经接近”。
    has_fingertips = fingertip_pos.shape[-2] > 0 and num_fingers > 0
    if has_fingertips:
        fingertip_close = fingertips_object_dist <= 0.12 * num_fingers
        min_keypoint_z = torch.min(fingertip_pos[:, :, 2], dim=-1).values
    else:
        fingertip_close = torch.zeros_like(object_delta_z, dtype=torch.bool)
        fingertips_object_dist = torch.full_like(object_delta_z, 3.0)
        min_keypoint_z = palm_pos[:, 2]

    # 手掌足够近或者指尖整体足够近，都认为手已经进入可抓取区域。
    palm_close = palm_object_dist <= 0.15
    hand_approach_flag = torch.logical_or(fingertip_close, palm_close)

    # 只有手靠近后，抬升高度才被视为有效 lift；否则可能只是物体被误碰或弹起。
    lift_object = torch.zeros_like(object_delta_z)
    lift_object = torch.where(hand_approach_flag, object_delta_z, lift_object)

    # 旧版奖励自己也会根据 episode 长度生成 reset 标志；IsaacLab 侧会合并使用。
    resets = reset_buf.clone()
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    # 返回一个“旧版语义”的进度 buffer，主要用于兼容旧接口；IsaacLab 仍维护真实计数。
    progress_buf = torch.where(resets > 0, torch.zeros_like(progress_buf), progress_buf)

    # 成功条件：物体至少抬高 0.1 m，并且手掌或指尖确实靠近物体。
    successes = torch.where(
        object_delta_z > 0.1,
        torch.where(hand_approach_flag, torch.ones_like(successes), torch.zeros_like(successes)),
        torch.zeros_like(successes),
    )

    # current_successes 只在 reset 结算时更新，因此 reward 是 episode 末尾二值奖励。
    current_successes = torch.where(resets > 0, successes, current_successes)
    reward = current_successes.to(torch.float32)

    # 手掌或最低指尖低于桌面高度时，记录一次碰桌事件。
    min_keypoint_z = torch.min(min_keypoint_z, palm_pos[:, 2])
    has_hit_table = torch.where(
        min_keypoint_z < table_heights,
        torch.ones_like(has_hit_table, dtype=torch.bool),
        has_hit_table,
    )

    # 辅助指标全部按环境并行输出，环境侧会再聚合均值写进训练日志。
    info["fingertips_object_dist"] = fingertips_object_dist
    info["palm_object_dist"] = palm_object_dist
    info["lift_object"] = lift_object
    info["horizontal_offset"] = horizontal_offset
    info["reward"] = reward
    info["hand_approach_flag"] = hand_approach_flag

    return (
        reward,
        resets,
        progress_buf,
        successes,
        current_successes,
        has_hit_table,
        info,
    )


# 奖励注册表：cfg.reward_type == "binary" 时，环境会调用 reward_binary。
REWARD_DICT = {
    "binary": reward_binary,
}
