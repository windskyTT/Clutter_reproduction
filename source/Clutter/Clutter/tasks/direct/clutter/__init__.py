# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Gymnasium registration for the Clutter DirectRLEnv task.

This file replaces the old IsaacGym task-map registration style. IsaacLab
discovers environments through Gymnasium. Importing this package is
therefore enough to register `Clutter-Grasp-Direct-v0` and make it available to
`gymnasium.make(...)` and `isaaclab_tasks.utils.parse_env_cfg(...)`.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


# 主环境注册名：训练/播放脚本默认使用这个 id 创建 Clutter 抓取环境。
gym.register(
    id="Clutter-Grasp-Direct-v0",
    entry_point=f"{__name__}.clutter_env:ClutterEnv",
    disable_env_checker=True,
    kwargs={
        # IsaacLab 通过这些 entry point 延迟加载环境配置和 RL 配置。
        "env_cfg_entry_point": f"{__name__}.clutter_env_cfg:ClutterEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
        # 自定义 one-step PPO 配置入口，供迁移后的脚本读取。
        "ppo_onestep_cfg_entry_point": f"{agents.__name__}.ppo_onestep_cfg:PPOOneStepCfg",
    },
)
