# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Clutter IsaacLab extension package.

导入 `Clutter` 包时只注册任务环境。模板 UI extension 会创建窗口和 UI 依赖，
命令行训练/播放脚本并不需要它；保持包入口轻量，可以减少 Gym 环境解析阶段
因为额外 UI 模块导致的启动/关闭问题。
"""

# Register Gym environments.
from .tasks import *
