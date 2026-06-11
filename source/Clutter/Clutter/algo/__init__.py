"""Clutter 强化学习算法包入口。

用途：
- 统一导出 Clutter 内部算法模块，脚本侧可以写 `from Clutter.algo import ppo_onestep`。
- 避免训练脚本直接依赖旧项目的 `algo` 顶层包。
"""

# 当前迁移阶段只接入 one-step PPO；后续若有新算法也在这里继续导出。
from . import ppo_onestep

__all__ = ["ppo_onestep"]
