"""one-step PPO 算法包入口。

用途：
- 暴露训练/推理脚本需要的三个核心对象：
  `ActorCritic` 策略网络、`PPO` runner、`RolloutStorage` 数据缓存。
- 把算法实现收敛到 `Clutter.algo.ppo_onestep` 命名空间中。
"""

# 策略网络：根据观测输出 planner action，并提供 critic value。
from .module import ActorCritic
# 训练器：负责采样 one-step rollout、计算 PPO loss、保存/加载模型。
from .ppo import PPO
# 缓存：保存一次 PPO 更新所需的观测、动作、奖励和旧策略统计量。
from .storage import RolloutStorage

__all__ = ["ActorCritic", "PPO", "RolloutStorage"]
