"""PointNet utility package migrated from DemoGrasp.

这个包主要服务于 `ppo_onestep.module.PointNetBackbone` 的视觉策略分支。
第一阶段保留 PointNet / ManiSkill-Learn 相关工具，暂不迁入依赖 `spconv`
和旧 `algo.ppo_utils` 的 sparse UNet 实现。
"""

__all__ = ["maniskill_learn"]
