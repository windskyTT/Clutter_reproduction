"""Clutter one-step PPO 默认配置。

用途：
- 替代旧 Hydra `tasks/train/PPOOneStep.yaml` 的最小 Python 配置入口。
- 训练脚本和播放脚本都通过 `default_ppo_onestep_cfg()` 获取配置。
- Gymnasium 注册表通过 `PPOOneStepCfg()` 获取默认 agent 配置。
- 返回 OmegaConf 对象，使 `cfg.key` 和 `cfg.get("key")` 两种访问方式都可用。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from omegaconf import OmegaConf


DEFAULT_PPO_ONESTEP_CFG: dict[str, Any] = {
    # 算法名保留 DemoGrasp 的语义，入口据此选择 Clutter.algo.ppo_onestep。
    "name": "ppo_onestep",
    # 保留旧 PPOOneStep.yaml 的默认日志目录；训练入口会按 Clutter 路径体系覆盖到 logs/。
    "log_dir": "./runs_ppo",
    # False 表示只使用状态观测；True 时会尝试启用 PointNet 点云 backbone。
    "is_vision": False,
    "policy": {
        # 当前默认使用 MLP；PointNet 相关字段保留给后续视觉观测迁移。
        "backbone_type": "pn",
        # freeze_backbone=True 表示点云特征网络不参与梯度更新。
        "freeze_backbone": False,
        # actor/critic 隐藏层尺寸，沿用旧 one-step PPO 的大 MLP 设置。
        "pi_hid_sizes": [1024, 1024, 512, 512],
        "vf_hid_sizes": [1024, 1024, 512, 512],
        "activation": "elu",
        # pc_shape/pc_emb_dim 仅在 is_vision=True 时生效。
        "pc_shape": [512, 3],
        "pc_emb_dim": 128,
    },
    # test=True 进入播放/推理分支；False 进入训练分支。
    "test": False,
    "resume": 0,
    # checkpoint 保存间隔，单位是 PPO iteration。
    "save_interval": 100,
    "print_log": True,
    # 最大 PPO 迭代次数，训练脚本可用 --max_iterations 覆盖。
    "max_iterations": 20000,
    # PPO clip 范围，限制新旧策略概率比变化。
    "cliprange": 0.2,
    # 熵正则系数；0 表示不额外鼓励随机探索。
    "ent_coef": 0.0,
    # one-step planner 默认每个 env 每轮只收集 1 个 transition。
    "nsteps": 1,
    "noptepochs": 5,
    "nminibatches": 4,
    # 梯度裁剪阈值，防止偶发大梯度破坏训练。
    "max_grad_norm": 1.0,
    "optim_stepsize": 3.0e-4,
    # adaptive 会根据 KL 调整学习率。
    "schedule": "adaptive",
    "desired_kl": 0.016,
    # gamma/lam 保留 PPO 接口语义；当前 one-step returns 中影响较小。
    "gamma": 0.96,
    "lam": 0.95,
    # 初始动作分布标准差，控制训练初期探索幅度。
    "init_noise_std": 0.8,
    "surrogate_loss_coef": 1.0,
    "value_loss_coef": 2.0,
    # 若环境提供 is_init_state_valid，可丢弃无效 reset 样本。
    "discard_invalid_resets": False,
    # 播放模式默认评估轮数。
    "times_testing_all_objects": 10,
}


def default_ppo_onestep_cfg(test: bool = False, overrides: dict[str, Any] | None = None):
    """构造 one-step PPO 配置。

    Args:
        test: 是否进入播放/推理模式。
        overrides: 调用方临时覆盖的配置字段，例如播放脚本覆盖评估轮数。
    """
    cfg = OmegaConf.create(deepcopy(DEFAULT_PPO_ONESTEP_CFG))
    cfg.test = bool(test)
    if overrides:
        cfg = OmegaConf.merge(cfg, overrides)
    return cfg


def PPOOneStepCfg():
    """Gymnasium registry entry point for the migrated one-step PPO config.

    IsaacLab's `load_cfg_from_registry()` 会导入这个可调用对象并再次调用它，
    因此这里返回和训练脚本一致的默认 OmegaConf 配置即可。
    """
    return default_ppo_onestep_cfg()
