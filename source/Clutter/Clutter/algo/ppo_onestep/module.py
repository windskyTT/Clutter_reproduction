"""one-step PPO 使用的 Actor-Critic 网络。

用途：
- Actor 根据环境观测输出归一化 planner action。
- Critic 估计当前观测/state 的 value，用于 PPO advantage 和 value loss。
- 可选支持点云 backbone；默认迁移路径先使用纯状态 MLP。
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Independent, Normal


def get_activation(act_name: str) -> nn.Module:
    """根据配置名创建 MLP 激活层。"""
    if act_name == "elu":
        return nn.ELU()
    if act_name == "selu":
        return nn.SELU()
    if act_name in {"relu", "crelu"}:
        return nn.ReLU()
    if act_name == "lrelu":
        return nn.LeakyReLU()
    if act_name == "tanh":
        return nn.Tanh()
    if act_name == "sigmoid":
        return nn.Sigmoid()
    raise ValueError(f"Invalid activation function: {act_name}")


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """兼容 dict 和 OmegaConf 两种配置对象的读取方式。"""
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _atanh(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Numerically stable inverse tanh for already-squashed actions."""
    # PPO evaluate 阶段拿到的是 tanh 后的动作，需要反解回 Gaussian 的 pre-tanh 空间。
    x = torch.clamp(x, -1.0 + eps, 1.0 - eps)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def _tanh_squash_and_log_prob(
    dist_base: Independent,
    pre_tanh: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Squash Gaussian samples to [-1, 1] and correct their log probability."""
    actions = torch.tanh(pre_tanh)
    # tanh 会改变概率密度，因此 log_prob 需要加上变量替换的 Jacobian 修正。
    log_det_jacob = torch.sum(torch.log(1.0 - actions.pow(2) + eps), dim=-1)
    log_prob = dist_base.log_prob(pre_tanh) - log_det_jacob
    return actions, log_prob


class PointNetBackbone(nn.Module):
    """Lazy PointNet wrapper used only when `is_vision=True`.

    Clutter 当前迁移第 1 步主要跑状态观测；点云网络如果还没有迁入，会在真正
    启用视觉策略时给出明确错误，而不会阻塞普通 MLP policy 的导入。
    """

    def __init__(self, pc_dim: int, feature_dim: int):
        super().__init__()
        try:
            # 延迟导入 PointNet，避免纯状态策略在 pn_utils 尚未迁完时无法导入。
            from ..pn_utils.maniskill_learn.networks.backbones.pointnet import getPointNet
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Vision policy requires Clutter.algo.pn_utils to be migrated before use."
            ) from exc
        self.backbone = getPointNet({"input_feature_dim": pc_dim, "feat_dim": feature_dim})

    def forward(self, input_pc: torch.Tensor) -> torch.Tensor:
        return self.backbone(input_pc)


class ActorCritic(nn.Module):
    """Gaussian actor with a value critic for PPO.

    Actor 输出的是 planner action 的 pre-tanh 均值；采样后用 tanh 压到 [-1, 1]，
    这和 DemoGrasp 中动作作为归一化目标位姿/手部自由度的语义保持一致。
    """

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        states_shape: tuple[int, ...],
        actions_shape: tuple[int, ...],
        initial_std: float,
        model_cfg,
        asymmetric: bool = False,
        use_pcl: bool = False,
    ):
        super().__init__()
        # asymmetric=True 时 critic 可以使用 state_space，否则 actor/critic 共用 policy observation。
        self.asymmetric = asymmetric
        self.use_pcl = use_pcl
        self.backbone_type = _cfg_get(model_cfg, "backbone_type", "pn")
        self.freeze_backbone = bool(_cfg_get(model_cfg, "freeze_backbone", False))

        # policy 配置来自 ppo_onestep_cfg.py，保留旧训练超参数的网络宽度。
        actor_hidden_dim = list(_cfg_get(model_cfg, "pi_hid_sizes", [256, 256, 256]))
        critic_hidden_dim = list(_cfg_get(model_cfg, "vf_hid_sizes", [256, 256, 256]))
        activation = get_activation(_cfg_get(model_cfg, "activation", "elu"))

        self.num_obs = int(np.prod(obs_shape))
        self.act_dim = int(np.prod(actions_shape))

        if self.use_pcl:
            # 点云观测拼在状态观测末尾，先通过 PointNet 压成 pc_emb_dim，再送入 MLP。
            self.pc_shape = list(_cfg_get(model_cfg, "pc_shape", [512, 3]))
            self.pc_emb_dim = int(_cfg_get(model_cfg, "pc_emb_dim", 128))
            if self.backbone_type != "pn":
                raise ValueError(f"Invalid backbone type: {self.backbone_type}")
            self.backbone = PointNetBackbone(pc_dim=self.pc_shape[-1], feature_dim=self.pc_emb_dim)
        else:
            # 纯状态观测时没有点云部分，MLP 输入维度就是 obs_shape 展平后的长度。
            self.pc_shape = [0, 0]
            self.pc_emb_dim = 0
            self.backbone = None

        # 使用点云时，用 pc_emb_dim 替换原始 N*D 点云维度。
        self.num_state_based_obs = self.num_obs - math.prod(self.pc_shape) + self.pc_emb_dim
        self.pc_start_idx = self.num_obs - math.prod(self.pc_shape)

        # Actor 输出 planner action 的 pre-tanh mean；Critic 输出标量 value。
        self.actor_mean = self._build_mlp(self.num_state_based_obs, actor_hidden_dim, self.act_dim, activation)
        critic_input_dim = int(np.prod(states_shape)) if self.asymmetric else self.num_state_based_obs
        self.critic = self._build_mlp(critic_input_dim, critic_hidden_dim, 1, activation)

        # log_std 是可训练参数，表示对角 Gaussian 每个动作维度的探索噪声。
        init_log_std = float(np.log(initial_std))
        self.log_std = nn.Parameter(torch.full((self.act_dim,), init_log_std))
        # 正交初始化沿用 PPO 常见做法，末层 actor gain 小一些可避免初始动作过激。
        self._init_orthogonal(self.actor_mean, [np.sqrt(2)] * len(actor_hidden_dim) + [0.01])
        self._init_orthogonal(self.critic, [np.sqrt(2)] * len(critic_hidden_dim) + [1.0])

    @staticmethod
    def _build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int, activation: nn.Module) -> nn.Sequential:
        """构建多层感知机，供 actor 和 critic 共用。"""
        layers: list[nn.Module] = []
        last_dim = input_dim
        for hidden_dim in hidden_dims:
            layers += [nn.Linear(last_dim, hidden_dim), activation]
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, output_dim))
        return nn.Sequential(*layers)

    @staticmethod
    def _init_orthogonal(sequential: nn.Sequential, gains: list[float]) -> None:
        """对 Linear 层做正交初始化，提升 PPO 训练稳定性。"""
        idx = 0
        for module in sequential:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=gains[min(idx, len(gains) - 1)])
                nn.init.zeros_(module.bias)
                idx += 1

    def _encode_observations(self, observations: torch.Tensor) -> torch.Tensor:
        """Replace raw point cloud observations with PointNet features when enabled."""
        observations = observations.reshape(observations.shape[0], -1)
        if not self.use_pcl:
            return observations

        # freeze_backbone=True 时不更新点云 backbone，只训练后续 actor/critic MLP。
        if self.freeze_backbone:
            with torch.no_grad():
                pc = observations[:, self.pc_start_idx :].reshape(-1, *self.pc_shape)
                pc_feature = self.backbone(pc).reshape(-1, self.pc_emb_dim)
        else:
            pc = observations[:, self.pc_start_idx :].reshape(-1, *self.pc_shape)
            pc_feature = self.backbone(pc).reshape(-1, self.pc_emb_dim)
        return torch.cat([observations[:, : self.pc_start_idx], pc_feature], dim=1)

    def forward(self, observations: torch.Tensor, states: torch.Tensor | None = None, inference: bool = False):
        """Sample an action during training or return deterministic action during play."""
        observations = self._encode_observations(observations)
        mean = self.actor_mean(observations)
        std = self.log_std.exp()
        base = Independent(Normal(mean, std), 1)

        if inference:
            # 推理/播放时使用均值动作，减少随机性，便于评估 checkpoint。
            return torch.tanh(mean).detach()

        # 训练时从 Gaussian 采样，并保存 log_prob/value 给 PPO loss 使用。
        pre_tanh = base.rsample()
        actions, log_prob = _tanh_squash_and_log_prob(base, pre_tanh)
        critic_input = states.reshape(states.shape[0], -1) if self.asymmetric and states is not None else observations
        value = self.critic(critic_input)

        return (
            actions.detach(),
            log_prob.detach(),
            value.detach(),
            torch.tanh(mean).detach(),
            self.log_std.expand(mean.shape[0], -1).detach(),
        )

    def evaluate(self, observations: torch.Tensor, states: torch.Tensor | None, actions: torch.Tensor, eps: float = 1e-6):
        """Evaluate sampled actions under the current policy for PPO loss."""
        observations = self._encode_observations(observations)
        mean = self.actor_mean(observations)
        std = self.log_std.exp()
        base = Independent(Normal(mean, std), 1)

        actions = actions.reshape(actions.shape[0], -1)
        # 已保存的 actions 是 tanh 后的 [-1, 1] 动作，evaluate 时要回到 pre-tanh。
        pre_tanh = _atanh(actions, eps=eps)
        log_det_jacob = torch.sum(torch.log(1.0 - actions.pow(2) + eps), dim=-1)
        log_prob = base.log_prob(pre_tanh) - log_det_jacob
        entropy = base.entropy()

        # asymmetric critic 使用单独 state，否则用编码后的 policy observation。
        critic_input = states.reshape(states.shape[0], -1) if self.asymmetric and states is not None else observations
        value = self.critic(critic_input)

        return log_prob, entropy, value, torch.tanh(mean), self.log_std.expand(mean.shape[0], -1)
