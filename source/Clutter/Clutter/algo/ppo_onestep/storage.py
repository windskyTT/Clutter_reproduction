"""one-step PPO 的 rollout 数据缓存。

用途：
- 保存 vectorized env 中每个并行环境采样到的一步 planner transition。
- 为 PPO 更新提供 observations/states/actions/rewards/old log prob/value 等张量。
- 保持接口接近旧 PPO storage，降低算法迁移成本。
"""

from __future__ import annotations

import torch
from torch.utils.data.sampler import BatchSampler, SequentialSampler, SubsetRandomSampler


class RolloutStorage:
    """Store a short vectorized rollout and expose mini-batches for PPO.

    DemoGrasp 的 one-step PPO 每次通常只采样一个 planner action，因此这个
    buffer 保持得很轻量：只保存当前观测、动作、奖励和 PPO 更新所需的旧策略信息。
    """

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        obs_shape: tuple[int, ...],
        states_shape: tuple[int, ...],
        actions_shape: tuple[int, ...],
        device: str = "cpu",
        sampler: str = "sequential",
    ):
        self.device = device
        self.sampler = sampler
        # num_transitions_per_env 通常为 1，对应 one-step planner；保留泛化写法便于调参。
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs
        # step 指向当前写入位置，clear() 后回到 0。
        self.step = 0

        # Core rollout tensors:
        # shape = [时间步, 并行环境数, 单环境维度...]。
        self.observations = torch.zeros(num_transitions_per_env, num_envs, *obs_shape, device=self.device)
        self.states = torch.zeros(num_transitions_per_env, num_envs, *states_shape, device=self.device)
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, dtype=torch.bool, device=self.device)

        # PPO 更新需要保存采样时的 value、log_prob、均值和标准差。
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.values = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=self.device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=self.device)

    def change_num_envs(self, new_num_envs: int) -> None:
        """当丢弃无效 reset 时，按有效环境数量重建 buffer。"""
        if self.step != 0:
            raise AssertionError("Can only change num_envs at the beginning of rollout")
        # 复用当前 shape/device/sampler，只改变并行环境数量。
        self.__init__(
            new_num_envs,
            self.num_transitions_per_env,
            tuple(self.observations.shape[2:]),
            tuple(self.states.shape[2:]),
            tuple(self.actions.shape[2:]),
            device=self.device,
            sampler=self.sampler,
        )

    def add_transitions(
        self,
        observations: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        values: torch.Tensor,
        actions_log_prob: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
    ) -> None:
        """Append one vectorized transition to the rollout buffer."""
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Rollout buffer overflow")

        # copy_ 保持 storage 张量地址稳定，避免反复分配 GPU 内存。
        self.observations[self.step].copy_(observations)
        self.states[self.step].copy_(states)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards.view(-1, 1))
        self.dones[self.step].copy_(dones.view(-1, 1))
        self.values[self.step].copy_(values)
        self.actions_log_prob[self.step].copy_(actions_log_prob.view(-1, 1))
        self.mu[self.step].copy_(mu)
        self.sigma[self.step].copy_(sigma)
        self.step += 1

    def clear(self) -> None:
        """只重置写入指针；底层张量复用，减少训练过程中的内存抖动。"""
        self.step = 0

    def compute_returns(self, last_values=None, gamma=None, lam=None) -> None:
        """Compute one-step returns and normalized advantages.

        这里保持 DemoGrasp 的 one-step 行为：回报就是当前奖励，优势是奖励减去
        critic value。gamma/lam 参数保留给接口兼容。
        """
        self.returns = self.rewards.clone()
        self.advantages = self.rewards - self.values
        self.advantages = (self.advantages - self.advantages.mean()) / (self.advantages.std() + 1e-8)

    def get_statistics(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mean trajectory length and mean reward for logging."""
        # one-step PPO 会把最后一个时间步视作 done，便于复用通用 episode 统计逻辑。
        done = self.dones.cpu().clone()
        done[-1] = True
        flat_dones = done.permute(1, 0, 2).reshape(-1, 1)
        done_indices = torch.cat((flat_dones.new_tensor([-1], dtype=torch.int64), flat_dones.nonzero()[:, 0]))
        trajectory_lengths = done_indices[1:] - done_indices[:-1]
        return trajectory_lengths.float().mean(), self.rewards.mean()

    def mini_batch_generator(self, num_mini_batches: int):
        """Yield flattened transition indices for PPO mini-batches."""
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = max(1, batch_size // max(1, num_mini_batches))

        # sequential 适合大规模向量环境，减少随机采样 CPU 开销；random 保留给需要打乱的实验。
        if self.sampler == "sequential":
            sampler = SequentialSampler(range(batch_size))
        elif self.sampler == "random":
            sampler = SubsetRandomSampler(range(batch_size))
        else:
            raise ValueError(f"Unsupported sampler: {self.sampler}")

        return BatchSampler(sampler, mini_batch_size, drop_last=False)
