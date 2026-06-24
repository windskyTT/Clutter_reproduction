"""适配 IsaacLab DirectRLEnv 的 one-step PPO runner。

用途：
- 替代旧训练入口中直接依赖 IsaacGym VecTask 的 PPO 调用方式。
- 将 IsaacLab 的 `reset()` / `step()` / `obs_dict["policy"]` 接口转换成 one-step PPO 需要的数据。
- 同时兼容已迁移完成的抓取环境 hook：`generate_reaching_plan_idx()` 和 `compute_reference_actions()`。

核心流程：
1. reset 所有并行环境，取得 policy observation。
2. Actor 采样 planner action。
3. 若环境提供抓取规划 hook，则先生成 reaching plan，再按参考动作 replay 完一段 episode。
4. 用成功率/环境奖励作为 one-step reward，写入 RolloutStorage。
5. 对采样到的一步数据执行 PPO 更新。
"""

from __future__ import annotations

import os
import statistics
import time
from collections import deque # 导入双端队列，用于构建固定长度的滑动窗口
from typing import Any # 导入泛型类型 Any，用于类型注解

import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.spaces import Dict as DictSpace
from gymnasium.spaces import Space

try:
    # 训练需要 TensorBoard；播放 checkpoint 时不需要，因此这里做可选导入。
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:  # TensorBoard is only required for training, not import/play.
    SummaryWriter = None

from .storage import RolloutStorage


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read config values from either dict or OmegaConf containers."""
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _numel(shape: tuple[int, ...]) -> int:
    """计算展平后的张量元素数量。"""
    value = 1
    for item in shape:
        value *= int(item)
    return value


class PPO:
    """One-step PPO planner.

    旧 DemoGrasp 代码直接调用 IsaacGym VecTask 的 `reset_idx/step/get_state`。
    这里把这些访问集中封装，使同一个 runner 能接受 IsaacLab 的
    `reset()` 和五元组 `step()`，也能在真正的 grasp 环境迁好后调用
    `generate_reaching_plan_idx()` 和 `compute_reference_actions()`。
    """

    def __init__(self, vec_env, actor_critic_class, train_param, log_dir="run", apply_reset=False, action_dim=6):
        # Gymnasium wrapper 可能隐藏自定义环境方法；runner 内部只使用 unwrapped env。
        self.vec_env = vec_env.unwrapped if hasattr(vec_env, "unwrapped") else vec_env
        self.device = torch.device(getattr(self.vec_env, "device", "cpu"))
        self.num_envs = int(getattr(self.vec_env, "num_envs"))

        # PPO hyper-parameters：字段名保持旧配置命名，减少迁移时配置转换工作。
        self.clip_param = _cfg_get(train_param, "cliprange")
        self.num_learning_epochs = _cfg_get(train_param, "noptepochs")
        self.num_mini_batches = _cfg_get(train_param, "nminibatches")
        self.num_learning_iterations = _cfg_get(train_param, "max_iterations")
        self.num_transitions_per_env = _cfg_get(train_param, "nsteps")
        self.value_loss_coef = _cfg_get(train_param, "value_loss_coef", 2.0)
        self.entropy_coef = _cfg_get(train_param, "ent_coef")
        self.gamma = _cfg_get(train_param, "gamma")
        self.lam = _cfg_get(train_param, "lam")
        self.max_grad_norm = _cfg_get(train_param, "max_grad_norm", 2.0)
        self.use_clipped_value_loss = _cfg_get(train_param, "use_clipped_value_loss", False)
        self.init_noise_std = _cfg_get(train_param, "init_noise_std", 0.3)
        self.discard_invalid_resets = _cfg_get(train_param, "discard_invalid_resets", False)
        self.model_cfg = _cfg_get(train_param, "policy")
        self.sampler = _cfg_get(train_param, "sampler", "sequential")
        self.is_vision = _cfg_get(train_param, "is_vision", False)
        self.desired_kl = _cfg_get(train_param, "desired_kl", None)
        self.schedule = _cfg_get(train_param, "schedule", "fixed")
        self.step_size = _cfg_get(train_param, "optim_stepsize")

        # 测试相关参数：runner.run() 会根据 train_param.test 选择训练或播放分支。
        self.is_testing_all_objects = _cfg_get(train_param, "is_testing_all_objects", False)
        self.times_testing_all_objects = _cfg_get(train_param, "times_testing_all_objects", 10)
        self.plan = _cfg_get(train_param, "plan", False)
        self.sleep_per_step = float(_cfg_get(train_param, "sleep_per_step", 0.0))
        self.hold_after_round = float(_cfg_get(train_param, "hold_after_round", 0.0))

        # IsaacLab observation_space 可能是 Dict({"policy": ...})，这里抽出 policy branch。
        self.observation_space = self._policy_space(getattr(self.vec_env, "observation_space", None))
        self.state_space = self._policy_space(getattr(self.vec_env, "state_space", None))
        self.action_space = self._policy_space(getattr(self.vec_env, "action_space", None))

        # storage 和网络只需要单环境 shape；向量化维度 num_envs 会单独处理。
        self.obs_shape = self._space_shape(self.observation_space)
        self.state_shape = self._space_shape(self.state_space, default=(0,))
        # action_dim 是 planner action 维度，不一定等于底层物理控制 action 维度。
        self.action_shape = (int(action_dim),)
        self.asymmetric = _numel(self.state_shape) > 0

        # actor_critic_class 作为参数传入，方便后续替换网络结构或做 ablation。
        self.actor_critic = actor_critic_class(
            self.obs_shape,
            self.state_shape,
            self.action_shape,
            self.init_noise_std,
            self.model_cfg,
            asymmetric=self.asymmetric,
            use_pcl=self.is_vision,
        ).to(self.device)

        # RolloutStorage 保存一批 one-step transition，再交给 update() 做 PPO 优化。
        self.storage = RolloutStorage(
            self.num_envs,
            self.num_transitions_per_env,
            self.obs_shape,
            self.state_shape,
            self.action_shape,
            self.device,
            self.sampler,
        )
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.step_size)
        print(self.actor_critic)

        # 训练日志和 checkpoint 控制项。
        self.save_interval = _cfg_get(train_param, "save_interval")
        self.log_dir = log_dir
        self.print_log = _cfg_get(train_param, "print_log")
        self.tot_timesteps = 0
        self.tot_time = 0
        self.is_testing = _cfg_get(train_param, "test")
        self.debug_reference_actions = bool(_cfg_get(train_param, "debug_reference_actions", self.is_testing))
        self.current_learning_iteration = 0
        if not self.is_testing:
            if SummaryWriter is None:
                raise ModuleNotFoundError("Training requires tensorboard. Install it or run play mode.")
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

        self.apply_reset = apply_reset
        # 常用的“所有环境 id”张量，避免训练循环里反复创建。
        self._all_env_ids = torch.arange(self.num_envs, device=self.device)

    def _policy_space(self, space: Space | None) -> Space | None:
        """Pick the `policy` branch from IsaacLab Dict observation spaces."""
        if isinstance(space, DictSpace):
            # DirectRLEnv 默认 observation dict 使用 "policy" 作为 RL policy 输入。
            if "policy" in space.spaces:
                return space.spaces["policy"]
            # 兼容旧环境/旧 wrapper 中常见的 "obs" key。
            if "obs" in space.spaces:
                return space.spaces["obs"]
        return space

    def _space_shape(self, space: Space | int | None, default: tuple[int, ...] | None = None) -> tuple[int, ...]:
        """Convert Gym space shape to a single-env tensor shape."""
        if default is None:
            default = (0,)
        if space is None:
            return default
        if isinstance(space, int):
            return (space,) if space > 0 else default
        if not isinstance(space, Space):
            return tuple(getattr(space, "shape", default))

        shape = tuple(space.shape)
        # IsaacLab vector envs may expose `(num_envs, dim)`; storage wants only `dim`.
        if len(shape) >= 2 and shape[0] == self.num_envs:
            shape = shape[1:]
        return shape if shape else default

    def _unpack_reset(self, reset_result):
        """Handle both Gymnasium `(obs, info)` and IsaacLab `obs` returns."""
        # 标准 Gymnasium reset 返回 (obs, info)；部分 IsaacLab 路径只返回 obs。
        if isinstance(reset_result, tuple):
            return reset_result[0]
        return reset_result

    def _policy_obs(self, obs_dict_or_tensor) -> torch.Tensor:
        """Extract the policy observation tensor from IsaacLab/DemoGrasp outputs."""
        obs = self._unpack_reset(obs_dict_or_tensor)
        if isinstance(obs, dict):
            if "policy" in obs:
                obs = obs["policy"]
            elif "obs" in obs:
                obs = obs["obs"]
            else:
                # 兜底逻辑只用于调试型环境；正式环境应明确提供 policy/obs。
                tensor_values = [value for value in obs.values() if isinstance(value, torch.Tensor)]
                if not tensor_values:
                    raise KeyError("Observation dict must contain `policy`, `obs`, or a tensor value.")
                obs = tensor_values[0]
        return obs.to(self.device).reshape(self.num_envs, -1)

    def _reset_all(self) -> torch.Tensor:
        """Reset all environments using the API available on the current env."""
        if hasattr(self.vec_env, "reset_idx"):
            # 抓取环境可能保留按 env_ids reset 的接口。
            reset_result = self.vec_env.reset_idx(self._all_env_ids)
        else:
            # 原生 DirectRLEnv 使用 reset()。
            reset_result = self.vec_env.reset()
        return self._policy_obs(reset_result)

    def _get_state(self) -> torch.Tensor:
        """Return asymmetric critic state, or an empty tensor when unavailable."""
        if hasattr(self.vec_env, "get_state"):
            state = self.vec_env.get_state()
            if isinstance(state, dict):
                # 优先使用 critic state；没有时退回 policy/obs。
                state = state.get("critic", state.get("policy", state.get("obs")))
            if state is not None:
                return state.to(self.device).reshape(self.num_envs, -1)
        return torch.zeros(self.num_envs, _numel(self.state_shape), device=self.device)

    def _parse_step(self, step_result):
        """Normalize IsaacLab's five-return step and old four-return step."""
        if len(step_result) == 5:
            # IsaacLab/Gymnasium: obs, reward, terminated, truncated, extras。
            obs, reward, terminated, truncated, extras = step_result
            done = torch.logical_or(terminated, truncated)
        elif len(step_result) == 4:
            # 旧 VecTask 风格: obs, reward, done, extras。
            obs, reward, done, extras = step_result
        else:
            raise RuntimeError(f"Unexpected env.step return length: {len(step_result)}")
        return self._policy_obs(obs), reward.to(self.device), done.to(self.device), extras

    def _has_grasp_plan_api(self) -> bool:
        """Check whether the migrated grasp env exposes DemoGrasp planner hooks."""
        # 有这两个方法时，runner 使用“策略输出 plan，再由环境 replay reference action”的流程。
        return hasattr(self.vec_env, "generate_reaching_plan_idx") and hasattr(self.vec_env, "compute_reference_actions")

    def _success_reward(self, fallback_reward: torch.Tensor) -> torch.Tensor:
        """Use grasp success tensors when present; otherwise fall back to env reward."""
        successes = getattr(self.vec_env, "successes", None)
        if successes is None:
            successes = getattr(self.vec_env, "current_successes", None)
        if successes is None:
            # 模板环境没有 success tensor，就用 step() 返回的 reward 作为训练信号。
            return fallback_reward.reshape(self.num_envs)

        successes = successes.to(self.device).reshape(self.num_envs)
        has_hit_table = getattr(self.vec_env, "has_hit_table", None)
        if has_hit_table is not None:
            # 抓取过程中碰桌视为失败，保持旧任务的 reward 过滤语义。
            successes = torch.where(has_hit_table.to(self.device).bool(), torch.zeros_like(successes), successes)
        return successes

    def _valid_reset_mask(self) -> torch.Tensor:
        """Return valid reset mask for datasets that reject bad initial states."""
        valid = getattr(self.vec_env, "is_init_state_valid", None)
        if valid is None:
            return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        return valid.to(self.device).bool()

    def _execute_plan_or_native_step(self, actions: torch.Tensor):
        """Run either DemoGrasp reference replay or a plain IsaacLab env step."""
        if not self._has_grasp_plan_api():
            # 迁移早期可直接对普通 DirectRLEnv step 一次，验证 runner/网络/存储逻辑。
            obs, reward, done, extras = self._parse_step(self.vec_env.step(actions))
            one_step_done = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            return obs, reward.reshape(self.num_envs), one_step_done, extras

        # 抓取环境路径：policy 只负责生成规划参数，实际低层控制由 reference action 执行。
        self.vec_env.generate_reaching_plan_idx(self._all_env_ids, actions=actions)
        max_episode_length = int(getattr(self.vec_env, "max_episode_length", 1))
        start_eef = getattr(self.vec_env, "eef_pos", None)
        robot = getattr(self.vec_env, "robot", None)
        robot_data = getattr(robot, "data", None)
        start_joint = getattr(robot_data, "joint_pos", None)
        if start_eef is not None:
            start_eef = start_eef.clone()
        if start_joint is not None:
            start_joint = start_joint.clone()
        last_obs = None
        last_reward = torch.zeros(self.num_envs, device=self.device)
        last_done = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        last_extras = {}

        for step_id in range(max_episode_length):
            env_action = self.vec_env.compute_reference_actions()
            if self.debug_reference_actions and step_id in (0, 1, 5, 10, 20, 40):
                print(
                    f"[DEBUG] replay step={step_id}, action_abs_mean={env_action.abs().mean().item():.4f}, "
                    f"action_min={env_action.min().item():.4f}, action_max={env_action.max().item():.4f}",
                    flush=True,
                )
            last_obs, last_reward, last_done, last_extras = self._parse_step(self.vec_env.step(env_action))
            if self.debug_reference_actions and step_id in (0, 1, 5, 10, 20, 40):
                eef_move = -1.0
                joint_move = -1.0
                if start_eef is not None and hasattr(self.vec_env, "eef_pos"):
                    eef_move = (self.vec_env.eef_pos - start_eef).norm(dim=-1).mean().item()
                if start_joint is not None and hasattr(self.vec_env, "robot"):
                    joint_move = (self.vec_env.robot.data.joint_pos - start_joint).abs().mean().item()
                print(
                    f"[DEBUG] replay state step={step_id}, eef_move={eef_move:.5f}, joint_move={joint_move:.5f}",
                    flush=True,
                )
            if self.sleep_per_step > 0:
                time.sleep(self.sleep_per_step)
            # DemoGrasp 在接近 episode 末尾统计是否抓取成功。
            if step_id >= max_episode_length - 2:
                break

        one_step_reward = self._success_reward(last_reward)
        one_step_done = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        return last_obs, one_step_reward, one_step_done, last_extras

    def load(self, path, is_testing: bool = False) -> None:
        """Load actor-critic weights from a checkpoint."""
        self.actor_critic.load_state_dict(torch.load(path, map_location=self.device))
        self.actor_critic.eval() if is_testing else self.actor_critic.train()
        self.current_learning_iteration = 0

    def save(self, path) -> None:
        """Save actor-critic weights."""
        torch.save(self.actor_critic.state_dict(), path)

    def test(self, ckpt_path) -> None:
        """加载 checkpoint 并切换到 eval 模式。"""
        self.load(ckpt_path, is_testing=True)

    def run(self) -> None:
        """根据配置选择训练或播放模式。"""
        if self.is_testing:
            self._run_eval()
        else:
            self._run_train()

    def _run_eval(self) -> None:
        """Play the deterministic policy for a fixed number of one-step episodes."""
        self.actor_critic.eval()
        with torch.inference_mode():
            for round_idx in range(int(self.times_testing_all_objects)):
                current_obs = self._reset_all()
                current_states = self._get_state()
                # inference=True 使用确定性均值动作，方便评估策略质量。
                actions = self.actor_critic(current_obs, current_states, inference=True)
                _, rewards, _, _ = self._execute_plan_or_native_step(actions)
                print(f"Round {round_idx}: mean one-step reward/success = {rewards.float().mean().item():.4f}")
                if self.hold_after_round > 0:
                    time.sleep(self.hold_after_round)

    def _run_train(self) -> None:
        """Collect one-step rollouts and update the policy."""
        rewbuffer = deque(maxlen=self.num_envs)
        lenbuffer = deque(maxlen=self.num_envs)
        cur_reward_sum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        for it in range(self.current_learning_iteration, int(self.num_learning_iterations)):
            start = time.time()
            ep_infos = []
            reward_sum = []
            episode_length = []

            if self.discard_invalid_resets and self.num_transitions_per_env != 1:
                raise AssertionError("discard_invalid_resets currently requires nsteps=1")

            for _ in range(int(self.num_transitions_per_env)):
                # one-step PPO 每次 rollout 前重新 reset，策略针对初始状态生成 planner action。
                current_obs = self._reset_all()
                current_states = self._get_state()
                valid_mask = self._valid_reset_mask()
                if self.discard_invalid_resets:
                    valid_count = int(valid_mask.sum().item())
                    if valid_count == 0:
                        raise RuntimeError("All reset states were invalid; cannot update PPO.")
                    self.storage.change_num_envs(valid_count)

                # Actor 采样动作，同时返回 PPO 更新所需的旧 log_prob/value/均值/方差。
                actions, actions_log_prob, values, mu, sigma = self.actor_critic(current_obs, current_states)
                _, rewards, dones, _ = self._execute_plan_or_native_step(actions)

                if self.discard_invalid_resets:
                    # 只保留 reset 有效的环境，避免坏初始状态污染 PPO 更新。
                    self.storage.add_transitions(
                        current_obs[valid_mask],
                        current_states[valid_mask],
                        actions[valid_mask],
                        rewards[valid_mask],
                        dones[valid_mask],
                        values[valid_mask],
                        actions_log_prob[valid_mask],
                        mu[valid_mask],
                        sigma[valid_mask],
                    )
                else:
                    self.storage.add_transitions(
                        current_obs, current_states, actions, rewards, dones, values, actions_log_prob, mu, sigma
                    )

                if self.print_log:
                    # 记录短 episode reward/length，用于 TensorBoard 和终端日志。
                    cur_reward_sum += rewards
                    cur_episode_length += 1
                    new_ids = dones.nonzero(as_tuple=False).squeeze(-1)
                    reward_sum.extend(cur_reward_sum[new_ids].cpu().numpy().tolist())
                    episode_length.extend(cur_episode_length[new_ids].cpu().numpy().tolist())
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0

            if self.print_log:
                rewbuffer.extend(reward_sum)
                lenbuffer.extend(episode_length)

            collection_time = time.time() - start
            mean_trajectory_length, mean_reward = self.storage.get_statistics()

            start = time.time()
            # one-step returns/advantages 在 storage 内计算；这里保留 gamma/lam 参数兼容接口。
            self.storage.compute_returns(None, self.gamma, self.lam)
            mean_value_loss, mean_surrogate_loss = self.update()
            self.storage.clear()
            learn_time = time.time() - start

            current_success_rate = float(mean_reward.item())
            current_hit_table_rate = 0.0
            if hasattr(self.vec_env, "successes"):
                current_success_rate = float(self.vec_env.successes.float().mean().item())
            if hasattr(self.vec_env, "has_hit_table"):
                current_hit_table_rate = float(self.vec_env.has_hit_table.float().mean().item())

            if self.print_log:
                self.log(locals())

            if it % int(self.save_interval) == 0:
                # 按迭代间隔保存模型，文件名与旧训练习惯保持接近。
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()

        self.save(os.path.join(self.log_dir, f"model_{self.num_learning_iterations}.pt"))

    def log(self, locs, width: int = 80, pad: int = 35) -> None:
        """Write TensorBoard metrics and print a compact training summary."""
        self.tot_timesteps += int(self.num_transitions_per_env) * self.num_envs
        self.tot_time += locs["collection_time"] + locs["learn_time"]
        iteration_time = locs["collection_time"] + locs["learn_time"]
        mean_std = self.actor_critic.log_std.exp().mean()

        # TensorBoard 指标：loss、探索噪声、平均奖励、成功率等。
        self.writer.add_scalar("Loss/value_function", locs["mean_value_loss"], locs["it"])
        self.writer.add_scalar("Loss/surrogate", locs["mean_surrogate_loss"], locs["it"])
        self.writer.add_scalar("Policy/mean_noise_std", mean_std.item(), locs["it"])
        if len(locs["rewbuffer"]) > 0:
            self.writer.add_scalar("Train/mean_reward", statistics.mean(locs["rewbuffer"]), locs["it"])
            self.writer.add_scalar("Train/mean_episode_length", statistics.mean(locs["lenbuffer"]), locs["it"])
        self.writer.add_scalar("Train/mean_reward/step", locs["mean_reward"].item(), locs["it"])
        self.writer.add_scalar("Train/mean_episode_length/episode", locs["mean_trajectory_length"].item(), locs["it"])
        self.writer.add_scalar("Train/current_success_rate", locs["current_success_rate"], locs["it"])
        self.writer.add_scalar("Train/current_hit_table_rate", locs["current_hit_table_rate"], locs["it"])

        fps = int(int(self.num_transitions_per_env) * self.num_envs / max(iteration_time, 1e-6))
        heading = f" Learning iteration {locs['it']}/{locs['num_learning_iterations']} "
        # 终端日志保持紧凑，便于长时间训练时快速观察趋势。
        log_string = (
            f"{'#' * width}\n"
            f"{heading.center(width, ' ')}\n\n"
            f"{'Computation:':>{pad}} {fps:.0f} steps/s "
            f"(collection: {locs['collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"
            f"{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"
            f"{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"
            f"{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"
            f"{'Mean reward/step:':>{pad}} {locs['mean_reward'].item():.2f}\n"
            f"{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length'].item():.2f}\n"
            f"{'Current success rate:':>{pad}} {locs['current_success_rate']:.2f}\n"
            f"{'Current hit table rate:':>{pad}} {locs['current_hit_table_rate']:.2f}\n"
            f"{'-' * width}\n"
            f"{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"
            f"{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"
            f"{'Total time:':>{pad}} {self.tot_time:.2f}s\n"
            f"{'ETA:':>{pad}} "
            f"{self.tot_time / (locs['it'] + 1) * (locs['num_learning_iterations'] - locs['it']):.1f}s\n"
        )
        print(log_string)

    def update(self) -> tuple[float, float]:
        """Run PPO epochs over the collected one-step batch."""
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        num_updates = 0

        for _ in range(int(self.num_learning_epochs)):
            batch = self.storage.mini_batch_generator(int(self.num_mini_batches))
            for indices in batch:
                # storage 的前两维是 [time, env]，更新时展平成 batch 维。
                obs_batch = self.storage.observations.view(-1, *self.storage.observations.size()[2:])[indices]
                states_batch = (
                    self.storage.states.view(-1, *self.storage.states.size()[2:])[indices] if self.asymmetric else None
                )
                actions_batch = self.storage.actions.view(-1, self.storage.actions.size(-1))[indices]
                target_values_batch = self.storage.values.view(-1, 1)[indices]
                returns_batch = self.storage.returns.view(-1, 1)[indices]
                old_actions_log_prob_batch = self.storage.actions_log_prob.view(-1, 1)[indices]
                advantages_batch = self.storage.advantages.view(-1, 1)[indices]
                old_mu_batch = self.storage.mu.view(-1, self.storage.actions.size(-1))[indices]
                old_sigma_batch = self.storage.sigma.view(-1, self.storage.actions.size(-1))[indices]

                actions_log_prob_batch, entropy_batch, value_batch, mu_batch, sigma_batch = self.actor_critic.evaluate(
                    obs_batch, states_batch, actions_batch
                )

                if self.desired_kl is not None and self.schedule == "adaptive":
                    # KL 过大降低学习率，KL 过小提高学习率，维持 PPO 更新幅度。
                    kl = torch.sum(
                        sigma_batch
                        - old_sigma_batch
                        + (
                            torch.square(old_sigma_batch.exp())
                            + torch.square(old_mu_batch - mu_batch)
                        )
                        / (2.0 * torch.square(sigma_batch.exp()))
                        - 0.5,
                        dim=-1,
                    )
                    kl_mean = torch.mean(kl)
                    if kl_mean > self.desired_kl * 2.0:
                        self.step_size = max(1e-5, self.step_size / 1.5)
                    elif 0.0 < kl_mean < self.desired_kl / 2.0:
                        self.step_size = min(1e-2, self.step_size * 1.5)
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.step_size

                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                # PPO clipped surrogate：限制新旧策略概率比，避免一次更新过大。
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                )
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                if self.use_clipped_value_loss:
                    # 可选 value clipping，与 policy clipping 类似，用于稳定 critic。
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                        -self.clip_param, self.clip_param
                    )
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                # PPO 总损失 = policy loss + value loss - entropy bonus。
                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                num_updates += 1

        num_updates = max(1, num_updates)
        return mean_value_loss / num_updates, mean_surrogate_loss / num_updates
