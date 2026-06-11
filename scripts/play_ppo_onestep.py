#!/usr/bin/env python3
"""
train_ppo_onestep.py 的姊妹篇，专门用于加载预训练的强化学习模型，并进行仿真测试/评估（Inference/Play）

加载并播放 Clutter one-step PPO 策略。

用途：
- 替代旧的 play policy shell/入口。
- 启动 IsaacLab 环境，加载 `assets/checkpoints/*.pt` 中的策略权重。
- 以 deterministic/inference 模式执行若干轮 one-step grasp 评估。
"""

from __future__ import annotations # 开启延迟评估类型注解。这允许你在代码中使用尚未定义的类名作为类型提示

import argparse
import math
import sys
from pathlib import Path

# 允许脚本在未 pip install extension 时直接从仓库运行。
# 这对迁移阶段很方便：修改 source/Clutter 后可以直接运行脚本验证。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "source" / "Clutter"
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    """解析播放/推理参数。
    IsaacLab 的 AppLauncher 参数也会追加到这里，因此必须在启动仿真前完成。
    """
    parser = argparse.ArgumentParser(description="Play Clutter one-step PPO policy.")
    parser.add_argument("--task", type=str, default="Clutter-Grasp-Direct-v0", help="Gymnasium task id.")
    parser.add_argument("--num_envs", type=int, default=None, help="Override the number of vectorized envs.")
    parser.add_argument("--seed", type=int, default=42, help="Torch random seed.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="assets/checkpoints/inspire.pt",
        # 默认 checkpoint 对应迁移方案中的 assets/checkpoints 目录。
        help="Checkpoint path, relative to the Clutter project root by default.",
    )
    # 评估的轮数。默认执行 10 轮 one-step 抓取评估，覆盖所有测试对象。
    parser.add_argument("--episodes", type=int, default=10, help="Number of one-step evaluation rounds.")
    parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable IsaacLab fabric.")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = parse_args()
# AppLauncher 负责创建/连接 Isaac Sim 应用；之后才能安全导入仿真相关模块。
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# AppLauncher 之后的导入才会拿到 Isaac Sim 注入的运行时依赖。
import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import Clutter.tasks.direct.clutter  # noqa: F401
from Clutter.algo import ppo_onestep
from Clutter.tasks.direct.clutter.agents.ppo_onestep_cfg import default_ppo_onestep_cfg
from Clutter.utils.paths import PROJECT_ROOT as CLUTTER_ROOT


def infer_planner_action_dim(env) -> int:
    """推断加载 checkpoint 时需要的 actor 输出维度。
    维度必须和训练 checkpoint 完全一致，否则 `load_state_dict` 会因为参数形状不匹配失败。
    """
    raw_env = env.unwrapped if hasattr(env, "unwrapped") else env
    # 抓取环境已迁移完成时，用 planner hook 判断 6D 位姿动作和手部自由度。
    if hasattr(raw_env, "generate_reaching_plan_idx"):
        action_dim = 6
        if getattr(raw_env, "randomize_grasp_pose", False):
            action_dim += int(getattr(raw_env, "num_active_hand_dofs", 0))
        return action_dim

    space = getattr(raw_env, "action_space", None)
    if space is None or not hasattr(space, "shape"):
        raise RuntimeError("Cannot infer action dimension: env has no action_space shape.")
    shape = tuple(space.shape)
    if len(shape) >= 2 and shape[0] == int(getattr(raw_env, "num_envs", 0)):
        shape = shape[1:]
    return int(math.prod(shape))


def resolve_checkpoint(path_text: str) -> Path:
    """解析 checkpoint 路径。
    用户传相对路径时，默认按 Clutter 项目根目录解析，避免受当前工作目录影响。
    """
    ckpt_path = Path(path_text).expanduser()
    if not ckpt_path.is_absolute():
        ckpt_path = CLUTTER_ROOT / ckpt_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return ckpt_path


def main() -> None:
    """创建环境、加载策略权重，并运行确定性推理。环境初始化与加载推理"""
    torch.manual_seed(args_cli.seed)
    env = None
    try:
        # 从 Gym 注册表读取环境配置，并应用命令行覆盖项。
        env_cfg = parse_env_cfg( # 创建底层物理仿真环境
            args_cli.task,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=not args_cli.disable_fabric,
        )
        env = gym.make(args_cli.task, cfg=env_cfg)
        checkpoint = resolve_checkpoint(args_cli.checkpoint)

        # test=True 会让 PPO runner 进入 eval 分支，不创建 TensorBoard writer。告诉配置生成器这是测试模式。
        # 内部逻辑可能会关闭探索噪声（Exploration Noise），使得神经网络直接输出当前最优的确定性动作（Argmax），并且不需要收集缓冲区数据进行反向传播
        # overrides 将命令行传入的测试轮数强制覆盖进配置字典里
        train_cfg = default_ppo_onestep_cfg(test=True, overrides={"times_testing_all_objects": args_cli.episodes})
        # 实例化 PPO 算法 注意这里的 log_dir=None，因为只是播放模型，不需要写出 TensorBoard 日志或保存新的权重
        runner = ppo_onestep.PPO(
            vec_env=env.unwrapped,
            actor_critic_class=ppo_onestep.ActorCritic,
            train_param=train_cfg,
            log_dir=None,
            apply_reset=False,
            action_dim=infer_planner_action_dim(env),
        )
        print(f"[INFO] Loading policy checkpoint from {checkpoint}")
        # 核心步骤。调用算法的 test 方法，内部会执行 torch.load，把 checkpoint 里的神经网络权重覆盖到当前创建的模型中，并切换为 eval() 模式
        runner.test(str(checkpoint))
        runner.run()
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        # 推理结束或异常退出时都要关闭 Isaac Sim。
        simulation_app.close()
