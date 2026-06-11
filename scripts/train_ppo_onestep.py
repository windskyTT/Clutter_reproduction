#!/usr/bin/env python3
"""
基于 IsaacLab 的强化学习（PPO）训练入口脚本
由于 Isaac Sim/IsaacLab 涉及底层物理引擎、GPU 渲染和 Omniverse 框架，
它的启动顺序和资源管理与普通的 Python 脚本有很大不同（例如必须先启动 App，再导入深度学习库）

训练 Clutter 抓取任务的 one-step PPO 策略。

用途：
- 替代旧的 `run_rl_grasp.py` 训练入口。
- 负责启动 Isaac Sim / IsaacLab、创建 Gymnasium 环境、创建 PPO runner。
- 训练日志、配置快照和模型 checkpoint 统一写入 Clutter 项目根目录下的 `logs/`。

注意：
- IsaacLab 要求先通过 `AppLauncher` 启动仿真应用，再导入 gym/torch/任务包。
- 具体的强化学习逻辑在 `Clutter.algo.ppo_onestep` 中，这个脚本只做入口编排。
"""

from __future__ import annotations # 开启延迟评估类型注解。这允许你在代码中使用尚未定义的类名作为类型提示

import argparse # 用于解析命令行参数
import json # 用于处理 JSON 数据，用来保存训练配置
import math # 提供数学函数，用来计算动作维度
import sys # 提供对 Python 解释器的访问，用于修改模块搜索路径
from datetime import datetime # 用于获取当前时间，生成唯一的日志目录名称 生成时间戳
from pathlib import Path # 提供面向对象的文件系统路径操作，方便处理文件路径

# 直接从仓库运行脚本时，把 extension 包加入 Python 路径。
# 这样即使还没有 `pip install -e source/Clutter`，也能找到 `Clutter.*` 包。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# __file__ 是当前脚本的路径。.resolve() 获取绝对路径。.parents[1] 获取它的“爷爷”目录（向上两级），这里定义为整个项目的根目录 PROJECT_ROOT
SOURCE_DIR = PROJECT_ROOT / "source" / "Clutter" # 接出项目自定义源码包 Clutter 所在的目录
if str(SOURCE_DIR) not in sys.path: # 检查该源码目录是否已经在 Python 的搜索路径中
    sys.path.insert(0, str(SOURCE_DIR)) # 如果不在，就将其插入到 sys.path 的最前面

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace: # 定义参数解析函数
    """解析训练命令行参数。

    这里必须在 AppLauncher 启动之前完成，因为 IsaacLab 会把自身的渲染、设备、
    headless 等参数追加到同一个 parser 中。
    """
    parser = argparse.ArgumentParser(description="Train Clutter one-step PPO policy.")
    # `task` 指定要运行的强化学习任务名称，注册在 Gymnasium 中
    parser.add_argument("--task", type=str, default="Clutter-Grasp-Direct-v0", help="Gymnasium task id.")
    # `num_envs` 用于覆盖环境配置中的并行环境数量，便于调试时减小规模
    parser.add_argument("--num_envs", type=int, default=None, help="Override the number of vectorized envs.")
    # 'seed'指定随机种子以保证实验的可复现性
    parser.add_argument("--seed", type=int, default=42, help="Torch random seed.")
    # `checkpoint` 用于指定预训练模型的路径，如果提供了就从该 checkpoint 恢复训练
    parser.add_argument("--checkpoint", type=str, default="", help="Optional policy checkpoint to resume from.")
    parser.add_argument("--run_name", type=str, default="", help="Optional log directory name.") # 自定义日志文件夹的名称
    parser.add_argument("--max_iterations", type=int, default=None, help="Override PPO training iterations.")
    # `disable_fabric` 是一个布尔开关，如果设置了这个参数，就会禁用 IsaacLab 的 fabric 功能（Fabric 是 Omniverse 中加速数据传输到底层的机制）
    parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable IsaacLab fabric.")
    # 将 Isaac Sim 原生的命令行参数（例如 --headless 无头模式, --device GPU选择等）追加到我们自定义的 parser 中
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = parse_args() # 执行解析，获取终端输入的参数
app_launcher = AppLauncher(args_cli)
# AppLauncher 会启动 Isaac Sim 应用，并根据 args_cli 设置 headless/device/渲染参数。
simulation_app = app_launcher.app # 在这一行，Omniverse 和底层物理引擎正式启动。必须在这一步之后才能导入后续与 GPU 或强化学习相关的库

"""AppLauncher 启动后再导入 IsaacLab 运行时相关模块，避免 Omniverse 依赖未初始化
 由于它们会分配 GPU 显存，所以必须在 simulation_app 启动后导入，避免与 Isaac Sim 的显存分配发生冲突"""
import gymnasium as gym # 用于创建和管理强化学习环境
import torch # 用于深度学习模型的构建和训练
from omegaconf import OmegaConf # 用于处理配置文件，特别是 YAML 格式的配置 处理和保存层级化配置文件

import isaaclab_tasks # 导入 IsaacLab 自带任务包，注册 Gym 环境 
# noqa: F401 是告诉代码检查工具（如 flake8）忽略“已导入但未使用”的警告，因为仅仅导入这个包就会触发其内部的 Gymnasium 环境注册机制 
from isaaclab_tasks.utils import parse_env_cfg # 解析并合并环境配置项的工具函数。

# 导入项目中的自己强化学习算法（ppo_onestep）、默认配置和路径定义。同样，导入任务包是为了向 gym 注册 --task 中使用的环境名称
import Clutter.tasks.direct.clutter  # noqa: F401
from Clutter.algo import ppo_onestep
from Clutter.tasks.direct.clutter.agents.ppo_onestep_cfg import default_ppo_onestep_cfg
from Clutter.utils.paths import LOGS_DIR, PROJECT_ROOT as CLUTTER_ROOT


def infer_planner_action_dim(env) -> int:
    """辅助函数，推断 PPO 算法它的 Actor 网络 输出动作维度。

    抓取任务里 one-step policy 输出的是“规划动作”：前 6 维通常表示末端位姿扰动；
    如果环境启用随机抓取姿态，还会追加手部主动自由度。若当前环境仍是 IsaacLab
    模板环境，则退回到 Gym action_space，方便迁移期间做最小 smoke test
    """
    raw_env = env.unwrapped if hasattr(env, "unwrapped") else env #去除 Gymnasium 的 wrapper，获取最底层的 IsaacLab 环境实例
    # 检查环境配置。由于这是“one-step”策略（单步规划抓取），它强制要求环境必须启用随机追踪参考配置。如果没有，直接抛出异常
    if hasattr(raw_env, "randomize_tracking_reference") and not raw_env.randomize_tracking_reference:
        raise RuntimeError("ppo_onestep expects randomize_tracking_reference=True on the grasp environment.")

    # 针对项目内自定义环境的特殊逻辑
    if hasattr(raw_env, "generate_reaching_plan_idx"):
        action_dim = 6
        # 如果启用了随机抓取姿态，那么需要再加上手部的自由度（num_active_hand_dofs），并将它们合并返回
        if getattr(raw_env, "randomize_grasp_pose", False):
            action_dim += int(getattr(raw_env, "num_active_hand_dofs", 0))
        return action_dim

    # Fallback 逻辑：如果上面找不到特定属性，说明使用的是旧版或通用的直接 RL 环境。此时直接读取 env.action_space.shape
    space = getattr(raw_env, "action_space", None)
    if space is None or not hasattr(space, "shape"):
        raise RuntimeError("Cannot infer action dimension: env has no action_space shape.")
    shape = tuple(space.shape)
    # 因为 IsaacLab 是并行环境，动作空间形状经常是 (num_envs, action_dim)。这里剔除表示环境数量的第一个维度，算出单个环境的动作维度大小。
    if len(shape) >= 2 and shape[0] == int(getattr(raw_env, "num_envs", 0)):
        shape = shape[1:]
    return int(math.prod(shape))


def build_runner(env, train_cfg, checkpoint: str = "", run_name: str = "") -> ppo_onestep.PPO:
    """创建 PPO runner训练器，并把本次训练配置写入日志目录。
    """
    run_name = run_name or f"{args_cli.task}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    log_dir = LOGS_DIR / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # 将当前的训练超参数配置（train_cfg）转换成原生字典，并以缩进格式保存到日志文件夹下的 config.json 中，以便后期追踪实验设定
    with (log_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(OmegaConf.to_container(train_cfg, resolve=True), f, indent=4)

    # vec_env 传入 unwrapped，避免 Gymnasium wrapper 隐藏 DirectRLEnv 的自定义方法。
    # 实例化 PPO 算法核心类。传入并行环境 vec_env、网络架构 actor_critic_class、参数配置、日志路径以及刚刚计算出的 action_dim
    runner = ppo_onestep.PPO(
        vec_env=env.unwrapped,
        actor_critic_class=ppo_onestep.ActorCritic,
        train_param=train_cfg,
        log_dir=str(log_dir),
        apply_reset=False,
        action_dim=infer_planner_action_dim(env),
    )

    if checkpoint:
        ckpt_path = Path(checkpoint).expanduser()
        if not ckpt_path.is_absolute():
            # 相对路径统一按 Clutter 项目根目录解析，和 play 脚本保持一致。
            ckpt_path = CLUTTER_ROOT / ckpt_path
        print(f"[INFO] Loading pre-trained policy from {ckpt_path}")
        runner.load(str(ckpt_path)) # 将检查点加载到 PPO 模型中进行微调/续训
    return runner


def main() -> None:
    """创建 IsaacLab 环境并启动 PPO 训练主循环。"""
    torch.manual_seed(args_cli.seed) # 固定 PyTorch 的随机种子
    env = None
    try: # 确保即使训练代码崩溃，finally 块中的环境清理逻辑也会执行
        # parse_env_cfg 从 Gym 注册表读取 env_cfg_entry_point，并应用设备/并行数覆盖。
        # 结合 Gymnasium 注册表里的默认配置与终端传入的参数（如并行数量、计算设备、Fabric加速等），生成最终的环境配置文件
        env_cfg = parse_env_cfg(
            args_cli.task,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=not args_cli.disable_fabric,
        )
        env = gym.make(args_cli.task, cfg=env_cfg) # 调用 Gymnasium API，利用合并好的配置正式创建并行强化学习环境实例

        # 读取 Clutter 内部 one-step PPO 默认配置；命令行只覆盖常用字段
        # 获取算法默认的训练参数（如学习率、batch size 等），并根据命令行参数覆盖 max_iterations 和日志目录等字段
        train_cfg = default_ppo_onestep_cfg(test=False)
        train_cfg.log_dir = str(LOGS_DIR)
        if args_cli.max_iterations is not None:
            train_cfg.max_iterations = args_cli.max_iterations

        # 调用前面定义的函数，完成 PPO 的组装
        runner = build_runner(env, train_cfg, checkpoint=args_cli.checkpoint, run_name=args_cli.run_name)
        runner.run() # 正式开始强化学习训练主循环（采集数据、计算优势函数、反向传播更新网络等逻辑都在这里面执行）
    finally:
        if env is not None:
            env.close() # 训练结束后或发生异常时，安全关闭 Gymnasium 环境实例


if __name__ == "__main__": # 确保只有在直接运行该脚本时才会执行以下代码（被别的脚本 import 时不会执行）
    try:
        main()
    finally:
        # 无论训练是否异常退出，都关闭 Isaac Sim 应用，释放 GPU/窗口资源。
        simulation_app.close()

