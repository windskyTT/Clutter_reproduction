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
import traceback
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
    parser.add_argument("--sleep_per_step", type=float, default=0.0, help="Sleep seconds after each replay step.")
    parser.add_argument("--hold_after_round", type=float, default=0.0, help="Sleep seconds after each eval round.")
    parser.add_argument(
        "--checkpoint_preset",
        type=str,
        default="auto",
        choices=("auto", "none", "demograsp_inspire_vision"),
        help=(
            "Compatibility preset for old DemoGrasp checkpoints. "
            "'auto' applies the Inspire vision preset when the checkpoint file is inspire.pt."
        ),
    )
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
from Clutter.utils.paths import CLUTTER_ROOT


def log_stage(message: str) -> None:
    """打印播放阶段日志，并立即 flush，方便定位 IsaacSim 启动后的卡点/异常点。"""
    print(f"[PLAY] {message}", flush=True)


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


def obs_dim_from_type(obs_type: str, points_per_object: int) -> int:
    """按 ClutterEnv.compute_required_observations() 的拼接规则计算观测维度。

    旧 DemoGrasp 用 Hydra 根据 `observationType` 自动累加维度；迁移到 IsaacLab
    Python cfg 后，脚本覆盖观测类型时也需要同步更新 `num_observations` 和
    `observation_space`，否则 env 的实际观测和 Gym space 会不一致。
    """
    dims = {
        "armdof": 7,
        "handdof": 6,
        "fulldof": 19,
        "eefpose": 7,
        "ftpos": 15,
        "palmpose": 7,
        "lastact": 13,
        "objxyz": 3,
        "objpose": 7,
        "objinitpose": 7,
        "objpcl": int(points_per_object) * 3,
    }
    total = 0
    for item in obs_type.split("+"):
        item = item.strip()
        if not item:
            continue
        if item not in dims:
            raise ValueError(f"Unsupported observation item {item!r} in obs_type={obs_type!r}.")
        total += dims[item]
    return total


def should_apply_demograsp_inspire_preset(checkpoint: Path) -> bool:
    """判断是否自动套用旧 DemoGrasp Inspire 视觉 checkpoint 的运行参数。"""
    if args_cli.checkpoint_preset == "none":
        return False
    if args_cli.checkpoint_preset == "demograsp_inspire_vision":
        return True
    return checkpoint.name == "inspire.pt"


def apply_demograsp_inspire_preset(env_cfg) -> dict:
    """应用旧 Gym 指令中 `ckpt/inspire.pt` 对应的 Clutter/IsaacLab 覆盖项。

    旧命令核心覆盖项：
    - observationType="eefpose+objinitpose+objpcl"
    - armController=pose
    - enablePointCloud=True
    - randomizeTrackingReference=True
    - randomizeGraspPose=True
    - train.params.is_vision=True

    在新脚本里，环境覆盖写入 `env_cfg`，训练/网络覆盖作为 dict 返回给
    `default_ppo_onestep_cfg()`。
    """
    env_cfg.obs_type = "eefpose+objinitpose+objpcl"
    env_cfg.enable_point_cloud = True
    env_cfg.arm_controller = "pose"
    env_cfg.randomize_tracking_reference = True
    env_cfg.randomize_grasp_pose = True
    env_cfg.episode_length_steps = 50
    env_cfg.episode_length_s = env_cfg.episode_length_steps * env_cfg.decimation / 60.0

    obs_dim = obs_dim_from_type(env_cfg.obs_type, env_cfg.points_per_object)
    env_cfg.num_observations = obs_dim
    env_cfg.observation_space = obs_dim

    # checkpoint 的 PointNet 输入仍是 512x3；PointNet 内部会拼接去均值坐标，
    # 因此权重中第一层卷积显示为 6 维输入，这是旧实现的正常行为。
    return {
        "is_vision": True,
        "policy": {
            "pc_shape": [int(env_cfg.points_per_object), 3],
            "pc_emb_dim": 128,
        },
    }


def main() -> None:
    """创建环境、加载策略权重，并运行确定性推理。环境初始化与加载推理"""
    torch.manual_seed(args_cli.seed)
    env = None
    try:
        checkpoint = resolve_checkpoint(args_cli.checkpoint)
        # 从 Gym 注册表读取环境配置，并应用命令行覆盖项。
        log_stage(f"Parsing env cfg: task={args_cli.task}, num_envs={args_cli.num_envs}")
        env_cfg = parse_env_cfg( # 创建底层物理仿真环境
            args_cli.task,
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=not args_cli.disable_fabric,
        )
        train_overrides = {
            "times_testing_all_objects": args_cli.episodes,
            "sleep_per_step": args_cli.sleep_per_step,
            "hold_after_round": args_cli.hold_after_round,
        }
        if should_apply_demograsp_inspire_preset(checkpoint):
            log_stage("Applying DemoGrasp Inspire vision checkpoint preset.")
            train_overrides = {
                **train_overrides,
                **apply_demograsp_inspire_preset(env_cfg),
            }

        log_stage(
            "Creating Gym env: "
            f"obs_type={env_cfg.obs_type}, obs_dim={env_cfg.num_observations}, "
            f"arm_controller={env_cfg.arm_controller}, point_cloud={env_cfg.enable_point_cloud}"
        )
        env = gym.make(args_cli.task, cfg=env_cfg)
        log_stage("Gym env created.")

        # test=True 会让 PPO runner 进入 eval 分支，不创建 TensorBoard writer。告诉配置生成器这是测试模式。
        # 内部逻辑可能会关闭探索噪声（Exploration Noise），使得神经网络直接输出当前最优的确定性动作（Argmax），并且不需要收集缓冲区数据进行反向传播
        # overrides 将命令行传入的测试轮数强制覆盖进配置字典里
        train_cfg = default_ppo_onestep_cfg(test=True, overrides=train_overrides)
        # 实例化 PPO 算法 注意这里的 log_dir=None，因为只是播放模型，不需要写出 TensorBoard 日志或保存新的权重
        log_stage("Creating PPO runner.")
        runner = ppo_onestep.PPO(
            vec_env=env.unwrapped,
            actor_critic_class=ppo_onestep.ActorCritic,
            train_param=train_cfg,
            log_dir=None,
            apply_reset=False,
            action_dim=infer_planner_action_dim(env),
        )
        log_stage(f"Loading policy checkpoint from {checkpoint}")
        # 核心步骤。调用算法的 test 方法，内部会执行 torch.load，把 checkpoint 里的神经网络权重覆盖到当前创建的模型中，并切换为 eval() 模式
        runner.test(str(checkpoint))
        log_stage("Running policy evaluation.")
        runner.run()
    finally:
        if env is not None:
            log_stage("Closing Gym env.")
            env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("[ERROR] play_ppo_onestep.py failed with an exception:", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise
    finally:
        # 推理结束或异常退出时都要关闭 Isaac Sim。
        simulation_app.close()
