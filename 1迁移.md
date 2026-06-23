# DemoGrasp 到 Clutter / IsaacLab 2.3.2 迁移方案

## 背景

当前环境：

- Isaac Sim 和 IsaacLab 使用预编译二进制安装在 `/home/windsky`。
- 已经有配置好的 `clutter` conda 环境。
- 已经使用 `./isaaclab.sh --new` 创建 IsaacLab extension 项目：
  `/home/windsky/project/Clutter`
- DemoGrasp 项目位于：
  `/home/windsky/project/Demograsp`
- 目标是把 DemoGrasp 依赖的 IsaacGym Preview 4 / IsaacGymEnvs 替换为 IsaacLab 2.3.2。
- 迁移后 Clutter 中不能出现任何 `Demograsp` 文件夹。

本方案只说明需要修改、复制、保留、放弃的内容和迁移路径，不执行实际迁移操作。

## 总结

DemoGrasp 可以迁移到 IsaacLab 2.3.2，但不能整仓库直接复制进 Clutter。

DemoGrasp 的核心环境 `tasks/grasp.py` 是基于 IsaacGym Preview 4 的 `VecTask`，大量使用：

```python
from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgymenvs.tasks.base.vec_task import VecTask
self.gym.*
```

这些 API 在 IsaacLab 中不能照搬。迁移时必须把环境层重写成 IsaacLab 的 `DirectRLEnv`。  
奖励函数、PPO 算法、PointNet、数据工具和资产可以迁移，但要拆进 Clutter 的 extension 包结构中。

当前 Clutter 实际 Python 包路径是：

```text
/home/windsky/project/Clutter/source/Clutter/Clutter/
```

所以后文目标路径都以这个实际结构为准。

## 最终目录结构

Clutter 中不要有：

```text
/home/windsky/project/Clutter/Demograsp/
/home/windsky/project/Clutter/source/Clutter/Clutter/Demograsp/
import DemoGrasp
import demograsp
```

建议最终结构如下：

```text
/home/windsky/project/Clutter/
├── assets/                                      # 资产根目录
│   ├── camera_pad.urdf
│   ├── franka/
│   ├── fr3_gripper/
│   │   ├── fr3_dclaw_gripper.urdf
│   │   ├── fr3_panda_gripper.urdf
│   │   └── meshes/
│   ├── inspire_tac/
│   │   ├── fr3_inspire_tac_L_right_safety.urdf
│   │   ├── fr3_inspire_tac_L_right_safety_visual_realistic.urdf
│   │   └── meshes/
│   ├── shadow_hand_simple/
│   ├── ur5_allegro/
│   ├── ur5_meshes/
│   ├── ur5_svh/
│   ├── union_ycb_unidex/ 没弄
│   │   ├── meshes/
│   │   ├── pointclouds/
│   │   ├── urdf/
│   │   ├── train_set.yaml
│   │   ├── test_set_seen_cat.yaml
│   │   ├── test_set_unseen_cat.yaml
│   │   ├── union_ycb_debugset.yaml
│   │   └── union_ycb_debugset_names.yaml
│   ├── textures/ 没弄
│   │   ├── background/
│   │   ├── object/
│   │   └── white.png
│   ├── references/                              # 从 tasks/grasp_ref_*.pkl 迁来
│   │   ├── grasp_ref_allegro.pkl
│   │   ├── grasp_ref_dclaw_gripper.pkl
│   │   ├── grasp_ref_inspire.pkl
│   │   ├── grasp_ref_panda_gripper.pkl
│   │   ├── grasp_ref_shadow.pkl
│   │   └── grasp_ref_svh.pkl
│   └── checkpoints/                             # 从 ckpt/*.pt 迁来
│       ├── fr3_dclaw.pt
│       ├── fr3_panda_gripper.pt
│       ├── fr3_shadow.pt
│       ├── inspire.pt
│       ├── shadow.pt
│       ├── ur5_allegro.pt
│       └── ur5_svh.pt
├── scripts/
│   ├── train_ppo_onestep.py                     # 新建，替代 run_rl_grasp.py 和 train.sh
│   ├── play_ppo_onestep.py                      # 新建，替代 play_policy.sh
│   └── collect_dataset.py                       # 可选，替代 collect_real_dataset 入口
└── source/
    └── Clutter/
        ├── config/
        │   └── extension.toml
        ├── pyproject.toml
        ├── setup.py
        └── Clutter/
            ├── __init__.py
            ├── algo/
            │   ├── __init__.py
            │   ├── ppo_onestep/
            │   │   ├── __init__.py
            │   │   ├── module.py
            │   │   ├── ppo.py
            │   │   └── storage.py
            │   └── pn_utils/
            ├── data/
            │   ├── __init__.py
            │   ├── dataset_utils.py
            │   └── lerobot/                     # 仅数据采集需要
            ├── tasks/
            │   └── direct/
            │       └── clutter/
            │           ├── __init__.py          # Gymnasium 环境注册
            │           ├── clutter_env.py       # IsaacLab DirectRLEnv 环境
            │           ├── clutter_env_cfg.py   # 环境配置
            │           ├── reward.py
            │           ├── utils.py
            │           ├── torch_math.py
            │           ├── agents/
            │           │   ├── __init__.py
            │           │   └── ppo_onestep_cfg.py
            │           └── config/
            │               └── hand/
            │                   ├── fr3_dclaw_gripper.yaml
            │                   ├── fr3_inspire_tac.yaml
            │                   ├── fr3_panda_gripper.yaml
            │                   ├── fr3_shadow.yaml
            │                   ├── shadow_simple.yaml
            │                   ├── ur5_allegro.yaml
            │                   └── ur5_svh.yaml
            └── utils/
                ├── __init__.py
                └── paths.py
```

## 需要大改或重写的代码

### 1. `run_rl_grasp.py`

源路径：

```text
/home/windsky/project/Demograsp/run_rl_grasp.py
```

目标路径：

```text
/home/windsky/project/Clutter/scripts/train_ppo_onestep.py
/home/windsky/project/Clutter/scripts/play_ppo_onestep.py
```

处理方式：不要原样复制，按 IsaacLab 入口重写。

原代码依赖：

```python
import hydra
import gym
from isaacgym import gymapi
from isaacgym import gymutil
import isaacgymenvs
from isaacgymenvs.utils.utils import set_np_formatting, set_seed
from isaacgymenvs.utils.torch_jit_utils import *
import tasks
```

迁移后应改为：

```python
from isaaclab.app import AppLauncher
import gymnasium as gym
import torch

import Clutter.tasks.direct.clutter
from Clutter.algo import ppo_onestep
```

新入口需要负责：

- 启动 Isaac Sim / IsaacLab `AppLauncher`。
- 使用 `gymnasium.make("Clutter-Grasp-Direct-v0", cfg=...)` 创建环境。
- 加载 Clutter 内部配置。
- 加载 PPO one-step 算法。
- 训练时保存到 `/home/windsky/project/Clutter/logs` 
- 播放策略时从 `/home/windsky/project/Clutter/assets/checkpoints/*.pt` 读取 checkpoint。

原 PPO 期望环境接口是：

```text
env.num_envs
env.device
env.observation_space
env.state_space
env.action_space
env.reset_idx(...)
env.step(...)
env.get_state()
env.generate_reaching_plan_idx(...)
env.compute_reference_actions()
env.obs_dict["obs"]
env.successes
env.has_hit_table
```

IsaacLab `DirectRLEnv` 默认接口是：

```text
obs_dict, reward, terminated, truncated, extras = env.step(actions)
obs_dict = env.reset()
obs_dict["policy"]
```

做法：
修改 PPO，使它适配 IsaacLab 原生接口。


建议先用第 2 种，迁移成本小。

### 2. `tasks/__init__.py`

源路径：

```text
/home/windsky/project/Demograsp/tasks/__init__.py
```

目标路径：

```text
/home/windsky/project/Clutter/source/Clutter/Clutter/tasks/direct/clutter/__init__.py
```

原代码：

```python
from isaacgymenvs.tasks import isaacgym_task_map
from .grasp import Grasp

isaacgym_task_map["grasp"] = Grasp
```

迁移后不能再使用 `isaacgym_task_map`，要改为 Gymnasium 注册：

```python
import gymnasium as gym

from . import agents

gym.register(
    id="Clutter-Grasp-Direct-v0",
    entry_point=f"{__name__}.clutter_env:ClutterEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.clutter_env_cfg:ClutterEnvCfg",
        "ppo_onestep_cfg_entry_point": f"{agents.__name__}.ppo_onestep_cfg:PPOOneStepCfg",
    },
)
```

当前 Clutter 里已有示例注册名：

```text
Template-Clutter-Direct-v0
```

建议替换为：

```text
Clutter-Grasp-Direct-v0
```

### 3. `tasks/grasp.py`

源路径：

```text
/home/windsky/project/Demograsp/tasks/grasp.py
```

目标路径：

```text
/home/windsky/project/Clutter/source/Clutter/Clutter/tasks/direct/clutter/clutter_env.py
```

处理方式：参考逻辑迁移，不能原样运行。

必须替换的 IsaacGym API：

```text
VecTask                                      -> DirectRLEnv
create_sim                                  -> _setup_scene
_create_envs                                -> _setup_scene
_prepare_robot_asset                        -> ArticulationCfg + UrdfFileCfg 或 UsdFileCfg
_prepare_object_asset                       -> RigidObjectCfg / RigidObjectCollection
_prepare_table_asset                        -> CuboidCfg 或 spawn primitive
self.gym.load_asset                         -> sim_utils.UrdfFileCfg / sim_utils.UsdFileCfg
self.gym.create_actor                       -> scene assets config
self.gym.create_box                         -> sim_utils.CuboidCfg
self.gym.acquire_actor_root_state_tensor    -> RigidObject.data.root_state_w
self.gym.acquire_dof_state_tensor           -> Articulation.data.joint_pos / joint_vel
self.gym.acquire_rigid_body_state_tensor    -> Articulation.data.body_state_w
self.gym.acquire_jacobian_tensor            -> Articulation.root_physx_view.get_jacobians()
gymtorch.wrap_tensor                        -> IsaacLab tensor buffers
gymtorch.unwrap_tensor                      -> 不再需要
self.gym.refresh_*_tensor                   -> IsaacLab 自动同步或 asset.update
self.gym.set_actor_root_state_tensor_indexed -> write_root_pose_to_sim / write_root_velocity_to_sim
self.gym.set_dof_state_tensor_indexed       -> write_joint_state_to_sim
self.gym.set_dof_position_target_tensor     -> set_joint_position_target
self.gym.simulate                           -> DirectRLEnv 内部负责
self.gym.fetch_results                      -> DirectRLEnv 内部负责
self.gym.create_camera_sensor               -> IsaacLab Camera / TiledCamera
self.gym.get_camera_image_gpu_tensor        -> Camera.data.output
self.gym.set_rigid_body_color               -> USD prim material 或 visual material
self.gym.set_rigid_body_texture             -> USD material 绑定
self.gym.set_light_parameters               -> light cfg 或 USD light prim
```

IsaacLab `DirectRLEnv` 生命周期建议写成：

```python
class ClutterEnv(DirectRLEnv):
    cfg: ClutterEnvCfg

    def __init__(self, cfg: ClutterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._init_runtime_buffers()
        self._load_tracking_references()

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.object = RigidObject(...)
        self.table = RigidObject(...)
        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["object"] = self.object
        self.scene.clone_environments(copy_from_source=False)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clamp(-1.0, 1.0)

    def _apply_action(self):
        self.robot.set_joint_position_target(self.target_joint_pos)

    def _get_observations(self) -> dict:
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        return terminated, time_out

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        ...
```

DemoGrasp 中可以保留逻辑但要换底层 API 的函数：

```text
init_configs
generate_reaching_plan_idx
compute_reference_actions
pre_physics_step
compute_observations
compute_required_observations
transform_obj_pcl_2_world
compute_arm_ik
_control_ik
compute_reward
compute_real_observation_dict
orientation_error
linear_interpolate_poses
```

DemoGrasp 中需要重写底层实现的函数：

```text
__init__
create_sim
_create_ground_plane
_prepare_camera_pad_assets
_load_cameras
_create_envs
_prepare_robot_asset
_prepare_object_asset
_prepare_table_asset
reset_idx
step
post_physics_step
```

### 4. `tasks/config.yaml` 和 `tasks/task/grasp.yaml`

源路径：

```text
/home/windsky/project/Demograsp/tasks/config.yaml
/home/windsky/project/Demograsp/tasks/task/grasp.yaml
```

目标路径：

```text
/home/windsky/project/Clutter/source/Clutter/Clutter/tasks/direct/clutter/clutter_env_cfg.py
/home/windsky/project/Clutter/source/Clutter/Clutter/tasks/direct/clutter/agents/ppo_onestep_cfg.py
```

处理方式：不要原样复制。拆成 IsaacLab configclass 和 PPO config。

环境配置迁移到 `clutter_env_cfg.py`：

```text
num_envs
env_spacing
episode_length_s
decimation
sim.dt
sim.render_interval
asset_root
robot asset path
object list
table height
reset ranges
observation type
point cloud config
camera config
reward config
domain randomization config
```

PPO 配置迁移到 `agents/ppo_onestep_cfg.py`：

```text
cliprange
ent_coef
nsteps
noptepochs
nminibatches
max_iterations
max_grad_norm
optim_stepsize
schedule
desired_kl
gamma
lam
init_noise_std
policy.pi_hid_sizes
policy.vf_hid_sizes
policy.activation
policy.pc_shape
policy.pc_emb_dim
is_vision
save_interval
print_log
```

## 需要复制但小改的代码

### 1. `tasks/reward.py`

源路径：

```text
/home/windsky/project/Demograsp/tasks/reward.py
```

目标路径：

```text
/home/windsky/project/Clutter/source/Clutter/Clutter/tasks/direct/clutter/reward.py
```

处理方式：直接复制，基本不用改。

原因：它是纯 PyTorch 代码，不依赖 IsaacGym。

保留：

```text
reward_binary
REWARD_DICT
```

在 IsaacLab 中由 `_get_rewards()` 调用。

### 2. `tasks/utils.py`

源路径：

```text
/home/windsky/project/Demograsp/tasks/utils.py
```

目标路径：

```text
/home/windsky/project/Clutter/source/Clutter/Clutter/tasks/direct/clutter/utils.py
/home/windsky/project/Clutter/source/Clutter/Clutter/tasks/direct/clutter/torch_math.py
```

可以保留：

```text
batch_linear_interpolate_poses
COLORS_DICT
load_object_point_clouds
transform_points
farthest_point_sample
index_points
```

需要修改：

```python
from isaacgymenvs.utils.torch_jit_utils import *
```

替换为 IsaacLab 或自建工具：

```python
from isaaclab.utils.math import quat_mul, quat_conjugate
from .torch_math import quat_diff_rad, slerp, scale, unscale, to_torch
```

建议在 `torch_math.py` 中补齐 DemoGrasp 用到的旧工具函数：

```text
to_torch
torch_rand_float
scale
unscale
quat_mul
quat_conjugate
quat_from_angle_axis
quat_diff_rad
slerp
tensor_clamp
```

### 3. `algo/ppo_onestep/`

源路径：

```text
/home/windsky/project/Demograsp/algo/ppo_onestep/
```

目标路径：

```text
/home/windsky/project/Clutter/source/Clutter/Clutter/algo/ppo_onestep/
```

需要复制：

```text
__init__.py
module.py
ppo.py
storage.py
```

需要修改：

```python
from gym.spaces import Space
```

改为：

```python
from gymnasium.spaces import Space
```

删除没有实际使用的：

```python
from isaacgymenvs.utils.torch_jit_utils import *
```

包路径需要从：

```python
from algo import ppo_onestep
from algo.pn_utils...
```

改成：

```python
from Clutter.algo import ppo_onestep
from Clutter.algo.pn_utils...
```

### 4. `algo/pn_utils/`

源路径：

```text
/home/windsky/project/Demograsp/algo/pn_utils/
```

目标路径：

```text
/home/windsky/project/Clutter/source/Clutter/Clutter/algo/pn_utils/
```

处理方式：复制后小改。

主要修改：

- 所有 `algo.pn_utils...` 改为 `Clutter.algo.pn_utils...`
- 如果只使用 PointNet，可以先只复制 PointNet 相关最小子集。
- `sparseunet.py` 依赖较多，如果第一阶段不使用 sparse UNet，可以暂时不迁。

第一阶段建议至少保留：

```text
algo/pn_utils/maniskill_learn/networks/backbones/pointnet.py
algo/pn_utils/maniskill_learn/networks/modules/
algo/pn_utils/maniskill_learn/utils/
```

更省事的方式是先整体复制 `pn_utils/`，跑通后再删未用部分。

### 5. `tasks/hand/*.yaml`

源路径：

```text
/home/windsky/project/Demograsp/tasks/hand/*.yaml
```

目标路径：

```text
/home/windsky/project/Clutter/source/Clutter/Clutter/tasks/direct/clutter/config/hand/
```

需要复制：

```text
fr3_dclaw_gripper.yaml
fr3_inspire_tac.yaml
fr3_panda_gripper.yaml
fr3_shadow.yaml
shadow_simple.yaml
ur5_allegro.yaml
ur5_svh.yaml
```

配置中的相对资产路径可以保留：

```text
inspire_tac/fr3_inspire_tac_L_right_safety.urdf
inspire_tac/fr3_inspire_tac_L_right_safety_visual_realistic.urdf
fr3_gripper/fr3_dclaw_gripper.urdf
fr3_gripper/fr3_panda_gripper.urdf
shadow_hand_simple/fr3_shadow_right.urdf
shadow_hand_simple/fr3_shadow_right_visual_realistic.urdf
shadow_hand_simple/right_with_base.urdf
ur5_allegro/ur5_allegro.urdf
ur5_svh/ur5_svh.urdf
```

但代码里的 `asset_root` 必须统一指向：

```text
/home/windsky/project/Clutter/assets
```

### 6. 数据集工具

仅当你要迁移 `collect_real_dataset` 时需要。

源路径：

```text
/home/windsky/project/Demograsp/data/dataset_utils.py
/home/windsky/project/Demograsp/data/lerobot/
```

目标路径：

```text
/home/windsky/project/Clutter/source/Clutter/Clutter/data/dataset_utils.py
/home/windsky/project/Clutter/source/Clutter/Clutter/data/lerobot/
```

需要修改：

```python
HF_LEROBOT_HOME = ...
```

建议输出到：

```text
/home/windsky/project/Clutter/outputs/datasets
```

或：

```text
/home/windsky/project/Clutter/data/datasets
```

不要继续写到 DemoGrasp 路径。

## 需要复制的资产

当前检查结果：Clutter 的 `assets/` 已经基本覆盖 DemoGrasp 的 `assets/`，并额外包含：

```text
assets/references/
assets/checkpoints/
```

完整迁移路径如下。

### 机器人和物体资产

```text
/home/windsky/project/Demograsp/assets/camera_pad.urdf
-> /home/windsky/project/Clutter/assets/camera_pad.urdf

/home/windsky/project/Demograsp/assets/franka/
-> /home/windsky/project/Clutter/assets/franka/

/home/windsky/project/Demograsp/assets/fr3_gripper/
-> /home/windsky/project/Clutter/assets/fr3_gripper/

/home/windsky/project/Demograsp/assets/inspire_tac/
-> /home/windsky/project/Clutter/assets/inspire_tac/

/home/windsky/project/Demograsp/assets/shadow_hand_simple/
-> /home/windsky/project/Clutter/assets/shadow_hand_simple/

/home/windsky/project/Demograsp/assets/ur5_allegro/
-> /home/windsky/project/Clutter/assets/ur5_allegro/

/home/windsky/project/Demograsp/assets/ur5_meshes/
-> /home/windsky/project/Clutter/assets/ur5_meshes/

/home/windsky/project/Demograsp/assets/ur5_svh/
-> /home/windsky/project/Clutter/assets/ur5_svh/

/home/windsky/project/Demograsp/assets/union_ycb_unidex/
-> /home/windsky/project/Clutter/assets/union_ycb_unidex/

/home/windsky/project/Demograsp/assets/textures/
-> /home/windsky/project/Clutter/assets/textures/
```

### Reference grasp pkl

```text
/home/windsky/project/Demograsp/tasks/grasp_ref_allegro.pkl
-> /home/windsky/project/Clutter/assets/references/grasp_ref_allegro.pkl

/home/windsky/project/Demograsp/tasks/grasp_ref_dclaw_gripper.pkl
-> /home/windsky/project/Clutter/assets/references/grasp_ref_dclaw_gripper.pkl

/home/windsky/project/Demograsp/tasks/grasp_ref_inspire.pkl
-> /home/windsky/project/Clutter/assets/references/grasp_ref_inspire.pkl

/home/windsky/project/Demograsp/tasks/grasp_ref_panda_gripper.pkl
-> /home/windsky/project/Clutter/assets/references/grasp_ref_panda_gripper.pkl

/home/windsky/project/Demograsp/tasks/grasp_ref_shadow.pkl
-> /home/windsky/project/Clutter/assets/references/grasp_ref_shadow.pkl

/home/windsky/project/Demograsp/tasks/grasp_ref_svh.pkl
-> /home/windsky/project/Clutter/assets/references/grasp_ref_svh.pkl
```

对应配置要从：

```text
tasks/grasp_ref_inspire.pkl
```

改为：

```text
assets/references/grasp_ref_inspire.pkl
```

或在代码中解析成绝对路径：

```text
/home/windsky/project/Clutter/assets/references/grasp_ref_inspire.pkl
```

### Checkpoints

```text
/home/windsky/project/Demograsp/ckpt/fr3_dclaw.pt
-> /home/windsky/project/Clutter/assets/checkpoints/fr3_dclaw.pt

/home/windsky/project/Demograsp/ckpt/fr3_panda_gripper.pt
-> /home/windsky/project/Clutter/assets/checkpoints/fr3_panda_gripper.pt

/home/windsky/project/Demograsp/ckpt/fr3_shadow.pt
-> /home/windsky/project/Clutter/assets/checkpoints/fr3_shadow.pt

/home/windsky/project/Demograsp/ckpt/inspire.pt
-> /home/windsky/project/Clutter/assets/checkpoints/inspire.pt

/home/windsky/project/Demograsp/ckpt/shadow.pt
-> /home/windsky/project/Clutter/assets/checkpoints/shadow.pt

/home/windsky/project/Demograsp/ckpt/ur5_allegro.pt
-> /home/windsky/project/Clutter/assets/checkpoints/ur5_allegro.pt

/home/windsky/project/Demograsp/ckpt/ur5_svh.pt
-> /home/windsky/project/Clutter/assets/checkpoints/ur5_svh.pt
```

对应播放命令中的：

```text
checkpoint='ckpt/inspire.pt'
```

改成：

```text
--checkpoint /home/windsky/project/Clutter/assets/checkpoints/inspire.pt
```

或配置项：

```text
checkpoint: assets/checkpoints/inspire.pt
```

## 不需要复制的内容

```text
/home/windsky/project/Demograsp/.git
/home/windsky/project/Demograsp/docs/
/home/windsky/project/Demograsp/README.md
/home/windsky/project/Demograsp/train.sh
/home/windsky/project/Demograsp/play_policy.sh
/home/windsky/project/Demograsp/runs_ppo/
/home/windsky/project/Demograsp/videos/
/home/windsky/project/Demograsp/tasks/*.md
/home/windsky/project/Demograsp/文件夹说明.md
IsaacGym Preview 4 安装目录
IsaacGymEnvs 仓库
```

说明：

- `docs/` 是项目主页素材和视频，不是训练运行依赖。
- `README.md` 可以保留在 DemoGrasp 作为参考，但不必复制到 Clutter。
- `train.sh` 和 `play_policy.sh` 只作为参数参考，不建议作为最终入口。
- `runs_ppo/` 是训练输出，不属于源码。
- IsaacGym Preview 4 和 IsaacGymEnvs 是旧依赖，目标就是移除它们。

## 文件级迁移清单

| DemoGrasp 源路径 | Clutter 目标路径 | 是否复制 | 是否修改 | 说明 |
|---|---|---:|---:|---|
| `run_rl_grasp.py` | `scripts/train_ppo_onestep.py`, `scripts/play_ppo_onestep.py` | 参考 | 大改 | 改 AppLauncher + Gymnasium |
| `tasks/__init__.py` | `source/Clutter/Clutter/tasks/direct/clutter/__init__.py` | 参考 | 大改 | `isaacgym_task_map` 改 Gymnasium register |
| `tasks/grasp.py` | `source/Clutter/Clutter/tasks/direct/clutter/clutter_env.py` | 参考 | 重写 | `VecTask` 改 `DirectRLEnv` |
| `tasks/reward.py` | `source/Clutter/Clutter/tasks/direct/clutter/reward.py` | 是 | 基本不改 | 纯 torch |
| `tasks/utils.py` | `source/Clutter/Clutter/tasks/direct/clutter/utils.py` | 是 | 小改 | 替换 IsaacGymEnvs math 工具 |
| `tasks/config.yaml` | `clutter_env_cfg.py`, `ppo_onestep_cfg.py` | 参考 | 大改 | Hydra 配置拆分 |
| `tasks/task/grasp.yaml` | `clutter_env_cfg.py` | 参考 | 大改 | 环境配置改 configclass |
| `tasks/train/PPOOneStep.yaml` | `agents/ppo_onestep_cfg.py` | 是 | 小改 | PPO 参数保留 |
| `tasks/hand/*.yaml` | `tasks/direct/clutter/config/hand/*.yaml` | 是 | 小改 | 资产根路径由 Clutter 统一解析 |
| `algo/ppo_onestep/` | `source/Clutter/Clutter/algo/ppo_onestep/` | 是 | 小改 | gym 改 gymnasium，包路径改 Clutter |
| `algo/pn_utils/` | `source/Clutter/Clutter/algo/pn_utils/` | 是 | 小改 | 包路径改 Clutter |
| `data/dataset_utils.py` | `source/Clutter/Clutter/data/dataset_utils.py` | 可选 | 小改 | 仅数据采集需要 |
| `data/lerobot/` | `source/Clutter/Clutter/data/lerobot/` | 可选 | 小改 | 仅数据采集需要 |
| `assets/*` | `assets/*` | 是 | 可能 | URDF 可先用，后续可转 USD |
| `tasks/grasp_ref_*.pkl` | `assets/references/*.pkl` | 是 | 否 | reference 数据 |
| `ckpt/*.pt` | `assets/checkpoints/*.pt` | 是 | 否 | 预训练权重 |
| `docs/` | 无 | 否 | 否 | 主页素材 |
| `train.sh`, `play_policy.sh` | 无 | 否 | 否 | 只当参数参考 |

## 具体代码修改点

### PPO 环境返回值适配

DemoGrasp PPO 当前使用：

```python
obs, reward, reset, extras = self.vec_env.step(env_action)
```

IsaacLab 返回：

```python
obs_dict, reward, terminated, truncated, extras = env.step(actions)
```

需要改为：

```python
obs_dict, reward, terminated, truncated, extras = self.vec_env.step(env_action)
reset = terminated | truncated
obs = obs_dict["policy"]
```

如果用 wrapper，可以让 wrapper 返回旧格式：

```python
return {"obs": obs_dict["policy"]}, reward, reset, extras
```

### Observation key 适配

DemoGrasp 使用：

```text
obs_dict["obs"]
```

IsaacLab 推荐：

```text
obs_dict["policy"]
```

建议在环境内部使用 IsaacLab 风格：

```python
return {"policy": obs}
```

在 PPO wrapper 中映射：

```python
legacy_obs = {"obs": obs_dict["policy"]}
```

### 路径解析

不要在配置中写死 DemoGrasp 路径。

建议新增：

```text
/home/windsky/project/Clutter/source/Clutter/Clutter/utils/paths.py
```

内容逻辑：

```python
from pathlib import Path

CLUTTER_ROOT = Path(__file__).resolve().parents[4]
ASSET_ROOT = CLUTTER_ROOT / "assets"
REFERENCE_ROOT = ASSET_ROOT / "references"
CHECKPOINT_ROOT = ASSET_ROOT / "checkpoints"
```

这样所有路径都从 Clutter 自己解析。

### URDF 和 USD

第一阶段可以继续使用 URDF：

```python
sim_utils.UrdfFileCfg(
    asset_path=str(ASSET_ROOT / "inspire_tac/fr3_inspire_tac_L_right_safety.urdf"),
    fix_base=True,
)
```

后续建议把高频使用的机器人和物体资产转为 USD，提高加载稳定性和 IsaacLab 兼容性。

### Camera

DemoGrasp 旧相机逻辑使用：

```python
self.gym.create_camera_sensor
self.gym.attach_camera_to_body
self.gym.get_camera_image_gpu_tensor
```

IsaacLab 中应改为：

```text
isaaclab.sensors.Camera
isaaclab.sensors.TiledCamera
CameraCfg
```

建议第一阶段不迁移相机。等基础抓取环境跑通后，再恢复：

```text
render.enable
camera_1
camera_2
rgb
depth
seg
pcl
```

### Domain randomization

DemoGrasp 旧逻辑依赖 IsaacGymEnvs 的：

```python
self.apply_randomizations(self.randomization_params)
self.dr_randomizations
```

IsaacLab 中不能原样使用。建议拆成三类：

```text
reset randomization:
  object pose
  table height
  robot dof
  distractor objects

physics randomization:
  mass
  friction
  damping
  stiffness

render randomization:
  color
  texture
  light
  camera pose
```

第一阶段只保留 reset randomization。

## 推荐迁移顺序

### 阶段 1：最小可运行环境

只实现：

```text
机器人: fr3_inspire_tac
物体: union_ycb_unidex/union_ycb_debugset.yaml 中的少量对象
控制: qpos 或 pose 二选一，建议先 qpos
观测: armdof + handdof + eefpose + objpose
奖励: binary reward
相机: 关闭
点云: 关闭
干扰物: 关闭
纹理随机化: 关闭
```

目标：

```text
gymnasium.make("Clutter-Grasp-Direct-v0")
env.reset()
env.step(random_actions)
```

### 阶段 2：恢复 reference replay

迁移：

```text
generate_reaching_plan_idx
compute_reference_actions
tracking_reference pkl 加载
randomizeTrackingReference
randomizeGraspPose
```

目标：

```text
不用 PPO，也能 replay reference action。
```

### 阶段 3：接回 PPO one-step

迁移：

```text
algo/ppo_onestep
PointNet 可先不启用
train_ppo_onestep.py
play_ppo_onestep.py
```

目标：

```text
能加载 assets/checkpoints/inspire.pt
能运行 policy play
能开始训练新模型
```

### 阶段 4：恢复点云

迁移：

```text
enablePointCloud
load_object_point_clouds
transform_obj_pcl_2_world
objpcl observation
PointNet backbone
```

目标：

```text
observationType="eefpose+objinitpose+objpcl"
train.params.is_vision=True
```

### 阶段 5：恢复相机和数据采集

迁移：

```text
Camera / TiledCamera
compute_real_observation_dict
dataset_utils.py
collect_dataset.py
```

目标：

```text
能生成 rgb/depth 数据
能写 LeRobot 格式数据集
```

### 阶段 6：恢复全部手型和随机化

逐个恢复：

```text
fr3_inspire_tac
shadow_simple
fr3_shadow
ur5_allegro
ur5_svh
fr3_panda_gripper
fr3_dclaw_gripper
```

再恢复：

```text
distractor objects
multi-object full list
texture randomization
camera randomization
physics randomization
```

## 最小迁移命令映射参考

原训练命令片段：

```bash
python -u run_rl_grasp.py \
    task=grasp \
    train=PPOOneStep \
    hand=fr3_inspire_tac \
    num_envs=7000 \
    task.env.armController=pose \
    task.env.trackingReferenceFile=tasks/grasp_ref_inspire.pkl \
    task.env.enablePointCloud=True \
    train.params.is_vision=True
```

迁移后建议变成：

```bash
/home/windsky/isaaclab/isaaclab.sh -p /home/windsky/project/Clutter/scripts/train_ppo_onestep.py \
    --task Clutter-Grasp-Direct-v0 \
    --hand fr3_inspire_tac \
    --num_envs 7000 \
    --arm_controller pose \
    --tracking_reference assets/references/grasp_ref_inspire.pkl \
    --enable_point_cloud \
    --is_vision
```

原播放命令片段：

```bash
python -u run_rl_grasp.py \
    test=True \
    checkpoint='ckpt/inspire.pt'
```

迁移后建议变成：

```bash
/home/windsky/isaaclab/isaaclab.sh -p /home/windsky/project/Clutter/scripts/play_ppo_onestep.py \
    --task Clutter-Grasp-Direct-v0 \
    --hand fr3_inspire_tac \
    --checkpoint /home/windsky/project/Clutter/assets/checkpoints/inspire.pt
```

## 最重要的风险点

1. `tasks/grasp.py` 不是 import 替换能解决的，需要按 IsaacLab 生命周期重写。
2. DemoGrasp 的 `step()` 自己调用 `simulate/fetch_results`，IsaacLab 的 `DirectRLEnv` 已经管理仿真步进，不能双重 step。
3. 旧的 `gymtorch` tensor 访问方式不能继续用，要改成 `Articulation.data` 和 `RigidObject.data`。
4. 旧的 domain randomization 依赖 IsaacGymEnvs，必须拆开重做。
5. 旧 checkpoint 能否直接复用，取决于 observation 顺序、action 维度、action scaling 是否完全保持一致。
6. URDF 在 IsaacLab 中可以先用，但复杂 mesh、材质、相机和碰撞最好后续转 USD。
7. PPO 代码里有旧的 `gym.spaces` 和 `isaacgymenvs.utils.torch_jit_utils` import，需要清理。

## 一句话版本

复制资产、奖励、PPO、PointNet、hand 配置；不要复制 DemoGrasp 仓库目录；把 `tasks/grasp.py` 的任务逻辑迁进 Clutter 的 `DirectRLEnv`；把 `run_rl_grasp.py` 改成 IsaacLab `AppLauncher` 入口；所有路径都指向 `/home/windsky/project/Clutter` 内部。
