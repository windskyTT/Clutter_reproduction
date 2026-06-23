"""Robot hand configuration helpers used by the Clutter DirectRLEnv.

The old DemoGrasp hand files were YAML snippets consumed by IsaacGym code.
IsaacLab prefers Python config objects, so this module stores the semantic
hand data and can materialize an :class:`ArticulationCfg` when the simulator
imports are available.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from Clutter.utils.paths import ASSET_USD_ROOT


def joint_pos_from_order(joint_names: Sequence[str], values: Sequence[float]) -> dict[str, float]:
    """Build IsaacLab's joint position dictionary from an ordered value list."""

    if len(joint_names) != len(values):
        raise ValueError(f"joint_names has {len(joint_names)} entries but values has {len(values)} entries.")
    return {joint_name: float(value) for joint_name, value in zip(joint_names, values, strict=True)}


def resolve_usd(relative_path: str) -> Path:
    """Resolve a robot USD path below ``assets_usd``."""

    return ASSET_USD_ROOT / relative_path


@dataclass(frozen=True)
class HandCfg:
    """IsaacLab-facing description of one robot hand setup.

    ``make_articulation_cfg`` imports IsaacLab lazily so the data modules can be
    statically checked in a normal Python environment without launching Isaac Sim.
    """

    name: str
    robot_asset_file: str
    robot_asset_file_visual_realistic: str
    arm_dof_names: tuple[str, ...]
    active_hand_dof_names: tuple[str, ...]
    passive_joint_mimic: Mapping[str, tuple[str, float]]
    eef_link_name: str
    palm_link_name: str
    fingertip_link_names: tuple[str, ...]
    palm_offset: tuple[float, float, float]
    default_joint_pos: Mapping[str, float]
    num_actions: int
    hand_dof_start_idx: int | None = None
    fix_root_link: bool = True
    stiffness: float = 400.0
    damping: float = 40.0
    effort_limit_sim: float = 200.0
    velocity_limit_sim: float = 8.0
    init_pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    init_rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def robot_asset_path(self) -> Path:
        """Absolute USD path used for physics/runtime loading."""

        return resolve_usd(self.robot_asset_file)

    @property
    def robot_asset_visual_realistic_path(self) -> Path:
        """Absolute USD path used when visual-realistic rendering is selected."""

        return resolve_usd(self.robot_asset_file_visual_realistic)

    @property
    def num_arm_dofs(self) -> int:
        return len(self.arm_dof_names)

    @property
    def num_active_hand_dofs(self) -> int:
        return len(self.active_hand_dof_names)

    @property
    def default_hand_dof_start_idx(self) -> int:
        return self.num_arm_dofs if self.hand_dof_start_idx is None else self.hand_dof_start_idx

    def make_articulation_cfg(self, *, prim_path: str, activate_contact_sensors: bool = False):
        """Create the IsaacLab ``ArticulationCfg`` for this robot USD."""

        import isaaclab.sim as sim_utils
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import ArticulationCfg

        return ArticulationCfg(
            prim_path=prim_path,
            spawn=sim_utils.UsdFileCfg(
                usd_path=str(self.robot_asset_path),
                activate_contact_sensors=activate_contact_sensors,
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False,
                    fix_root_link=self.fix_root_link,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=self.init_pos,
                rot=self.init_rot,
                joint_pos=dict(self.default_joint_pos),
            ),
            actuators={
                "all_joints": ImplicitActuatorCfg(
                    joint_names_expr=[".*"],
                    stiffness=self.stiffness,
                    damping=self.damping,
                    effort_limit_sim=self.effort_limit_sim,
                    velocity_limit_sim=self.velocity_limit_sim,
                ),
            },
        )
