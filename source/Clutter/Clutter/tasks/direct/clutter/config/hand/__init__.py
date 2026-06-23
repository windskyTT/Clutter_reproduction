"""Python hand configs migrated from the legacy DemoGrasp YAML files."""

from __future__ import annotations

from .fr3_dclaw_gripper import HAND_CFG as FR3_DCLAW_GRIPPER_CFG
from .fr3_inspire_tac import HAND_CFG as FR3_INSPIRE_TAC_CFG
from .fr3_panda_gripper import HAND_CFG as FR3_PANDA_GRIPPER_CFG
from .fr3_shadow import HAND_CFG as FR3_SHADOW_CFG
from .hand_cfg import HandCfg
from .shadow_simple import HAND_CFG as SHADOW_SIMPLE_CFG
from .ur5_allegro import HAND_CFG as UR5_ALLEGRO_CFG
from .ur5_svh import HAND_CFG as UR5_SVH_CFG

HAND_CFGS: dict[str, HandCfg] = {
    FR3_DCLAW_GRIPPER_CFG.name: FR3_DCLAW_GRIPPER_CFG,
    FR3_INSPIRE_TAC_CFG.name: FR3_INSPIRE_TAC_CFG,
    FR3_PANDA_GRIPPER_CFG.name: FR3_PANDA_GRIPPER_CFG,
    FR3_SHADOW_CFG.name: FR3_SHADOW_CFG,
    SHADOW_SIMPLE_CFG.name: SHADOW_SIMPLE_CFG,
    UR5_ALLEGRO_CFG.name: UR5_ALLEGRO_CFG,
    UR5_SVH_CFG.name: UR5_SVH_CFG,
}

DEFAULT_HAND_CFG = FR3_INSPIRE_TAC_CFG

__all__ = [
    "DEFAULT_HAND_CFG",
    "FR3_DCLAW_GRIPPER_CFG",
    "FR3_INSPIRE_TAC_CFG",
    "FR3_PANDA_GRIPPER_CFG",
    "FR3_SHADOW_CFG",
    "HAND_CFGS",
    "HandCfg",
    "SHADOW_SIMPLE_CFG",
    "UR5_ALLEGRO_CFG",
    "UR5_SVH_CFG",
]

