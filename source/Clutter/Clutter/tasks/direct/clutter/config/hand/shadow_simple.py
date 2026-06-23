"""Standalone Shadow hand with a six-DoF virtual base."""

from __future__ import annotations

from .hand_cfg import HandCfg, joint_pos_from_order

ARM_DOF_NAMES = ("baseJX", "baseJY", "baseJZ", "baseJROLL", "baseJPITCH", "baseJYAW")

ACTIVE_HAND_DOF_NAMES = (
    "rh_FFJ4",
    "rh_FFJ3",
    "rh_FFJ2",
    "rh_MFJ4",
    "rh_MFJ3",
    "rh_MFJ2",
    "rh_RFJ4",
    "rh_RFJ3",
    "rh_RFJ2",
    "rh_LFJ5",
    "rh_LFJ4",
    "rh_LFJ3",
    "rh_LFJ2",
    "rh_THJ5",
    "rh_THJ4",
    "rh_THJ3",
    "rh_THJ2",
    "rh_THJ1",
)

PASSIVE_JOINT_MIMIC = {
    "rh_FFJ1": ("rh_FFJ2", 1.0),
    "rh_MFJ1": ("rh_MFJ2", 1.0),
    "rh_RFJ1": ("rh_RFJ2", 1.0),
    "rh_LFJ1": ("rh_LFJ2", 1.0),
}

DEFAULT_JOINT_ORDER = ARM_DOF_NAMES + ACTIVE_HAND_DOF_NAMES + tuple(PASSIVE_JOINT_MIMIC)

HAND_CFG = HandCfg(
    name="shadow_simple",
    robot_asset_file="shadow_hand_simple/right_with_base.usd",
    robot_asset_file_visual_realistic="shadow_hand_simple/right_with_base.usd",
    arm_dof_names=ARM_DOF_NAMES,
    active_hand_dof_names=ACTIVE_HAND_DOF_NAMES,
    passive_joint_mimic=PASSIVE_JOINT_MIMIC,
    eef_link_name="wrist",
    palm_link_name="wrist",
    fingertip_link_names=("rh_thdistal", "rh_ffdistal", "rh_mfdistal", "rh_rfdistal", "rh_lfdistal"),
    palm_offset=(0.0, 0.0, 0.0),
    default_joint_pos=joint_pos_from_order(
        DEFAULT_JOINT_ORDER,
        (0.5, -0.1, 0.4, 0.0, 1.57, 0.0) + (0.0,) * 22,
    ),
    num_actions=25,
    hand_dof_start_idx=7,
)

__all__ = ["HAND_CFG"]
