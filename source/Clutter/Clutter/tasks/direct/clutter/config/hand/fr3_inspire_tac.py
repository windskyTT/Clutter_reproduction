"""FR3 arm with Inspire tactile hand."""

from __future__ import annotations

from .hand_cfg import HandCfg, joint_pos_from_order

ARM_DOF_NAMES = (
    "fr3_joint1",
    "fr3_joint2",
    "fr3_joint3",
    "fr3_joint4",
    "fr3_joint5",
    "fr3_joint6",
    "fr3_joint7",
)

ACTIVE_HAND_DOF_NAMES = (
    "right_little_1_joint",
    "right_ring_1_joint",
    "right_middle_1_joint",
    "right_index_1_joint",
    "right_thumb_2_joint",
    "right_thumb_1_joint",
)

PASSIVE_JOINT_MIMIC = {
    "right_thumb_3_joint": ("right_thumb_2_joint", 0.6),
    "right_thumb_4_joint": ("right_thumb_2_joint", 0.8),
    "right_index_2_joint": ("right_index_1_joint", 1.05),
    "right_middle_2_joint": ("right_middle_1_joint", 1.05),
    "right_ring_2_joint": ("right_ring_1_joint", 1.05),
    "right_little_2_joint": ("right_little_1_joint", 1.18),
}

DEFAULT_JOINT_ORDER = ARM_DOF_NAMES + (
    "right_thumb_1_joint",
    "right_thumb_2_joint",
    "right_thumb_3_joint",
    "right_thumb_4_joint",
    "right_index_1_joint",
    "right_index_2_joint",
    "right_middle_1_joint",
    "right_middle_2_joint",
    "right_ring_1_joint",
    "right_ring_2_joint",
    "right_little_1_joint",
    "right_little_2_joint",
)

HAND_CFG = HandCfg(
    name="fr3_inspire_tac",
    robot_asset_file="inspire_tac/fr3_inspire_tac_L_right_safety.usd",
    robot_asset_file_visual_realistic="inspire_tac/fr3_inspire_tac_L_right_safety_visual_realistic.usd",
    arm_dof_names=ARM_DOF_NAMES,
    active_hand_dof_names=ACTIVE_HAND_DOF_NAMES,
    passive_joint_mimic=PASSIVE_JOINT_MIMIC,
    eef_link_name="fr3_link8",
    palm_link_name="base_link",
    fingertip_link_names=(
        "right_thumb_4",
        "right_index_2",
        "right_middle_2",
        "right_ring_2",
        "right_little_2",
    ),
    palm_offset=(0.0, 0.02, -0.05),
    default_joint_pos=joint_pos_from_order(
        DEFAULT_JOINT_ORDER,
        (0.0, 0.0, 0.0, -1.6, 0.0, 1.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ),
    num_actions=13,
)

__all__ = ["HAND_CFG"]
