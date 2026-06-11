"""Clutter 项目路径集中定义。

用途：
- 让训练/播放脚本统一解析项目根目录、assets、checkpoints、logs。
- 支持通过环境变量 `CLUTTER_ROOT` 覆盖项目根目录，方便从其他工作目录运行脚本。
"""

from __future__ import annotations

import os
from pathlib import Path


# 当前 Python 包目录：.../source/Clutter/Clutter
PACKAGE_DIR = Path(__file__).resolve().parents[1]
# extension 根目录：.../source/Clutter
EXTENSION_DIR = PACKAGE_DIR.parent
# 项目根目录：默认从包路径向上推到 /home/windsky/project/Clutter。
PROJECT_ROOT = Path(os.environ.get("CLUTTER_ROOT", PACKAGE_DIR.parents[2])).resolve()
# 资产目录：URDF、mesh、reference、checkpoint 都放在这里。
ASSETS_DIR = PROJECT_ROOT / "assets"
# 策略权重目录：播放脚本默认从这里读取 checkpoint。
CHECKPOINT_DIR = ASSETS_DIR / "checkpoints"
# 训练日志目录：train_ppo_onestep.py 会在这里创建 run 子目录。
LOGS_DIR = PROJECT_ROOT / "logs"

__all__ = ["PACKAGE_DIR", "EXTENSION_DIR", "PROJECT_ROOT", "ASSETS_DIR", "CHECKPOINT_DIR", "LOGS_DIR"]
