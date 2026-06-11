# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Clutter extension 的 Python 打包脚本。

用途：
- 让 IsaacLab 或 `pip install -e source/Clutter` 能安装/发现 `Clutter` 包。
- 读取 `config/extension.toml` 中的扩展元信息，避免版本、作者等信息重复维护。
- 使用 `find_packages()` 包含迁移后的子包，例如 `Clutter.algo.ppo_onestep`。
"""

import os

import toml
from setuptools import find_packages, setup

# extension.toml 与 setup.py 位于同一个 extension 根目录下。
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# 读取 extension 元数据，用于填充 setup() 的包信息。
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# 最小 Python 依赖。IsaacLab/Isaac Sim 相关依赖由外部环境提供，不在这里安装。
INSTALL_REQUIRES = [
    # NOTE: Add dependencies
    "psutil",
]

# setuptools 打包配置。IsaacLab extension 加载和 editable install 都会用到这里。
setup(
    name="Clutter",
    # Include migrated subpackages such as Clutter.algo.ppo_onestep.
    packages=find_packages(),
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    license="Apache-2.0",
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Isaac Sim :: 4.5.0",
        "Isaac Sim :: 5.0.0",
        "Isaac Sim :: 5.1.0",
    ],
    zip_safe=False,
)
