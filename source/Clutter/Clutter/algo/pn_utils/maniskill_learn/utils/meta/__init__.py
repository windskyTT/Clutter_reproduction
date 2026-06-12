"""Meta utilities used by the migrated ManiSkill-Learn PointNet code.

旧 ManiSkill-Learn 会在包入口直接导入 `Config`，而 `Config` 依赖第三方
`addict`。Clutter 当前只需要 PointNet 的 registry/build_from_cfg 路径，因此
这里把配置类改成可选导入：没有 `addict` 时仍允许 PointNet 正常导入。
"""

try:
    from .config import ConfigDict, Config, DictAction
except ModuleNotFoundError as exc:
    if exc.name != "addict":
        raise

    class ConfigDict(dict):
        """Minimal fallback for code paths that only need dict-like config access."""

        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as key_error:
                raise AttributeError(name) from key_error

        def __setattr__(self, name, value):
            self[name] = value

        def to_dict(self):
            return dict(self)

    class Config:
        """Placeholder that explains the optional dependency when full config parsing is used."""

        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("Config requires optional dependency `addict`.")

    class DictAction:
        """Placeholder for CLI config parsing, unused by the migrated PointNet backbone."""

        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("DictAction requires optional dependency `addict`.")

from .collect_env import collect_env
from .logger import get_logger, get_root_logger, print_log, flush_print
try:
    from .magic_utils import *
except ModuleNotFoundError as exc:
    if exc.name != "sorcery":
        raise
from .module_utils import (import_modules_from_strings, check_prerequisites, requires_package, requires_executable,
                           deprecated_api_warning)
from .path_utils import (is_filepath, fopen, check_file_exist, mkdir_or_exist, symlink, scandir, find_vcs_root,
                         get_filename, get_filename_suffix, copy_folder, copy_folders, add_suffix_to_filename,
                         get_dirname, to_abspath, replace_suffix)
from .process_utils import get_total_memory, get_memory_list, get_subprocess_ids, get_memory_dict
from .random_utils import set_random_seed
from .registry import Registry, build_from_cfg
from .timer import get_time_stamp, td_format
