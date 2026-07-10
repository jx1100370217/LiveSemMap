"""顶层运行配置 nav_config.yaml 的读取 (main.py / nav_web 共用)。

配置一次数据集, 三个入口脚本即可零参数运行; 命令行参数仍优先
(实现方式: 配置值作为 argparse 的 default)。
"""
import pathlib

import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "nav_config.yaml"


def load_run_config():
    """读 nav_config.yaml, 不存在或解析失败返回 {} (各脚本退回内置默认)。"""
    try:
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f) or {}
        if not isinstance(cfg, dict):
            return {}
        return cfg
    except FileNotFoundError:
        return {}
    except Exception as e:
        print(f"[run_config] {CONFIG_PATH} 解析失败, 忽略: {e}")
        return {}


def run_dir(cfg):
    """产物目录 logs/<save_as> (save_as 为 default 时为 logs/)。"""
    save_as = cfg.get("save_as", "default")
    p = REPO_ROOT / "logs"
    return p if save_as == "default" else p / save_as


def seq_name(cfg):
    """序列名 = 数据集目录名 (与 evaluate.prepare_savedir 的 stem 规则一致)。"""
    return pathlib.Path(cfg.get("dataset", "datasets/cfds_floor28")).stem
