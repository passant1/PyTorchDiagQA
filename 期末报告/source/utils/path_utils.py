"""
路径工具模块
处理开发环境和 PyInstaller 打包环境的资源路径
"""
import os
import sys


def get_base_path():
    """
    获取项目根目录路径
    兼容开发环境和 PyInstaller 打包环境
    """
    if getattr(sys, 'frozen', False):
        # PyInstaller 打包后
        # sys._MEIPASS 指向 _internal 目录（onedir 模式）
        # 数据文件由 spec 的 datas 配置复制到 _internal/ 下
        base = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
        # 如果 _internal 下没有数据文件，回退到 exe 目录
        if not os.path.exists(os.path.join(base, 'data')):
            exe_dir = os.path.dirname(sys.executable)
            if os.path.exists(os.path.join(exe_dir, 'data')):
                return exe_dir
        return base
    else:
        # 开发环境：从 utils/path_utils.py 向上两级到项目根目录
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_data_path(filename):
    """获取 data/ 目录下文件的绝对路径"""
    return os.path.join(get_base_path(), "data", filename)


def get_config_path():
    """获取 config.yaml 的绝对路径"""
    return os.path.join(get_base_path(), "config.yaml")


def get_asset_path(filename):
    """获取 assets/ 目录下文件的绝对路径"""
    return os.path.join(get_base_path(), "assets", filename)


def get_cache_path(filename=""):
    """获取 data/cache/ 目录下文件的绝对路径"""
    cache_dir = os.path.join(get_base_path(), "data", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    if filename:
        return os.path.join(cache_dir, filename)
    return cache_dir
