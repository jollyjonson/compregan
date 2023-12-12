import os
import shutil
from typing import Union


def get_package_dir() -> Union[str, os.PathLike]:
    return os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def get_tmp_file_dir() -> Union[str, os.PathLike]:
    tmp_file_dir_path = os.path.join(get_package_dir(), 'experiments', 'tmp')
    if not os.path.isdir(tmp_file_dir_path):
        os.mkdir(tmp_file_dir_path)
    return tmp_file_dir_path


def cleanup_tmp_dir() -> None:
    shutil.rmtree(get_tmp_file_dir())
