import os
import subprocess
from typing import Union


def save_pip_freeze_to_file(file_path: Union[os.PathLike, str]) -> None:
    with subprocess.Popen(['pip freeze'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True) as pip_proc:
        pip_freeze_output = pip_proc.stdout.read().decode('utf-8')
        pip_freeze_err = pip_proc.stderr.read().decode('utf-8')
    assert pip_freeze_err == ''
    with open(file_path, 'w') as file_handle:
        file_handle.write(pip_freeze_output)

