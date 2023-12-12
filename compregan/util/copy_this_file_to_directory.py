import os
import shutil
import traceback
from typing import Union


def copy_this_file_to_directory(directory: Union[str, os.PathLike]) -> None:
    assert os.path.isdir(directory)
    calling_file_path = traceback.extract_stack()[-2][0]
    assert os.path.exists(calling_file_path)
    shutil.copy(
        calling_file_path,
        os.path.join(directory, os.path.basename(calling_file_path)),
    )
