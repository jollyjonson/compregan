import os
import subprocess


def codebase_is_modified_according_to_git():
    module_directory = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    with subprocess.Popen(['git', 'status'], stdout=subprocess.PIPE, cwd=module_directory) as proc:
        return 'modified' in proc.stdout.read().decode('utf-8')
