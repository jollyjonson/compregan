import os
import sys


def debugger_connected() -> bool:
    return 'pydevd' in sys.modules


def run_through_pycharm() -> bool:
    return "PYCHARM_HOSTED" in os.environ
