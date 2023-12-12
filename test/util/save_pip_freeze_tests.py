import os
import unittest

from compregan.util.save_pip_freeze import save_pip_freeze_to_file


class PipFreezeSaverTests(unittest.TestCase):
    file_path = '.tmp_pip_freeze_test.txt'

    def tearDown(self) -> None:
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

    def test_running_pip_and_extracting_dependencies_works(self):
        save_pip_freeze_to_file(self.file_path)

        with open(self.file_path, 'r') as pip_file_handle:
            freeze_str = pip_file_handle.read()
            self.assertIn('tensorflow', freeze_str)


if __name__ == '__main__':
    unittest.main()
