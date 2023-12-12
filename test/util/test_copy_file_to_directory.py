import os
import shutil
import subprocess
import unittest


class CopyThisFileToDirectoryTest(unittest.TestCase):
    def test_copying_works_as_expected(self):
        tmp_dir_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "tmpCopyThisFileToDirectoryTest_test_copying_works_as_expected",
        )
        os.mkdir(tmp_dir_path)
        try:
            dir_inside_tmp_dir = os.path.join(tmp_dir_path, "tmp")
            os.mkdir(dir_inside_tmp_dir)

            file_name = "my_file.py"
            file_content = (
                f"from compregan.util import copy_this_file_to_directory\n"
                f"copy_this_file_to_directory('{dir_inside_tmp_dir}')"
            )
            file_path_before_copying = os.path.join(tmp_dir_path, file_name)
            with open(file_path_before_copying, "w") as file_handle:
                file_handle.write(file_content)

            with subprocess.Popen(
                ["python", file_path_before_copying]
            ) as proc:
                proc.wait()
                self.assertEqual(0, proc.returncode)

            file_path_after_copying = os.path.join(
                dir_inside_tmp_dir, file_name
            )

            self.assertTrue(os.path.exists(file_path_before_copying))
            self.assertTrue(os.path.exists(file_path_after_copying))

            with open(file_path_after_copying, "r") as file_handle:
                self.assertEqual(file_handle.read(), file_content)

        except Exception as e:
            raise e

        finally:
            shutil.rmtree(tmp_dir_path)


if __name__ == "__main__":
    unittest.main()
