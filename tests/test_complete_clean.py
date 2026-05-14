import importlib.util
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "src" / "complete_clean.py"

spec = importlib.util.spec_from_file_location("complete_clean", MODULE_PATH)
complete_clean = importlib.util.module_from_spec(spec)
sys.modules["complete_clean"] = complete_clean
spec.loader.exec_module(complete_clean)


class CompleteCleanPathTests(unittest.TestCase):
    def test_runs_directory_resolves_to_project_data_runs(self):
        runs_directory = Path(complete_clean.RUNS_DIRECTORY).resolve()

        self.assertEqual(PROJECT_ROOT / "data" / "runs", runs_directory)
        self.assertGreater(
            len(list(runs_directory.glob(complete_clean.FILE_PATTERN))),
            0,
            "Expected bundled TREC run files to be discoverable from the configured runs directory.",
        )


if __name__ == "__main__":
    unittest.main()
