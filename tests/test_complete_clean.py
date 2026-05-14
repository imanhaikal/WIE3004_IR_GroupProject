import json
import os
import importlib.util
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = PROJECT_ROOT / "src" / "complete_clean.py"

spec = importlib.util.spec_from_file_location("complete_clean", MODULE_PATH)
complete_clean = importlib.util.module_from_spec(spec)
sys.modules["complete_clean"] = complete_clean
spec.loader.exec_module(complete_clean)


def write_run(path, run_id, per_topic_counts):
    with open(path, "w", encoding="utf-8") as handle:
        for topic_id, count in per_topic_counts.items():
            for rank in range(1, count + 1):
                handle.write(
                    f"{topic_id} Q0 DOC-{run_id}-{topic_id}-{rank} "
                    f"{rank} {1000 - rank} {run_id}\n"
                )


class CompleteCleanPathTests(unittest.TestCase):
    def test_runs_directory_resolves_to_project_data_runs(self):
        runs_directory = Path(complete_clean.RUNS_DIRECTORY).resolve()

        self.assertEqual(PROJECT_ROOT / "data" / "runs", runs_directory)

    def test_notebook_config_resolves_project_root_from_notebooks_cwd(self):
        notebook_path = PROJECT_ROOT / "notebooks" / "Info_Retrieval.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        config_source = next(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
            and "RUNS_DIRECTORY" in "".join(cell["source"])
            and "PROJECT_ROOT" in "".join(cell["source"])
        )

        original_cwd = Path.cwd()
        namespace = {"Path": Path}
        try:
            os.chdir(PROJECT_ROOT / "notebooks")
            exec(config_source, namespace)
        finally:
            os.chdir(original_cwd)

        self.assertEqual(PROJECT_ROOT, namespace["PROJECT_ROOT"])
        self.assertEqual(PROJECT_ROOT / "data" / "runs", namespace["RUNS_DIRECTORY"])
        self.assertEqual("input.*", namespace["FILE_PATTERN"])

    def test_read_qrels_loads_standard_trec_qrels(self):
        with TemporaryDirectory() as tmp:
            qrels_path = Path(tmp) / "qrels.trec8.adhoc.parts1-5"
            qrels_path.write_text(
                "401 0 FBIS3-10009 0\n401 0 FBIS3-10059 1\n",
                encoding="utf-8",
            )

            qrels = complete_clean.read_qrels(qrels_path)

        self.assertEqual(["topic_id", "doc_id", "relevance"], list(qrels.columns))
        self.assertEqual(2, len(qrels))
        self.assertEqual("401", qrels.loc[0, "topic_id"])
        self.assertEqual(1, qrels.loc[1, "relevance"])

    def test_check_incomplete_runs_flags_missing_qrels_topic(self):
        with TemporaryDirectory() as tmp:
            runs_dir = Path(tmp)
            write_run(runs_dir / "input.missing", "missing", {"401": 3})

            original_threshold = complete_clean.DOC_THRESHOLD
            complete_clean.DOC_THRESHOLD = 3
            try:
                data = complete_clean.load_all_runs(runs_dir, "input.*")
                noisy = complete_clean.check_incomplete_runs(
                    data,
                    expected_topics={"401", "402"},
                )
            finally:
                complete_clean.DOC_THRESHOLD = original_threshold

        self.assertIn("missing", noisy.index)
        self.assertEqual(1, noisy.loc["missing", "topics_missing_docs"])

    def test_replace_noisy_runs_retains_loaded_backup_once(self):
        with TemporaryDirectory() as tmp:
            runs_dir = Path(tmp)
            write_run(runs_dir / "input.good", "good", {"401": 3, "402": 3})
            write_run(runs_dir / "input.noisy", "noisy", {"401": 1, "402": 3})
            write_run(runs_dir / "input.backup", "backup", {"401": 3, "402": 3})

            original_threshold = complete_clean.DOC_THRESHOLD
            complete_clean.DOC_THRESHOLD = 3
            try:
                all_runs = complete_clean.load_all_runs(runs_dir, "input.*")
                cleaned = complete_clean.replace_noisy_runs(
                    all_runs,
                    runs_dir,
                    {"noisy": "input.backup"},
                    expected_topics={"401", "402"},
                )
            finally:
                complete_clean.DOC_THRESHOLD = original_threshold

        self.assertNotIn("noisy", set(cleaned["run_id"]))
        self.assertEqual(1, cleaned[cleaned["run_id"] == "backup"]["run_id"].nunique())
        self.assertEqual({"good", "backup"}, set(cleaned["run_id"]))

    def test_replace_noisy_runs_loads_configured_backup_file(self):
        with TemporaryDirectory() as tmp:
            runs_dir = Path(tmp)
            write_run(runs_dir / "input.good", "good", {"401": 3, "402": 3})
            write_run(runs_dir / "input.noisy", "noisy", {"401": 1, "402": 3})
            write_run(runs_dir / "input.backup", "backup", {"401": 3, "402": 3})

            original_threshold = complete_clean.DOC_THRESHOLD
            complete_clean.DOC_THRESHOLD = 3
            try:
                primary = complete_clean.load_all_runs(runs_dir, "input.good")
                noisy = complete_clean.read_clean_trec(runs_dir / "input.noisy")
                primary = pd.concat([primary, noisy], ignore_index=True)
                cleaned = complete_clean.replace_noisy_runs(
                    primary,
                    runs_dir,
                    {"noisy": "input.backup"},
                    expected_topics={"401", "402"},
                )
            finally:
                complete_clean.DOC_THRESHOLD = original_threshold

        self.assertEqual({"good", "backup"}, set(cleaned["run_id"]))

    def test_replace_noisy_runs_fails_when_backup_file_missing(self):
        with TemporaryDirectory() as tmp:
            runs_dir = Path(tmp)
            write_run(runs_dir / "input.noisy", "noisy", {"401": 1, "402": 3})

            original_threshold = complete_clean.DOC_THRESHOLD
            complete_clean.DOC_THRESHOLD = 3
            try:
                primary = complete_clean.load_all_runs(runs_dir, "input.*")
                with self.assertRaisesRegex(FileNotFoundError, "Missing backup run file"):
                    complete_clean.replace_noisy_runs(
                        primary,
                        runs_dir,
                        {"noisy": "input.backup"},
                        expected_topics={"401", "402"},
                    )
            finally:
                complete_clean.DOC_THRESHOLD = original_threshold

    def test_replace_noisy_runs_rejects_extra_topic_before_filtering_noisy_run(self):
        with TemporaryDirectory() as tmp:
            runs_dir = Path(tmp)
            write_run(runs_dir / "input.noisy", "noisy", {"401": 1, "402": 3, "999": 3})
            write_run(runs_dir / "input.backup", "backup", {"401": 3, "402": 3})

            original_threshold = complete_clean.DOC_THRESHOLD
            complete_clean.DOC_THRESHOLD = 3
            try:
                data = complete_clean.load_all_runs(runs_dir, "input.*")
                with self.assertRaisesRegex(ValueError, "topics not present in qrels"):
                    complete_clean.replace_noisy_runs(
                        data,
                        runs_dir,
                        {"noisy": "input.backup"},
                        expected_topics={"401", "402"},
                    )
            finally:
                complete_clean.DOC_THRESHOLD = original_threshold

    def test_default_noisy_run_replacement_mapping_is_configured(self):
        self.assertEqual(
            {
                "isa50": "input.acsys8alo2",
                "mds08a1": "input.apl8ctd",
            },
            complete_clean.NOISY_RUN_REPLACEMENTS,
        )

    def test_export_data_writes_under_output_directory(self):
        with TemporaryDirectory() as tmp:
            original_output_directory = complete_clean.OUTPUT_DIRECTORY
            complete_clean.OUTPUT_DIRECTORY = Path(tmp) / "processed"
            try:
                complete_clean.export_data(
                    pd.DataFrame(
                        [
                            {
                                "topic_id": "401",
                                "doc_id": "DOC-1",
                                "rank": 1,
                                "score": 1.0,
                                "run_id": "run",
                            }
                        ]
                    )
                )
                exported = list(complete_clean.OUTPUT_DIRECTORY.glob("clean_group11_runs_*"))
            finally:
                complete_clean.OUTPUT_DIRECTORY = original_output_directory

        self.assertEqual(2, len(exported))
        self.assertTrue(all(path.parent.name == "processed" for path in exported))

    def test_validate_final_runs_accepts_complete_expected_systems(self):
        with TemporaryDirectory() as tmp:
            runs_dir = Path(tmp)
            write_run(runs_dir / "input.good", "good", {"401": 3, "402": 3})
            write_run(runs_dir / "input.backup", "backup", {"401": 3, "402": 3})

            original_threshold = complete_clean.DOC_THRESHOLD
            complete_clean.DOC_THRESHOLD = 3
            try:
                data = complete_clean.load_all_runs(runs_dir, "input.*")
                complete_clean.validate_final_runs(
                    data,
                    expected_topics={"401", "402"},
                    expected_system_count=2,
                )
            finally:
                complete_clean.DOC_THRESHOLD = original_threshold

    def test_validate_final_runs_rejects_extra_non_qrels_topic(self):
        with TemporaryDirectory() as tmp:
            runs_dir = Path(tmp)
            write_run(runs_dir / "input.extra", "extra", {"401": 3, "402": 3, "999": 3})

            original_threshold = complete_clean.DOC_THRESHOLD
            complete_clean.DOC_THRESHOLD = 3
            try:
                data = complete_clean.load_all_runs(runs_dir, "input.*")
                with self.assertRaisesRegex(ValueError, "topics not present in qrels"):
                    complete_clean.validate_final_runs(
                        data,
                        expected_topics={"401", "402"},
                        expected_system_count=1,
                    )
            finally:
                complete_clean.DOC_THRESHOLD = original_threshold

    def test_validate_final_runs_rejects_wrong_system_count(self):
        with TemporaryDirectory() as tmp:
            runs_dir = Path(tmp)
            write_run(runs_dir / "input.good", "good", {"401": 3, "402": 3})

            original_threshold = complete_clean.DOC_THRESHOLD
            complete_clean.DOC_THRESHOLD = 3
            try:
                data = complete_clean.load_all_runs(runs_dir, "input.*")
                with self.assertRaisesRegex(ValueError, "Expected 2 systems"):
                    complete_clean.validate_final_runs(
                        data,
                        expected_topics={"401", "402"},
                        expected_system_count=2,
                    )
            finally:
                complete_clean.DOC_THRESHOLD = original_threshold

    def test_validate_final_runs_rejects_incomplete_run(self):
        with TemporaryDirectory() as tmp:
            runs_dir = Path(tmp)
            write_run(runs_dir / "input.incomplete", "incomplete", {"401": 3, "402": 1})

            original_threshold = complete_clean.DOC_THRESHOLD
            complete_clean.DOC_THRESHOLD = 3
            try:
                data = complete_clean.load_all_runs(runs_dir, "input.*")
                with self.assertRaisesRegex(ValueError, "incomplete runs"):
                    complete_clean.validate_final_runs(
                        data,
                        expected_topics={"401", "402"},
                        expected_system_count=1,
                    )
            finally:
                complete_clean.DOC_THRESHOLD = original_threshold


if __name__ == "__main__":
    unittest.main()
