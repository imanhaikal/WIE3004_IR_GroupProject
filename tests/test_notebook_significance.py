import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "src" / "Info_Retrieval (full with cleaning & precision & MAP).ipynb"


class NotebookSignificanceTests(unittest.TestCase):
    def test_notebook_contains_pairwise_significance_section(self):
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        sources = ["".join(cell.get("source", [])) for cell in notebook["cells"]]

        self.assertTrue(
            any("### 3.5 Pairwise Significance Testing" in source for source in sources)
        )
        self.assertTrue(any("run_significance_analysis" in source for source in sources))
        self.assertTrue(any("def prepare_score_matrix" in source for source in sources))
        self.assertTrue(any("def calculate_pairwise_significance" in source for source in sources))
        self.assertTrue(any("def create_p_value_matrix" in source for source in sources))
        self.assertFalse(any("from stats import" in source for source in sources))
        self.assertTrue(any("precision_at_10_matrix.csv" in source for source in sources))
        self.assertTrue(any("MAP_at_10.csv" in source for source in sources))
        self.assertTrue(any("NDCG_at_10.csv" in source for source in sources))


if __name__ == "__main__":
    unittest.main()
