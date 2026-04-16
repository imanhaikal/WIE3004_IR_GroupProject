# 📚 WIE3004 / WIE2005 Information Retrieval: System Evaluation Group Project

**Semester 2, 2025/2026** | **Instructor:** Prof. Ts. Dr. Sri Devi Ravana

## 📝 Project Overview
This repository contains the codebase for evaluating the performance of various Information Retrieval (IR) systems using the **TREC 8 Ad-hoc track** dataset. The project simulates search engine evaluation by computing performance metrics, analyzing ranking correlations, and conducting statistical significance tests across 10-15 provided system run files.

### 🎯 Key Objectives:
1. **Compute Evaluation Metrics:** Calculate Precision at depth 10 (P@10), Mean Average Precision (MAP), and at least one other metric (e.g., P@5, P@100, or NDCG) across 50 query topics.
2. **Correlation Coefficient Analysis:** Compare how system performance rankings change when using different evaluation metrics (e.g., P@10 vs. MAP@10).
3. **Significance Testing:** Perform paired one-sided Student's t-tests (or Wilcoxon signed-rank tests) to determine if the performance differences between pairs of systems are statistically significant (p-value < 0.05).

---

## 👥 Team Members & Roles
* **Member 1 [Karl]:** Data Engineer (Data Parsing & Cleaning)
* **Member 2 [Cheng Hooi]:** Metrics Lead A (P@10 & MAP Calculation)
* **Member 3 [Yuling]:** Metrics Lead B & Correlation Analyst
* **Member 4 [Iman]:** Statistician (Significance Testing)
* **Member 5 [Hani]:** Project Manager & Lead Editor

---

## ⚠️ Important Dataset Notice (DO NOT COMMIT DATA)
**The TREC dataset is subject to an institutional agreement and MUST NOT be shared publicly.** 
The `data/` folder is explicitly added to the `.gitignore` file. **Do not force-add or commit any `.txt` or run files to this repository.** Every member must download the dataset locally to their own machine.

**Where to get the data:**
1. **Qrels File:** Download the TREC 8 Ad-hoc Qrels file from [NIST](https://pages.nist.gov/trec-browser/trec8/adhoc/data/).
2. **System Runs:** Download our assigned group's folder containing 10-15 run files from the [Class Google Drive link provided by the Prof](https://drive.google.com/drive/folders/10Rsivn7bEGM7PcWseGx8dRbYjgEwHhkF?usp=drive_link).

Place all downloaded files into the `data/` folder locally:
```text
WIE3004-IR-Group-Project/
├── data/
│   ├── qrels.txt               # Relevance judgments
│   └── runs/                   # The 10-15 assigned system run files
```

---

## ⚙️ Local Setup Instructions

Follow these steps to set up the project on your local machine.

**1. Clone the repository:**
```bash
git clone https://github.com/imanhaikal/WIE3004_IR_GroupProject.git
cd WIE3004_IR_GroupProject
```

**2. Create a Virtual Environment:**
* **Windows:**
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```
* **Mac/Linux:**
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

**3. Install Dependencies:**
```bash
pip install -r requirements.txt
```

---

## 📁 Repository Structure

```text
WIE3004-IR-Group-Project/
│
├── data/                   # (IGNORED BY GIT) Put your Qrels and runs here
│   └── .gitkeep            # Keeps the empty folder in Git
│
├── notebooks/              # Jupyter notebooks for sandbox testing/EDA
│   └── data_cleaning.ipynb 
│
├── src/                    # Main modular Python scripts
│   ├── parser.py           # Functions to read TREC files into DataFrames
│   ├── metrics.py          # Functions for P@k, MAP, etc.
│   ├── stats.py            # Significance testing & Correlation functions
│   └── main.py             # Main execution script
│
├── requirements.txt        # Project dependencies (pandas, numpy, scipy)
└── README.md               # You are here!
```

---

## 📅 Project Timeline & Deadlines (Submission: June 11, 2026)

* **Phase 1: Data Preparation (Due Apr 30)** - Data ingestion, parsing formats, and handling missing topic noise.
* **Phase 2: Metrics Pipeline (Due May 14)** - Calculating per-topic scores and overall system averages for 3+ metrics.
* **Phase 3: Stats & Correlation (Due May 28)** - Generating rank correlation values and pairwise significance matrices (p < 0.05).
* **Phase 4: Documentation (Due Jun 4)** - Compiling score matrices and analyses into the final written report.
* **Phase 5: Presentation Prep (Due Jun 10)** - Finalising slides and rehearsing for the mandatory all-members online presentation.

---

## 🤝 Collaboration Guidelines
1. **Branching:** Please do not work directly on the `main` branch. Create a feature branch for your role:
   ```bash
   git checkout -b feature/data-parser
   ```
2. **Committing:** Write clear commit messages.
   ```bash
   git add .
   git commit -m "Added function to calculate MAP"
   git push origin feature/data-parser
   ```
3. **Merging:** Open a Pull Request (PR) on GitHub when your part is done so the team can review it before merging into `main`.
