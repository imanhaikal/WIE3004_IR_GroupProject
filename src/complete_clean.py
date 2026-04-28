"""
TREC Run File Processing and Validation
-----------------------------------------------
This script:
1. Loads multiple TREC run files from a directory
2. Cleans and standardizes the data format
3. Validates document counts and structure
4. Detects noisy systems (missing documents)
5. Exports a clean dataset for evaluation

"""

import pandas as pd
import os
import glob
from datetime import datetime

# ==============================
# CONFIGURATION
# ==============================

RUNS_DIRECTORY = "C:/Users/User/Desktop/IR assignment/WIE3004_IR_GroupProject/set11"
FILE_PATTERN = "*"
DOC_THRESHOLD = 950      # Less than 950 considered noisy


# ==============================
# DATA LOADING & CLEANING
# ==============================

def read_clean_trec(file_path):
    """
    Steps:
    - Assign standard column names
    - Convert rank and score to numeric
    - Drop unnecessary 'Q0' column
    """

    column_names = ["topic_id", "Q0", "doc_id", "rank", "score", "run_id"]

    try:
        df = pd.read_csv(
            file_path,
            sep=r'\s+',
            engine='python',
            names=column_names,
            dtype=str,
        )
    except Exception as e:
        print(f"ERROR reading {file_path}: {e}")
        return pd.DataFrame()

    # Validate column structure
    if df.shape[1] != 6:
        print(f"WARNING: Unexpected column format in {file_path}")
        return pd.DataFrame()

    # Convert numeric fields
    df['rank'] = pd.to_numeric(df['rank'], errors='coerce')
    df['score'] = pd.to_numeric(df['score'], errors='coerce')

    # Check for invalid numeric values
    if df[['rank', 'score']].isna().any().any():
        print(f"WARNING: Invalid numeric values detected in {file_path}")

    # Drop unused column
    df = df.drop(columns=['Q0'])

    return df


def load_all_runs(directory, pattern):
    """
    Loads all files and combine into one DataFrame
    """

    file_paths = glob.glob(os.path.join(directory, pattern))

    if not file_paths:
        raise ValueError("No files found in the specified directory.")

    print(f"Found {len(file_paths)} files. Loading...")

    dataframes = [read_clean_trec(f) for f in file_paths]
    combined_df = pd.concat(dataframes, ignore_index=True)

    # Global sorting (efficient)
    combined_df = combined_df.sort_values(
        by=['run_id', 'topic_id', 'score', 'doc_id'],
        ascending=[True, True, False, True]
    )

    return combined_df


# ==============================
# VALIDATION & DIAGNOSTICS
# ==============================

def check_incomplete_runs(df):
    """
    Identify systems that retrieve fewer than 950 documents per topic
    """

    document_counts = (
        df.groupby(['run_id', 'topic_id'])
        .size()
        .reset_index(name='doc_count')
    )

    noisy_systems = (
        document_counts.groupby('run_id')
        .agg(
            avg_docs=('doc_count', 'mean'),
            min_docs=('doc_count', 'min'),
            topics_missing_docs=('doc_count', lambda x: (x < DOC_THRESHOLD).sum())
        )
        .loc[lambda x: x['topics_missing_docs'] > 0]
    )

    return noisy_systems


def check_duplicates(df):

    duplicates = df.duplicated(subset=['run_id', 'topic_id', 'doc_id'])

    if duplicates.any():
        print("WARNING: Duplicate documents detected!")
        print(df[duplicates].head())
    else:
        print("OK: No duplicate documents found.")


def inspect_file(directory, file_name):

    full_path = os.path.join(directory, file_name)
    print("\n" + "=" * 50)
    print(f"INSPECTING: {file_name}")

    if not os.path.exists(full_path):
        print("ERROR: File not found.")
        return

    df = read_clean_trec(full_path)

    if df.empty:
        print("ERROR: File could not be processed.")
        return

    # Document count per topic
    topic_counts = df.groupby('topic_id').size().reset_index(name='doc_count')

    broken_topics = topic_counts[topic_counts['doc_count'] < DOC_THRESHOLD]

    print("\n[1] DOCUMENT COUNT CHECK")
    if not broken_topics.empty:
        print(broken_topics)
    else:
        print("OK: All topics meet document threshold.")

    # Topic ID format check (401–450)
    pattern = r'^40[1-9]$|^4[1-4][0-9]$|^450$'
    malformed = df[~df['topic_id'].str.match(pattern, na=False)]

    print("\n[2] FORMAT CHECK")
    if not malformed.empty:
        print("WARNING: Malformed topic IDs detected")
        print(malformed.head())
    else:
        print("OK: Topic IDs are correctly formatted.")


# ==============================
# EXPORT FUNCTION
# ==============================

def export_data(df):
    """
    Export cleaned dataset to CSV and Pickle formats with timestamp.
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_name = f"clean_group11_runs_{timestamp}.csv"
    pkl_name = f"clean_group11_runs_{timestamp}.pkl"

    df.to_csv(csv_name, index=False)
    df.to_pickle(pkl_name)

    print(f"Data exported:\n- {csv_name}\n- {pkl_name}")


# ==============================
# MAIN
# ==============================

def main():

    print("=== TREC DATA PROCESSING START ===")

    # Step 1: Load data
    all_runs_data = load_all_runs(RUNS_DIRECTORY, FILE_PATTERN)

    print(f"\nTotal rows loaded: {len(all_runs_data)}")

    # Step 2: Validate structure
    print("\n--- DATA STRUCTURE ---")
    print(all_runs_data.info())

    # Step 3: Check missing values
    na_total = all_runs_data.isna().sum().sum()
    print(f"\nMissing values: {na_total}")

    # Step 4: Duplicate check
    check_duplicates(all_runs_data)

    # Step 5: Noisy system detection
    print("\n--- NOISY SYSTEM CHECK ---")
    noisy = check_incomplete_runs(all_runs_data)

    if not noisy.empty:
        print("WARNING: Noisy systems detected")
        print(noisy)

        # REMOVE noisy systems
        noisy_run_ids = noisy.index.tolist()
        clean_data = all_runs_data[~all_runs_data['run_id'].isin(noisy_run_ids)]

        print(f"\nRemoved {len(noisy_run_ids)} noisy systems")
    else:
        print("OK: All systems meet document requirements.")
        clean_data = all_runs_data

    # Step 6: Export clean dataset
    export_data(clean_data)

    print("\n=== PROCESSING COMPLETE ===")


# ==============================
# RUN SCRIPT
# ==============================

if __name__ == "__main__":
    main()

    # Optional: Inspect specific  files
    inspect_file(RUNS_DIRECTORY, "input.mds08a1")
    inspect_file(RUNS_DIRECTORY, "input.isa50")