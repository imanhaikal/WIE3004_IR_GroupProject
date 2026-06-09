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
from pathlib import Path

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


def read_qrels(file_path):
    """
    Load standard TREC qrels into topic/document/relevance columns.
    """

    column_names = ["topic_id", "ignore", "doc_id", "relevance"]

    qrels = pd.read_csv(
        file_path,
        sep=r'\s+',
        engine='python',
        names=column_names,
        dtype={
            "topic_id": str,
            "ignore": str,
            "doc_id": str,
            "relevance": str,
        },
    )

    qrels = qrels.drop(columns=["ignore"])
    qrels["relevance"] = (
        pd.to_numeric(qrels["relevance"], errors="coerce")
        .fillna(0)
        .astype(int)
    )

    return qrels


def get_expected_topics(qrels):
    topics = set(qrels["topic_id"].astype(str).unique())
    if not topics:
        raise ValueError("Qrels file contains no topics")
    return topics


def load_all_runs(directory, pattern):
    """
    Loads all files and combine into one DataFrame
    """

    directory = Path(directory)
    file_paths = sorted(
        Path(path)
        for path in glob.glob(str(directory / pattern))
        if Path(path).is_file()
    )

    if not file_paths:
        raise ValueError("No files found in the specified directory.")

    print(f"Found {len(file_paths)} files. Loading...")

    dataframes = [read_clean_trec(f) for f in file_paths]
    dataframes = [df for df in dataframes if not df.empty]

    if not dataframes:
        raise ValueError("No valid run files could be loaded.")

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

def check_incomplete_runs(df, expected_topics=None):
    """
    Identify systems that retrieve fewer than DOC_THRESHOLD documents per topic.
    """

    document_counts = (
        df.groupby(['run_id', 'topic_id'])
        .size()
        .rename('doc_count')
    )

    if expected_topics is not None:
        run_ids = sorted(df['run_id'].dropna().unique())
        expected_index = pd.MultiIndex.from_product(
            [run_ids, sorted(str(topic) for topic in expected_topics)],
            names=['run_id', 'topic_id'],
        )
        document_counts = document_counts.reindex(expected_index, fill_value=0)

    document_counts = document_counts.reset_index()

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


def validate_topic_ids(df, expected_topics):
    if expected_topics is None:
        return

    expected_topics = {str(topic) for topic in expected_topics}
    actual_topics = set(df['topic_id'].astype(str).unique())
    extra_topics = actual_topics - expected_topics

    if extra_topics:
        raise ValueError(
            "Run data contains topics not present in qrels: "
            + ", ".join(sorted(extra_topics))
        )


def validate_final_runs(df, expected_topics=None, expected_system_count=None):
    if expected_system_count is None:
        expected_system_count = EXPECTED_SYSTEM_COUNT

    validate_topic_ids(df, expected_topics)

    noisy = check_incomplete_runs(df, expected_topics)
    if not noisy.empty:
        raise ValueError("Final dataset still contains incomplete runs:\n" + str(noisy))

    system_count = df['run_id'].nunique()
    if system_count != expected_system_count:
        raise ValueError(f"Expected {expected_system_count} systems, found {system_count}")


def load_backup_run(directory, backup_file):
    backup_path = Path(directory) / backup_file
    if not backup_path.exists():
        raise FileNotFoundError(f"Missing backup run file: {backup_path}")

    backup_data = read_clean_trec(backup_path)
    required_columns = {'topic_id', 'doc_id', 'rank', 'score', 'run_id'}

    if backup_data.empty or not required_columns.issubset(backup_data.columns):
        raise ValueError(f"Backup run file could not be loaded: {backup_path}")

    return backup_data


def replace_noisy_runs(df, directory, replacements=None, expected_topics=None):
    replacements = replacements or NOISY_RUN_REPLACEMENTS
    validate_topic_ids(df, expected_topics)
    noisy = check_incomplete_runs(df, expected_topics)

    if noisy.empty:
        return df

    noisy_run_ids = noisy.index.tolist()
    missing_mapping = [run_id for run_id in noisy_run_ids if run_id not in replacements]

    if missing_mapping:
        raise ValueError("No backup configured for noisy run(s): " + ", ".join(missing_mapping))

    cleaned = df[~df['run_id'].isin(noisy_run_ids)].copy()
    retained_run_ids = set(cleaned['run_id'].unique())
    backup_frames = []

    for noisy_run_id in noisy_run_ids:
        backup_file = replacements[noisy_run_id]
        backup_data = load_backup_run(directory, backup_file)
        backup_run_ids = set(backup_data['run_id'].unique())

        if len(backup_run_ids) != 1:
            raise ValueError(f"Backup file {backup_file} contains multiple run IDs: {backup_run_ids}")

        validate_topic_ids(backup_data, expected_topics)

        backup_run_id = next(iter(backup_run_ids))
        if backup_run_id in retained_run_ids:
            continue

        backup_noisy = check_incomplete_runs(backup_data, expected_topics)
        if not backup_noisy.empty:
            raise ValueError("Backup run file is incomplete:\n" + str(backup_noisy))

        backup_frames.append(backup_data)
        retained_run_ids.add(backup_run_id)

    if backup_frames:
        cleaned = pd.concat([cleaned, *backup_frames], ignore_index=True)

    return cleaned


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

    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_DIRECTORY / f"clean_group11_runs_{timestamp}.csv"
    pkl_path = OUTPUT_DIRECTORY / f"clean_group11_runs_{timestamp}.pkl"

    df.to_csv(csv_path, index=False)
    df.to_pickle(pkl_path)

    print(f"Data exported:\n- {csv_path}\n- {pkl_path}")


# ==============================
# MAIN
# ==============================

def main():

    print("=== TREC DATA PROCESSING START ===")

    # Step 1: Load qrels
    qrels = read_qrels(QRELS_FILE)
    expected_topics = get_expected_topics(qrels)
    print(f"Qrels loaded: {len(qrels)} rows across {len(expected_topics)} topics")

    # Step 2: Load data
    all_runs_data = load_all_runs(RUNS_DIRECTORY, FILE_PATTERN)

    print(f"\nTotal rows loaded: {len(all_runs_data)}")

    # Step 3: Validate structure
    print("\n--- DATA STRUCTURE ---")
    print(all_runs_data.info())

    # Step 4: Check missing values
    na_total = all_runs_data.isna().sum().sum()
    print(f"\nMissing values: {na_total}")

    # Step 5: Duplicate check
    check_duplicates(all_runs_data)

    # Step 6: Noisy system detection and replacement
    print("\n--- NOISY SYSTEM CHECK ---")
    noisy = check_incomplete_runs(all_runs_data, expected_topics)

    if not noisy.empty:
        print("WARNING: Noisy systems detected")
        print(noisy)
        clean_data = replace_noisy_runs(
            all_runs_data,
            RUNS_DIRECTORY,
            expected_topics=expected_topics,
        )
        print(f"\nReplaced {len(noisy)} noisy systems where backups were required")
    else:
        print("OK: All systems meet document requirements.")
        clean_data = all_runs_data

    # Step 7: Validate final dataset
    validate_final_runs(clean_data, expected_topics=expected_topics)
    print(f"Final validation passed: {clean_data['run_id'].nunique()} systems")

    # Step 8: Export clean dataset
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
