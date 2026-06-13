"""Utility script to verify the integrity of synthesized datasets and find bugs."""
import argparse
import pandas as pd
import os
import sys


def main():
    """Verifies the integrity of a synthesized Parquet dataset to find data poisoning bugs."""
    parser = argparse.ArgumentParser(description="Verify a synthesized Parquet dataset.")
    parser.add_argument("file_path", type=str, help="Path to the .parquet file")
    parser.add_argument("--fix", action="store_true", help="Remove poisoned pairs and overwrite the file")
    args = parser.parse_args()

    if not os.path.exists(args.file_path):
        print(f"Error: File not found at {args.file_path}")
        sys.exit(1)

    print(f"\nAnalyzing dataset: {args.file_path}...")
    df = pd.read_parquet(args.file_path)

    total_rows = len(df)
    hard_negatives = df[(df['label'] == 0) & (~df.duplicated(subset=['pair_id'], keep=False))]

    pairs_df = df[df.duplicated(subset=['pair_id'], keep=False)]

    orig = pairs_df[pairs_df['label'] == 0].set_index('pair_id')
    corr = pairs_df[pairs_df['label'] == 1].set_index('pair_id')

    merged = orig.join(corr, lsuffix='_orig', rsuffix='_corr', how='inner')

    poisoned = merged[merged['sentence_orig'] == merged['sentence_corr']]

    print("\n=== Dataset Integrity Stats ===")
    print(f"Total Rows: {total_rows}")
    print(f"Hard Negatives (Identity Translations): {len(hard_negatives)}")
    print(f"Total Synthesized Pairs: {len(merged)}")

    print("\n=== Bug Report ===")
    print(f"Data Poisoning Bugs (Identical Pairs): {len(poisoned)} ({(len(poisoned)/len(merged))*100:.2f}% of pairs)")

    if not poisoned.empty:
        print("\n--- Top 5 Poisoned Lemmas ---")
        print(poisoned['error_lemma_orig'].value_counts().head(5).to_string())

    print("\nNote: Poisoned pairs confuse the model during training. They should be filtered out!")

    # if args.fix and not poisoned.empty:
    #     poisoned_pair_ids = poisoned.index.tolist()
    #     cleaned_df = df[~df['pair_id'].isin(poisoned_pair_ids)].reset_index(drop=True)
    #     cleaned_df.to_parquet(args.file_path)
    #     print(f"\n[FIXED] Removed {len(poisoned_pair_ids) * 2} poisoned rows. Saved to {args.file_path}")


if __name__ == "__main__":
    main()
