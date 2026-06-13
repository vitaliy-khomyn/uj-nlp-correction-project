"""Data utilities for loading, pivoting, and partitioning GEC datasets."""
import os
import logging
import pandas as pd
from typing import Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def pivot_to_seq2seq(df: pd.DataFrame) -> pd.DataFrame:
    """Pivots the base corpus dataframe into a Seq2Seq training format.

    Args:
        df: The raw dataframe containing sentences, labels, and pair IDs.

    Returns:
        A randomized dataframe with 'source', 'target', and 'is_error' columns.
    """
    inj_df = df[df["label"] == 1].copy()
    orig_df = df[df["label"] == 0].copy()
    merged = pd.merge(inj_df, orig_df, on="pair_id", suffixes=("_src", "_tgt"))
    pairs = pd.DataFrame(
        {
            "source": merged["sentence_src"],
            "target": merged["sentence_tgt"],
            "is_error": 1,
            "error_lemma": merged["error_lemma_src"],
        }
    )

    injected_pair_ids = set(inj_df["pair_id"])
    hard_negatives = df[
        (df["label"] == 0) & (~df["pair_id"].isin(injected_pair_ids))
    ].copy()
    hn_pairs = pd.DataFrame(
        {
            "source": hard_negatives["sentence"],
            "target": hard_negatives["sentence"],
            "is_error": 0,
            "error_lemma": hard_negatives["error_lemma"],
        }
    )

    return (
        pd.concat([pairs, hn_pairs], ignore_index=True)
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )


def prepare_and_load_dataset(
    dataset_type: str = "10k",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validates the existence of synthetic datasets and loads them into memory.

    Args:
        dataset_type: Directory containing the target dataset ('10k' or '50k').

    Returns:
        A tuple of train, validation, test dataframes, and the concatenated full dataframe.
    """
    train_path = os.path.join(
        "data", "synthesized", dataset_type, "synthetic_train.parquet"
    )
    val_path = os.path.join(
        "data", "synthesized", dataset_type, "synthetic_val.parquet"
    )
    test_path = os.path.join(
        "data", "synthesized", dataset_type, "synthetic_test.parquet"
    )

    # fallback to root directory if the subfolder does not exist
    if not os.path.exists(train_path):
        train_path = os.path.join("data", "synthesized", "synthetic_train.parquet")
        val_path = os.path.join("data", "synthesized", "synthetic_val.parquet")
        test_path = os.path.join("data", "synthesized", "synthetic_test.parquet")

    for path in [train_path, val_path, test_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing synthetic dataset at {path}. Please run `python src/main/prepare_data.py` first."
            )

    train_df = pivot_to_seq2seq(pd.read_parquet(train_path))
    val_df = pivot_to_seq2seq(pd.read_parquet(val_path))
    test_df = pivot_to_seq2seq(pd.read_parquet(test_path))
    seq_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    logging.info(
        f"Training set size: {len(train_df)} | Val set size: {len(val_df)} | Test set size: {len(test_df)}"
    )

    return train_df, val_df, test_df, seq_df


def get_experiment_splits(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Slices the main datasets into specific subsets for the ablation experiments.

    Args:
        train_df: Training set.
        val_df: Validation set.
        test_df: Test set.

    Returns:
        A tuple containing train_errors_only, val_errors_only, train_ablation,
        val_ablation, and test_ablation dataframes.
    """
    train_df_errors_only = train_df[train_df["is_error"] == 1].copy()
    val_df_errors_only = val_df[val_df["is_error"] == 1].copy()

    prep_lemmas = ["w", "na", "z", "do", "dla", "od", "o"]
    train_df_ablation = train_df[
        (train_df["error_lemma"].isin(prep_lemmas)) | (train_df["is_error"] == 0)
    ].copy()
    val_df_ablation = val_df[
        (val_df["error_lemma"].isin(prep_lemmas)) | (val_df["is_error"] == 0)
    ].copy()
    test_df_ablation = test_df[
        (test_df["error_lemma"].isin(prep_lemmas)) | (test_df["is_error"] == 0)
    ].copy()

    return (
        train_df_errors_only,
        val_df_errors_only,
        train_df_ablation,
        val_df_ablation,
        test_df_ablation,
    )