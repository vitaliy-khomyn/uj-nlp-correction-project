"""Module containing ERRANT/M2-compliant evaluation metrics and inference helpers for GEC."""
import Levenshtein
import pandas as pd
import torch
import random
import re
import os
import json
from typing import List, Dict, Tuple, Set, Any, Optional
import contextlib
from src.config import config
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
import evaluate

# global metrics cache
_bertscore_metric = None
_gender_mismatches = None
_false_friends_words = None


def get_bertscore_metric() -> Any:
    """Lazy loads the Hugging Face evaluate BERTScore metric.

    Returns:
        The BERTScore evaluation metric object.
    """
    global _bertscore_metric
    if _bertscore_metric is None:
        _bertscore_metric = evaluate.load("bertscore")
    return _bertscore_metric


def _get_classification_resources() -> Tuple[Dict[str, Any], Set[str]]:
    """Loads and caches resources for categorizing L1 error interferences.

    Returns:
        A tuple of the gender mismatch map and the false friend word set.
    """
    global _gender_mismatches, _false_friends_words
    if _gender_mismatches is None or _false_friends_words is None:
        _gender_mismatches = {}
        _false_friends_words = set()

        gender_path = os.path.join("data", "generated", "gender_mismatches.json")
        if os.path.exists(gender_path):
            try:
                with open(gender_path, "r", encoding="utf-8") as f:
                    _gender_mismatches = json.load(f)
            except Exception:
                pass

        ff_path = os.path.join("data", "scraped", "unified_false_friends.json")
        if os.path.exists(ff_path):
            try:
                with open(ff_path, "r", encoding="utf-8") as f:
                    ff_data = json.load(f)
                    for item in ff_data:
                        pl_word = item.get("pl_word", "")
                        if pl_word:
                            _false_friends_words.add(pl_word.lower())
                        for ff_info in item.get("false_friends", {}).values():
                            l2_word = ff_info.get("word", "")
                            if l2_word:
                                _false_friends_words.add(l2_word.lower())
            except Exception:
                pass

    return _gender_mismatches, _false_friends_words


def classify_error(
    source: str,
    expected: str,
    gender_mismatches: Dict[str, Any],
    false_friends_words: Set[str],
) -> str:
    """Classifies the GEC error category between a source and expected sentence.

    Args:
        source: The source sentence.
        expected: The expected correct target sentence.
        gender_mismatches: Cached dictionary of known gender mismatches.
        false_friends_words: Cached set of known false friend words.

    Returns:
        One of 'prep', 'false_friend', 'gender', 'case', 'typos', 'other', or 'identity'.
    """
    s_clean = re.sub(r"[^\w\s]", "", source.lower()).split()
    e_clean = re.sub(r"[^\w\s]", "", expected.lower()).split()

    s_set = set(s_clean)
    e_set = set(e_clean)

    s_diff = s_set - e_set
    e_diff = e_set - s_set

    if not s_diff and not e_diff:
        return "identity"

    prepositions = {
        "w",
        "we",
        "na",
        "z",
        "ze",
        "do",
        "dla",
        "od",
        "o",
        "u",
        "nad",
        "pod",
        "przed",
        "po",
        "za",
        "przez",
        "k",
        "przy",
        "bez",
        "przeciw",
        "wobec",
        "dzięki",
    }

    if (s_diff & prepositions) or (e_diff & prepositions):
        return "prep"

    common_ffs = {
        "magazyn",
        "sklep",
        "suweniry",
        "pamiątki",
        "zagarę",
        "opaleniznę",
        "dywan",
        "kanapę",
        "arbuz",
        "dynia",
        "reklamówkę",
        "pakiet",
        "zdaczę",
        "resztę",
        "butelkę",
        "masła",
        "oleju",
        "pensjonat",
        "ekskursję",
        "wycieczkę",
        "zdjąć",
        "wynająć",
        "rozmawiać",
        "mówić",
        "chaziaj",
        "właściciel",
    }
    ffs = common_ffs.union(false_friends_words)
    if (s_diff & ffs) or (e_diff & ffs):
        return "false_friend"

    gender_words = {
        "ta",
        "ten",
        "to",
        "te",
        "mój",
        "moją",
        "moje",
        "ciekawy",
        "ciekawa",
        "piękny",
        "piękna",
    }
    if (s_diff & gender_words) or (e_diff & gender_words):
        return "gender"

    for w in s_diff:
        if w in gender_mismatches:
            return "gender"

    for w_s in s_diff:
        for w_e in e_diff:
            dist = Levenshtein.distance(w_s, w_e)
            if dist <= 2:
                if len(w_s) > 3 and len(w_e) > 3 and w_s[:3] == w_e[:3]:
                    return "case"
                else:
                    return "typos"

    return "other"


def get_m2_edits(source: str, target: str) -> Set[Tuple[int, int, str]]:
    """Extracts M2Scorer/ERRANT-compliant token-level edits using Levenshtein distance.

    Args:
        source: The source sentence.
        target: The target sentence.

    Returns:
        A set of edit operations formatted as (start_index, end_index, target_tokens).
    """
    s_clean = source.strip().lower()
    t_clean = target.strip().lower()

    s_tokens = re.findall(r"\w+|[^\w\s]", s_clean)
    t_tokens = re.findall(r"\w+|[^\w\s]", t_clean)

    edits = set()
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(s_tokens, t_tokens):
        if tag != "equal":
            edits.add((i1, i2, " ".join(t_tokens[j1:j2])))

    return edits


def calculate_f05(
    src_texts: List[str], expected: List[str], pred_texts: List[str]
) -> Tuple[float, float, float]:
    """Calculates Precision, Recall, and the F0.5 Score for Grammatical Error Correction.

    Args:
        src_texts: Source sentences.
        expected: Gold target sentences.
        pred_texts: Predicted target sentences.

    Returns:
        A tuple containing precision, recall, and F0.5 score.
    """
    tp = fp = fn = 0.0
    for src, exp, pred in zip(src_texts, expected, pred_texts):
        gold_edits = get_m2_edits(src, exp)
        pred_edits = get_m2_edits(src, pred)

        tp += len(gold_edits & pred_edits)
        fp += len(pred_edits - gold_edits)
        fn += len(gold_edits - pred_edits)

    p = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f05 = (
        (1 + 0.5**2) * p * r / (0.5**2 * p + r) if (p + r) > 0 else 0.0
    )
    return p, r, f05


class FluencyReranker:
    """Loads, scores, and reranks correction candidates based on Polish language fluency."""

    def __init__(self, model_name: str, device: torch.device):
        """Initializes the reranker model and tokenizer.

        Args:
            model_name: The name of the masked language model.
            device: The torch device to run the model on.
        """
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        import logging

        self.device = device
        dtype = (
            torch.bfloat16
            if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else (torch.float16 if torch.cuda.is_available() else torch.float32)
        )
        
        # load fluency reranker model directly
        logging.info(f"Loading fluency reranker model: {model_name}...")
        self.model = AutoModelForMaskedLM.from_pretrained(
            model_name, torch_dtype=dtype
        ).to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def compute_pll(self, sentence: str) -> float:
        """Computes the length-normalized Pseudo-Log-Likelihood score of a sentence in a single batch pass.

        Args:
            sentence: The input sentence to score.

        Returns:
            The computed length-normalized pseudo-log-likelihood score.
        """
        enc = self.tokenizer(sentence, return_tensors="pt")
        input_ids = enc["input_ids"][0]
        seq_len = len(input_ids)
        special_tokens = {
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id,
        }
        mask_indices = [
            i for i in range(seq_len) if input_ids[i].item() not in special_tokens
        ]
        if not mask_indices:
            return 0.0
        batch_input_ids = input_ids.repeat(len(mask_indices), 1)
        for row_idx, col_idx in enumerate(mask_indices):
            batch_input_ids[row_idx, col_idx] = self.tokenizer.mask_token_id
        batch_input_ids = batch_input_ids.to(self.device)
        with torch.inference_mode():
            outputs = self.model(input_ids=batch_input_ids)
            logits = outputs.logits
        log_probs = torch.log_softmax(logits, dim=-1)
        total_pll = 0.0
        for row_idx, col_idx in enumerate(mask_indices):
            target_token_id = input_ids[col_idx].item()
            total_pll += log_probs[row_idx, col_idx, target_token_id].item()
        
        # NORMALIZATION: Average logprob per token
        return total_pll / len(mask_indices)

    def compute_pll_batch(self, sentences: List[str]) -> List[float]:
        """Computes the length-normalized Pseudo-Log-Likelihood score of a list of sentences in parallel batches.

        Args:
            sentences: A list of sentences to score.

        Returns:
            A list of length-normalized pseudo-log-likelihood scores.
        """
        if not sentences:
            return []

        flat_input_ids = []
        flat_attention_mask = []
        flat_target_ids = []
        flat_mask_positions = []
        
        sentence_offsets = []
        current_offset = 0

        special_tokens = {
            self.tokenizer.cls_token_id,
            self.tokenizer.sep_token_id,
            self.tokenizer.pad_token_id,
        }

        for sentence in sentences:
            enc = self.tokenizer(sentence, return_tensors="pt")
            input_ids = enc["input_ids"][0]
            attn_mask = enc["attention_mask"][0]
            seq_len = len(input_ids)

            mask_indices = [
                i for i in range(seq_len) if input_ids[i].item() not in special_tokens
            ]

            if not mask_indices:
                sentence_offsets.append((current_offset, current_offset))
                continue

            sentence_offsets.append((current_offset, current_offset + len(mask_indices)))
            current_offset += len(mask_indices)

            for col_idx in mask_indices:
                masked_ids = input_ids.clone()
                masked_ids[col_idx] = self.tokenizer.mask_token_id
                
                flat_input_ids.append(masked_ids)
                flat_attention_mask.append(attn_mask)
                flat_target_ids.append(input_ids[col_idx].item())
                flat_mask_positions.append(col_idx)

        if not flat_input_ids:
            return [0.0] * len(sentences)

        from torch.nn.utils.rnn import pad_sequence
        batch_input_ids = pad_sequence(flat_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id).to(self.device)
        batch_attn_mask = pad_sequence(flat_attention_mask, batch_first=True, padding_value=0).to(self.device)

        flat_log_probs = []
        sub_batch_size = 128

        with torch.inference_mode():
            for start_idx in range(0, len(flat_input_ids), sub_batch_size):
                sub_input_ids = batch_input_ids[start_idx : start_idx + sub_batch_size]
                sub_attn_mask = batch_attn_mask[start_idx : start_idx + sub_batch_size]
                
                outputs = self.model(input_ids=sub_input_ids, attention_mask=sub_attn_mask)
                logits = outputs.logits
                log_probs = torch.log_softmax(logits, dim=-1)
                
                for offset_idx in range(logits.shape[0]):
                    global_idx = start_idx + offset_idx
                    col_idx = flat_mask_positions[global_idx]
                    target_token_id = flat_target_ids[global_idx]
                    prob = log_probs[offset_idx, col_idx, target_token_id].item()
                    flat_log_probs.append(prob)

        results = []
        for idx, (start, end) in enumerate(sentence_offsets):
            if start == end:
                results.append(0.0)
            else:
                sentence_log_probs = flat_log_probs[start:end]
                # Length-normalized average log-probability
                results.append(sum(sentence_log_probs) / len(sentence_log_probs))
        return results

    def rerank(self, source: str, candidates: List[str], threshold: float = 0.0) -> str:
        """Selects the candidate that improves fluency above a given threshold relative to the source.

        Args:
            source: The original input sentence.
            candidates: Alternative correction candidates.
            threshold: Minimum relative improvement required to swap.

        Returns:
            The selected candidate string.
        """
        src_pll = self.compute_pll(source)
        best_cand = source
        best_pll = src_pll

        unique_candidates = list(set(candidates))
        for cand in unique_candidates:
            if cand.strip() == source.strip():
                continue
            cand_pll = self.compute_pll(cand)
            if cand_pll > best_pll + threshold:
                best_pll = cand_pll
                best_cand = cand
        return best_cand

    def rerank_batch(self, sources: List[str], candidates_list: List[List[str]], threshold: float = 0.0) -> List[str]:
        """Reranks candidates for a batch of sources in a parallel/batched way.

        Args:
            sources: A list of original input sentences.
            candidates_list: A list of lists of candidate sentences.
            threshold: Minimum relative improvement required to swap.

        Returns:
            A list of selected candidate strings.
        """
        all_sentences = []
        sentence_to_idx = {}

        def add_sentence(s):
            if s not in sentence_to_idx:
                sentence_to_idx[s] = len(all_sentences)
                all_sentences.append(s)

        for src, cands in zip(sources, candidates_list):
            add_sentence(src)
            for cand in cands:
                add_sentence(cand)

        pllls = self.compute_pll_batch(all_sentences)

        results = []
        for src, cands in zip(sources, candidates_list):
            src_pll = pllls[sentence_to_idx[src]]
            best_cand = src
            best_pll = src_pll

            for cand in set(cands):
                if cand.strip() == src.strip():
                    continue
                cand_pll = pllls[sentence_to_idx[cand]]
                if cand_pll > best_pll + threshold:
                    best_pll = cand_pll
                    best_cand = cand
            results.append(best_cand)
        return results


def generate_predictions(
    test_data: List[Dict[str, str]],
    hf_model: Any,
    hf_tokenizer: Any,
    device: torch.device,
    max_seq_length: int = config.max_seq_length,
    batch_size: int = config.infer_batch_size,
    num_beams: int = 1,
    task_prefix: str = config.task_prefix,
) -> Tuple[List[str], List[str], List[str]]:
    """Runs inference on the dataset and returns raw source, expected, and predicted sequences.

    Args:
        test_data: A list of dictionaries containing 'source' and 'expected' keys.
        hf_model: The Hugging Face model for generation.
        hf_tokenizer: The Hugging Face tokenizer.
        device: The device to run inference on.
        max_seq_length: Maximum sequence length for generation.
        batch_size: Batch size for inference.
        num_beams: Number of beams for generation.
        task_prefix: Prefix to prepend to source sentences.

    Returns:
        A tuple of raw source sentences, expected sentences, and model predictions.
    """
    if hf_model is None or hf_tokenizer is None:
        return [], [], []

    all_preds, all_exps, all_raw_srcs = [], [], []

    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else contextlib.nullcontext()
    )

    reranker = None
    if getattr(config, "use_reranking", False):
        import logging

        logging.info("Initializing fluency reranker...")
        reranker = FluencyReranker(config.reranking_model_name, device)

    test_ds = Dataset.from_list(test_data)

    def tokenize_fn(examples):
        prefixed = [task_prefix + s for s in examples["source"]]
        return hf_tokenizer(prefixed, truncation=True, max_length=max_seq_length)

    test_ds = test_ds.map(
        tokenize_fn, batched=True, remove_columns=test_ds.column_names
    )
    test_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

    collator = DataCollatorWithPadding(tokenizer=hf_tokenizer)
    dataloader = DataLoader(
        test_ds, batch_size=batch_size, collate_fn=collator, shuffle=False
    )

    with torch.inference_mode():
        with autocast_ctx:
            for i, batch in enumerate(dataloader):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                if num_beams > 1:
                    if reranker is not None:
                        # generate top-K return sequences for reranking
                        outputs = hf_model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_length=max_seq_length,
                            num_beams=num_beams,
                            num_return_sequences=num_beams,
                            early_stopping=True,
                        )
                    else:
                        outputs = hf_model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_length=max_seq_length,
                            num_beams=num_beams,
                            early_stopping=True,
                        )
                else:
                    outputs = hf_model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=max_seq_length,
                    )

                pred_texts = hf_tokenizer.batch_decode(
                    outputs, skip_special_tokens=True
                )

                # extract original source and expected from the batch
                start_idx = i * batch_size
                batch_data = test_data[start_idx : start_idx + input_ids.shape[0]]
                all_exps.extend([item["expected"] for item in batch_data])
                all_raw_srcs.extend([item["source"] for item in batch_data])

                if reranker is not None:
                    # apply fluency filtering to batch predictions
                    effective_beams = num_beams if num_beams > 1 else 1
                    batch_sources = []
                    batch_cands_list = []
                    for batch_item_idx, item in enumerate(batch_data):
                        batch_sources.append(item["source"])
                        cands = pred_texts[
                            batch_item_idx
                            * effective_beams : (batch_item_idx + 1)
                            * effective_beams
                        ]
                        batch_cands_list.append(cands)
                    
                    best_cands = reranker.rerank_batch(
                        batch_sources, batch_cands_list, threshold=config.reranking_threshold
                    )
                    all_preds.extend(best_cands)
                else:
                    all_preds.extend(pred_texts)

    return all_raw_srcs, all_exps, all_preds


def _normalize_for_em(text: str) -> str:
    """Removes punctuation and normalizes whitespace for fair Exact Match comparison.

    Args:
        text: The input string.

    Returns:
        The normalized string.
    """
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


def print_evaluation_metrics(
    all_raw_srcs: List[str], all_exps: List[str], all_preds: List[str]
) -> Tuple[float, float, float, float, float, float, float, float]:
    """Calculates and logs evaluation metrics and displays sample inferences.

    Args:
        all_raw_srcs: List of original source sentences.
        all_exps: List of expected target sentences.
        all_preds: List of model-generated predictions.

    Returns:
        A tuple of global metrics: F0.5, Precision, Recall, EM, BERTScore, TPR, FPR, FNR.
    """
    p, r, f05_score = calculate_f05(all_raw_srcs, all_exps, all_preds)
    exact_matches = sum(
        1
        for p_str, e in zip(all_preds, all_exps)
        if _normalize_for_em(p_str) == _normalize_for_em(e)
    )
    em_score = exact_matches / len(all_exps) if len(all_exps) > 0 else 0

    err_srcs, err_exps, err_preds = [], [], []
    id_srcs, id_exps, id_preds = [], [], []

    for s, e, p_str in zip(all_raw_srcs, all_exps, all_preds):
        if s.strip() != e.strip():
            err_srcs.append(s)
            err_exps.append(e)
            err_preds.append(p_str)
        else:
            id_srcs.append(s)
            id_exps.append(e)
            id_preds.append(p_str)

    # calculate metrics for erroneous subset
    err_p, err_r, err_f05 = (
        calculate_f05(err_srcs, err_exps, err_preds)
        if err_srcs
        else (0.0, 0.0, 0.0)
    )
    err_em_count = sum(
        1
        for p_str, e in zip(err_preds, err_exps)
        if _normalize_for_em(p_str) == _normalize_for_em(e)
    )
    tpr = err_em_count / len(err_exps) if len(err_exps) > 0 else 0.0
    fnr = (
        sum(
            1
            for p_str, e in zip(err_preds, err_exps)
            if _normalize_for_em(p_str) != _normalize_for_em(e)
        )
        / len(err_exps)
        if len(err_exps) > 0
        else 0.0
    )

    # calculate metrics for identity subset
    id_em_count = sum(
        1
        for p_str, e in zip(id_preds, id_exps)
        if _normalize_for_em(p_str) == _normalize_for_em(e)
    )
    id_em = id_em_count / len(id_exps) if len(id_exps) > 0 else 0.0
    fpr = (
        sum(
            1
            for p_str, e in zip(id_preds, id_exps)
            if _normalize_for_em(p_str) != _normalize_for_em(e)
        )
        / len(id_exps)
        if len(id_exps) > 0
        else 0.0
    )

    bs_results = get_bertscore_metric().compute(
        predictions=all_preds, references=all_exps, lang="pl"
    )
    avg_bertscore = (
        sum(bs_results["f1"]) / len(bs_results["f1"]) if bs_results["f1"] else 0.0
    )

    print("--- Global Metrics ---")
    print(
        f"Overall EM: {em_score:.2%} | Precision: {p:.4f} | Recall: {r:.4f} | F0.5: {f05_score:.4f} | BERTScore: {avg_bertscore:.4f}"
    )
    print("--- Erroneous Subset (Errors) ---")
    print(
        f"F0.5 Score (Correction Quality): {err_f05:.4f} | True Positive Rate (EM): {tpr:.2%}"
    )
    print("--- Identity Subset (Correct Sentences) ---")
    print(
        f"False Positive Rate (Sentences Ruined): {fpr:.2%} | Exact Match (Correctly Unchanged): {id_em:.2%}\n"
    )

    # metrics by error category
    gender_mismatches, false_friends_words = _get_classification_resources()
    category_groups: Dict[str, List[Tuple[str, str, str]]] = {
        "prep": [],
        "false_friend": [],
        "gender": [],
        "case": [],
        "typos": [],
        "other": [],
        "identity": [],
    }

    for s, e, pred in zip(all_raw_srcs, all_exps, all_preds):
        cat = classify_error(s, e, gender_mismatches, false_friends_words)
        if cat in category_groups:
            category_groups[cat].append((s, e, pred))

    print("--- Metrics by Error Category ---")
    for cat, items in category_groups.items():
        if not items:
            continue
        cat_srcs = [it[0] for it in items]
        cat_exps = [it[1] for it in items]
        cat_preds = [it[2] for it in items]

        if cat == "identity":
            cat_em_count = sum(
                1
                for p_str, e in zip(cat_preds, cat_exps)
                if _normalize_for_em(p_str) == _normalize_for_em(e)
            )
            cat_em = cat_em_count / len(cat_exps)
            cat_fpr = 1.0 - cat_em
            print(
                f"[{cat.upper():<12}] Count: {len(items):<4} | Exact Match: {cat_em:.2%} | False Positive Rate: {cat_fpr:.2%}"
            )
        else:
            cat_p, cat_r, cat_f05 = calculate_f05(cat_srcs, cat_exps, cat_preds)
            cat_em_count = sum(
                1
                for p_str, e in zip(cat_preds, cat_exps)
                if _normalize_for_em(p_str) == _normalize_for_em(e)
            )
            cat_tpr = cat_em_count / len(cat_exps)
            print(
                f"[{cat.upper():<12}] Count: {len(items):<4} | F0.5: {cat_f05:.4f} | Precision: {cat_p:.4f} | Recall: {cat_r:.4f} | True Positive Rate: {cat_tpr:.2%}"
            )
    print()

    true_positives = [
        (s, e, p)
        for s, e, p in zip(err_srcs, err_exps, err_preds)
        if _normalize_for_em(p) == _normalize_for_em(e)
    ]
    false_negatives = [
        (s, e, p)
        for s, e, p in zip(err_srcs, err_exps, err_preds)
        if _normalize_for_em(p) != _normalize_for_em(e)
    ]
    false_positives = [
        (s, e, p)
        for s, e, p in zip(id_srcs, id_exps, id_preds)
        if _normalize_for_em(p) != _normalize_for_em(e)
    ]

    random.shuffle(true_positives)
    random.shuffle(false_negatives)
    random.shuffle(false_positives)

    print("--- Sample: True positives ---")
    for s, e, p_str in true_positives[:3]:
        print(f"Source:   {s}\nExpected: {e}\nModel:    {p_str}\n")

    print("--- Sample: False negatives ---")
    for s, e, p_str in false_negatives[:3]:
        print(f"Source:   {s}\nExpected: {e}\nModel:    {p_str}\n")

    print("--- Sample: False positives ---")
    for s, e, p_str in false_positives[:3]:
        print(f"Source:   {s}\nExpected: {e}\nModel:    {p_str}\n")

    return f05_score, p, r, em_score, avg_bertscore, tpr, fpr, fnr


def get_categorized_metrics(
    all_raw_srcs: List[str], all_exps: List[str], all_preds: List[str]
) -> Dict[str, Dict[str, Any]]:
    """Computes GEC metrics grouped by error category.

    Args:
        all_raw_srcs: List of original source sentences.
        all_exps: List of expected target sentences.
        all_preds: List of model-generated predictions.

    Returns:
        A dictionary mapping each category to its counts and rates.
    """
    gender_mismatches, false_friends_words = _get_classification_resources()
    category_groups: Dict[str, List[Tuple[str, str, str]]] = {
        "prep": [],
        "false_friend": [],
        "gender": [],
        "case": [],
        "typos": [],
        "other": [],
        "identity": [],
    }

    for s, e, pred in zip(all_raw_srcs, all_exps, all_preds):
        cat = classify_error(s, e, gender_mismatches, false_friends_words)
        if cat in category_groups:
            category_groups[cat].append((s, e, pred))

    results: Dict[str, Dict[str, Any]] = {}
    for cat, items in category_groups.items():
        if not items:
            results[cat] = {"count": 0, "f05": 0.0, "tpr": 0.0, "fpr": 0.0}
            continue
        cat_srcs = [it[0] for it in items]
        cat_exps = [it[1] for it in items]
        cat_preds = [it[2] for it in items]

        if cat == "identity":
            cat_em_count = sum(
                1
                for p_str, e in zip(cat_preds, cat_exps)
                if _normalize_for_em(p_str) == _normalize_for_em(e)
            )
            cat_em = cat_em_count / len(cat_exps)
            cat_fpr = 1.0 - cat_em
            results[cat] = {
                "count": len(items),
                "em": cat_em,
                "fpr": cat_fpr,
            }
        else:
            cat_p, cat_r, cat_f05 = calculate_f05(cat_srcs, cat_exps, cat_preds)
            cat_em_count = sum(
                1
                for p_str, e in zip(cat_preds, cat_exps)
                if _normalize_for_em(p_str) == _normalize_for_em(e)
            )
            cat_tpr = cat_em_count / len(cat_exps)
            results[cat] = {
                "count": len(items),
                "f05": cat_f05,
                "tpr": cat_tpr,
            }
    return results


def predict_seq2seq(
    test_data: List[Dict[str, str]],
    hf_model: Any,
    hf_tokenizer: Any,
    device: Any,
    max_seq_length: int = config.max_seq_length,
    batch_size: int = config.infer_batch_size,
    num_beams: int = 1,
    task_prefix: str = config.task_prefix,
) -> Tuple[float, float, float, float, float, float, float, float]:
    """Facade function to preserve backwards compatibility.

    Args:
        test_data: Test dataset records.
        hf_model: Hugging Face generator model.
        hf_tokenizer: Hugging Face tokenizer.
        device: Torch device.
        max_seq_length: Generation bounds.
        batch_size: Evaluation batch size.
        num_beams: Inference beam search size.
        task_prefix: Model prompt prefix.

    Returns:
        All calculated float GEC metrics.
    """
    all_raw_srcs, all_exps, all_preds = generate_predictions(
        test_data,
        hf_model,
        hf_tokenizer,
        device,
        max_seq_length,
        batch_size,
        num_beams,
        task_prefix,
    )
    return print_evaluation_metrics(all_raw_srcs, all_exps, all_preds)


def evaluate_baseline(test_df: pd.DataFrame, tokenizer: Any) -> Tuple[float, float]:
    """Evaluates the performance of a 'do-nothing' baseline on the test dataset.

    Args:
        test_df: Test dataset.
        tokenizer: HuggingFace tokenizer.

    Returns:
        The exact match score and the GLEU score.
    """
    from nltk.translate.gleu_score import corpus_gleu

    baseline_em = sum(test_df["source"] == test_df["target"]) / len(test_df)
    list_of_references_base = [
        [tokenizer.tokenize(l.strip())] for l in test_df["target"]
    ]
    hypotheses_base = [tokenizer.tokenize(p.strip()) for p in test_df["source"]]
    baseline_gleu = corpus_gleu(list_of_references_base, hypotheses_base)
    return baseline_em, baseline_gleu


def run_human_eval_ablation(
    ratios: List[float],
    human_gold_standard: List[Dict[str, str]],
    tokenizer: Any,
    device: Any,
    max_seq_length: int = config.max_seq_length,
    batch_size: int = config.infer_batch_size,
    num_beams: int = 1,
) -> Dict[float, Dict[str, float]]:
    """Runs human evaluation inference on the ablated Variant B models.

    Args:
        ratios: Identity ratios to evaluate.
        human_gold_standard: Ground truth evaluations.
        tokenizer: Hugging Face tokenizer.
        device: Torch device.
        max_seq_length: Sequence generation limits.
        batch_size: Evaluation batch size.
        num_beams: Number of beams for generation.

    Returns:
        A dictionary containing performance metrics indexed by ratio.
    """
    import gc
    import torch
    import json
    import os
    from transformers import AutoModelForSeq2SeqLM

    human_metrics = {}
    for r in ratios:
        print(f"\n=== INFERENCJA: Wariant B ({int(r*100)}% Identity) ===")
        dir_name = f"./results/var_b_{int(r*100)}"
        adapter_config_path = os.path.join(dir_name, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            from peft import PeftModel

            with open(adapter_config_path, "r") as f:
                conf = json.load(f)
            base_model_name = conf.get("base_model_name_or_path", "allegro/plt5-base")
            dtype = (
                torch.bfloat16
                if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
                else (torch.float16 if torch.cuda.is_available() else torch.float32)
            )
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                base_model_name, torch_dtype=dtype, use_safetensors=True
            ).to(device)
            tmp_model = PeftModel.from_pretrained(base_model, dir_name)
            tmp_model = tmp_model.merge_and_unload()
            tmp_model.tie_weights()
        else:
            dtype = (
                torch.bfloat16
                if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
                else (torch.float16 if torch.cuda.is_available() else torch.float32)
            )
            tmp_model = AutoModelForSeq2SeqLM.from_pretrained(
                dir_name, torch_dtype=dtype, use_safetensors=True
            ).to(device)
        f05, p, r_score, em, bertscore, tpr, fpr, fnr = predict_seq2seq(
            human_gold_standard,
            hf_model=tmp_model,
            hf_tokenizer=tokenizer,
            device=device,
            max_seq_length=max_seq_length,
            batch_size=batch_size,
            num_beams=num_beams,
        )
        human_metrics[r] = {
            "P": p,
            "R": r_score,
            "F0.5": f05,
            "EM": em,
            "BERTScore": bertscore,
            "TPR": tpr,
            "FPR": fpr,
            "FNR": fnr,
        }
        del tmp_model
        if "base_model" in locals():
            del base_model
        gc.collect()
        torch.cuda.empty_cache()

    return human_metrics
