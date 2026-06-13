import os
import sys

# Windows PyTorch & PyArrow DLL conflict fix: call bootstrap_environment before importing other packages
sys.path.append(os.path.abspath("."))
from src.utils.setup import bootstrap_environment
bootstrap_environment()

import json
import logging
import torch
from typing import List, Set, Dict, Tuple

from src.config import config
from src.utils.paths import HUMAN_EVAL_PATH
from src.models.evaluation import FluencyReranker, print_evaluation_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_candidates(sentence: str) -> List[str]:
    """Generates simple rule-based GEC candidates for false friends and prepositions."""
    words = sentence.split()
    candidates = [sentence]
    
    # Common False Friend mappings
    ff_map = {
        "magazyn": "sklep",
        "magazynie": "sklepie",
        "suweniry": "pamiątki",
        "zagarę": "opaleniznę",
        "dywan": "kanapę",
        "dywanie": "kanapie",
        "arbuz": "dynia",
        "reklamówkę": "torbę",
        "pakiet": "paczka",
        "zdaczę": "resztę",
        "butelkę": "oleju",
        "ekskursję": "wycieczkę",
        "ekskursji": "wycieczki",
        "zdjąć": "wynająć",
        "rozmawiać": "mówić",
        "chaziaj": "właściciel",
        "derewo": "drzewo",
        "tuman": "mgła",
        "sobesedowanie": "rozmowę",
        "rabotę": "pracę",
        "odpusku": "urlopu",
        "chołodylnik": "lodówkę"
    }
    
    prepositions = {"w", "we", "na", "z", "ze", "do", "dla", "od", "o", "po", "u", "za", "przed", "pod"}
    
    # 1. False Friends substitutions
    for i, word in enumerate(words):
        # strip punctuation
        clean_word = word.strip(".,!?\"'()").lower()
        if clean_word in ff_map:
            sub = ff_map[clean_word]
            # restore capitalization if needed
            if word[0].isupper():
                sub = sub.capitalize()
            # restore punctuation
            punc = "".join([c for c in word if c in ".,!?"])
            sub = sub + punc
            
            # create candidate sentence
            cand_words = list(words)
            cand_words[i] = sub
            candidates.append(" ".join(cand_words))
            
    # 2. Preposition swaps
    for i, word in enumerate(words):
        clean_word = word.strip(".,!?\"'()").lower()
        if clean_word in prepositions:
            for prep in prepositions:
                if prep != clean_word:
                    sub = prep
                    if word[0].isupper():
                        sub = sub.capitalize()
                    punc = "".join([c for c in word if c in ".,!?"])
                    sub = sub + punc
                    
                    cand_words = list(words)
                    cand_words[i] = sub
                    candidates.append(" ".join(cand_words))
                    
    return list(set(candidates))


def main() -> None:
    # 1. Load gold human OOD dataset
    logging.info("Loading human evaluation dataset...")
    if not os.path.exists(HUMAN_EVAL_PATH):
        raise FileNotFoundError(f"Dataset not found at {HUMAN_EVAL_PATH}")
        
    with open(HUMAN_EVAL_PATH, "r", encoding="utf-8") as f:
        human_std = json.load(f)
        
    if config.debug_mode:
        human_std = human_std[:config.debug_human_samples]
        
    # 2. Load Fluency Reranker
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Initializing HerBERT Fluency Reranker...")
    reranker = FluencyReranker(config.reranking_model_name, device)
    
    # 3. Generate predictions using rules and HerBERT reranking
    logging.info("Running baseline rule generator + HerBERT fluency reranker...")
    predictions = []
    
    for idx, item in enumerate(human_std):
        src = item["source"]
        # get all candidates
        cands = get_candidates(src)
        # rerank with threshold 0.0 (equivalent to Pipeline 4)
        best_cand = reranker.rerank(src, cands, threshold=0.0)
        predictions.append(best_cand)
        
        if (idx + 1) % 25 == 0:
            logging.info(f"Processed {idx + 1}/{len(human_std)} sentences...")
            
    # 4. Save and evaluate
    out_dir = "./results/base/baseline_rules_reranked"
    os.makedirs(out_dir, exist_ok=True)
    preds_path = os.path.join(out_dir, "human_preds_reranked.json")
    
    with open(preds_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
        
    logging.info("\n--- Evaluating Rule-Based Reranked Baseline ---")
    f05, p, r, em, bs, tpr, fpr, fnr = print_evaluation_metrics(
        [item["source"] for item in human_std],
        [item["expected"] for item in human_std],
        predictions
    )
    
    # Merge baseline metrics into results/base/experiment_summary.json for Jupyter visualization
    summary_path = "./results/base/experiment_summary.json"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    summary_data = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary_data = json.load(f)
        except Exception:
            pass
            
    summary_data["Baseline - Reguły + HerBERT"] = {
        "out_dir": out_dir,
        "Human_EM": em,
        "Human_F05": f05,
        "Human_BERTScore": bs,
        "Human_TPR": tpr,
        "Human_FPR": fpr,
        "Human_FNR": fnr,
        "HF_Eval": {}
    }
    
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
    logging.info(f"Predictions saved to {preds_path}")
    logging.info(f"Baseline metrics merged into {summary_path}")


if __name__ == "__main__":
    main()
