import os
import sys

# Windows PyTorch & PyArrow DLL conflict fix: call bootstrap_environment before importing other packages
sys.path.append(os.path.abspath("."))
from src.utils.setup import bootstrap_environment
bootstrap_environment()

import json
import logging
import torch
from typing import List, Set, Dict, Tuple, Any

from src.config import config
from src.utils.paths import OOD_EVAL_PATH
from src.models.evaluation import FluencyReranker, print_evaluation_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_candidates_spacy(doc: Any, polimorf_cache: Any) -> List[str]:
    """Generates grammatically correct GEC candidates using SpaCy dependencies and PoliMorf inflections."""
    tokens = list(doc)
    sentence_text = doc.text
    candidates = [sentence_text]
    
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
    
    # Target case mapping for prepositions
    prep_cases = {
        "w": "loc",
        "na": "loc",
        "z": "inst",
        "do": "gen",
        "dla": "gen",
        "od": "gen",
        "o": "loc",
        "po": "loc",
        "u": "gen",
        "za": "inst",
        "przed": "inst",
        "pod": "inst"
    }
    
    # 1. False Friends substitutions
    for i, tok in enumerate(tokens):
        clean_word = tok.text.lower()
        if clean_word in ff_map:
            sub = ff_map[clean_word]
            if tok.text[0].isupper():
                sub = sub.capitalize()
            cand_tokens = [t.text for t in tokens]
            cand_tokens[i] = sub
            cand_text = "".join([text + t.whitespace_ for text, t in zip(cand_tokens, tokens)])
            candidates.append(cand_text.strip())
            
    # 2. Preposition swaps with proper grammatical declension of the governed noun
    for i, tok in enumerate(tokens):
        if tok.pos_ == "ADP" and tok.text.lower() in prepositions:
            orig_prep = tok.text.lower()
            
            # Find governed noun
            noun_tok = None
            if tok.dep_ == "case" and tok.head.pos_ == "NOUN":
                noun_tok = tok.head
                
            for new_prep in prepositions:
                if new_prep == orig_prep:
                    continue
                
                cand_tokens = [t.text for t in tokens]
                cand_tokens[i] = new_prep.capitalize() if tok.text[0].isupper() else new_prep
                
                if noun_tok is not None and new_prep in prep_cases:
                    target_case = prep_cases[new_prep]
                    
                    from src.data.synthesis.synthesize_data import NUMBER_MAP, CASE_MAP, get_spacy_gender, get_first_morph, _match_case, _get_modifiers_to_update
                    num = NUMBER_MAP.get(get_first_morph(noun_tok.morph, "Number"))
                    gender = get_spacy_gender(noun_tok.morph)
                    
                    if num and gender:
                        morph_key = f"{num}:{target_case}:{gender}"
                        new_noun_form, _ = polimorf_cache.find_inflected_form(
                            noun_tok.lemma_.lower(), "subst", morph_key
                        )
                        if new_noun_form:
                            cand_tokens[noun_tok.i] = _match_case(noun_tok.text, new_noun_form)
                            
                            # Update modifying adjectives/determiners as well
                            modifiers = _get_modifiers_to_update(noun_tok, tokens)
                            for child in modifiers:
                                child_num = NUMBER_MAP.get(get_first_morph(child.morph, "Number"))
                                if child_num:
                                    new_adj_form = polimorf_cache.find_inflected_adj_or_det(
                                        child.lemma_.lower(), child_num, target_case, gender
                                    )
                                    if new_adj_form:
                                        cand_tokens[child.i] = _match_case(child.text, new_adj_form)
                
                # Fix phonotactics and join
                from src.data.synthesis.synthesize_data import _fix_phonotactics
                _fix_phonotactics(cand_tokens, i)
                
                cand_text = "".join([text + t.whitespace_ for text, t in zip(cand_tokens, tokens)])
                candidates.append(cand_text.strip())
                
    return list(set(candidates))


def main() -> None:
    # 1. Load OOD evaluation dataset
    logging.info("Loading OOD evaluation dataset...")
    if not os.path.exists(OOD_EVAL_PATH):
        raise FileNotFoundError(f"Dataset not found at {OOD_EVAL_PATH}")
        
    with open(OOD_EVAL_PATH, "r", encoding="utf-8") as f:
        human_std = json.load(f)
        
    if config.debug_mode:
        human_std = human_std[:config.debug_human_samples]
        
    # 2. Load Fluency Reranker
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Initializing HerBERT Fluency Reranker...")
    reranker = FluencyReranker(config.reranking_model_name, device)
    
    # 3. Load SpaCy and PoliMorf resources
    import spacy
    from src.utils.paths import UNIFIED_FF_PATH, POLIMORF_PARQUET_PATH, GENDER_MISMATCH_PATH, PREP_MISMATCH_PATH
    from src.data.synthesis.synthesize_data import load_resources, PolimorfCache
    
    logging.info("Loading PoliMorf resources...")
    _, polimorf_dict, _, _ = load_resources(UNIFIED_FF_PATH, POLIMORF_PARQUET_PATH, GENDER_MISMATCH_PATH, PREP_MISMATCH_PATH)
    polimorf_cache = PolimorfCache(polimorf_dict)
    
    logging.info("Loading SpaCy pipeline...")
    nlp = spacy.load("pl_core_news_lg", disable=["attribute_ruler"])
    
    # 4. Generate predictions using rules and HerBERT reranking in parallel batches
    logging.info("Running baseline rule generator + HerBERT fluency reranker...")
    sources = [item["source"] for item in human_std]
    
    logging.info("Generating candidate lists using grammatical rules in parallel...")
    docs = list(nlp.pipe(sources, batch_size=50))
    candidates_list = []
    for doc in docs:
        cands = get_candidates_spacy(doc, polimorf_cache)
        candidates_list.append(cands)
        
    logging.info("Batch reranking candidates using HerBERT...")
    predictions = reranker.rerank_batch(sources, candidates_list, threshold=0.0)
        
    # 5. Save and evaluate
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
        "OOD_EM": em,
        "OOD_F05": f05,
        "OOD_BERTScore": bs,
        "OOD_TPR": tpr,
        "OOD_FPR": fpr,
        "OOD_FNR": fnr,
        "HF_Eval": {}
    }
    
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
    logging.info(f"Predictions saved to {preds_path}")
    logging.info(f"Baseline metrics merged into {summary_path}")


if __name__ == "__main__":
    main()
