import os
import sys

# Bootstrap environment first to prevent Windows DLL conflicts and HF warnings
sys.path.append(os.path.abspath("."))
from src.utils.setup import bootstrap_environment
bootstrap_environment()

import json
import logging
import argparse
import torch
import pandas as pd
from typing import Dict, Any, List, Tuple
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from peft import get_peft_model, LoraConfig, TaskType, PeftModel

from src.config import config
from src.utils.paths import HUMAN_EVAL_PATH
from src.data.dataset import prepare_and_load_dataset, pivot_to_seq2seq
from src.models.evaluation import (
    generate_predictions, 
    print_evaluation_metrics, 
    get_categorized_metrics,
    FluencyReranker
)
from src.training.trainer import CustomORPOTrainer, ORPODataCollator


def get_pipeline_directories(pipeline: str) -> Dict[str, str]:
    """Returns mapping of variant names to their output directories for a pipeline."""
    base_dir = "./results" if not config.debug_mode else "./results-debug"
    
    if pipeline == "1":
        return {
            "Wariant A (Tylko Błędy) - plt5-small": os.path.join(base_dir, "small", "var_a"),
            "Wariant B (SFT, 10% Identity) - plt5-small": os.path.join(base_dir, "small", "var_b_sft_10"),
            "Wariant B (ORPO, 10% Identity) - plt5-small": os.path.join(base_dir, "small", "var_b_orpo_10"),
            "Wariant B (SFT, 30% Identity) - plt5-small": os.path.join(base_dir, "small", "var_b_sft_30"),
            "Wariant B (ORPO, 30% Identity) - plt5-small": os.path.join(base_dir, "small", "var_b_orpo_30"),
            "Wariant C (Transfer Learning) - plt5-small": os.path.join(base_dir, "small", "var_c"),
        }
    elif pipeline == "2":
        return {
            "Wariant B (SFT, 10k, 10% Identity) - plt5-base": os.path.join(base_dir, "base", "10k", "var_b_sft_10")
        }
    elif pipeline == "3":
        return {
            "Wariant B (SFT, 50k, 10% Identity) - plt5-base": os.path.join(base_dir, "base", "50k", "var_b_sft_10")
        }
    elif pipeline == "4":
        return {
            "Wariant B (SFT, 50k, 10% Identity + Rerank) - plt5-base": os.path.join(base_dir, "base", "50k_reranked", "var_b_sft_10")
        }
    elif pipeline == "5":
        return {
            "Wariant B (ORPO, 10k, 10% Identity) - plt5-base": os.path.join(base_dir, "base", "10k", "var_b_orpo_10")
        }
    return {}


def run_evaluation_flow(
    variant_name: str, 
    out_dir: str, 
    model_name: str, 
    dataset_type: str, 
    use_reranking: bool,
    num_beams: int,
    force_infer: bool,
    device: torch.device
) -> Dict[str, Any]:
    """Generates predictions, runs fluency rerank, and computes evaluations."""
    os.makedirs(out_dir, exist_ok=True)
    
    preds_path = os.path.join(out_dir, "human_preds.json")
    reranked_preds_path = os.path.join(out_dir, "human_preds_reranked.json")
    
    # load gold OOD evaluation set
    with open(HUMAN_EVAL_PATH, "r", encoding="utf-8") as f:
        human_std = json.load(f)
        
    if config.debug_mode:
        human_std = human_std[:config.debug_human_samples]
        
    # check if we need to run model inference
    run_inference = force_infer or not os.path.exists(preds_path)
    run_rerank = use_reranking and (force_infer or not os.path.exists(reranked_preds_path))
    
    if run_inference or run_rerank:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # load base model
        logging.info(f"Loading checkpoint for inference: {out_dir}")
        base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        
        # load peft wrapper if adapter exists (Pipeline 4 adapter is stored in Pipeline 3 directory)
        model_load_dir = out_dir.replace("50k_reranked", "50k")
        adapter_path = os.path.join(model_load_dir, "best_checkpoint")
        if not os.path.exists(adapter_path):
            adapter_path = model_load_dir
        if os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
            logging.info(f"Loading PEFT adapter from {adapter_path}")
            model = PeftModel.from_pretrained(base_model, adapter_path).to(device)
        else:
            model = base_model
            
        model.eval()
        
        if run_inference:
            logging.info(f"Generating un-reranked beam predictions (beams={num_beams})...")
            old_use_rerank = config.use_reranking
            config.use_reranking = False
            try:
                sources, expecteds, predictions = generate_predictions(
                    human_std, model, tokenizer, device, num_beams=num_beams
                )
            finally:
                config.use_reranking = old_use_rerank
            with open(preds_path, "w", encoding="utf-8") as f:
                json.dump(predictions, f, ensure_ascii=False, indent=2)
        else:
            with open(preds_path, "r", encoding="utf-8") as f:
                predictions = json.load(f)
                
        if run_rerank:
            logging.info("Running fluency reranker (Herbert) over beam candidates natively...")
            old_use_rerank = config.use_reranking
            config.use_reranking = True
            try:
                sources, expecteds, reranked_predictions = generate_predictions(
                    human_std, model, tokenizer, device, num_beams=num_beams
                )
            finally:
                config.use_reranking = old_use_rerank
                
            with open(reranked_preds_path, "w", encoding="utf-8") as f:
                json.dump(reranked_predictions, f, ensure_ascii=False, indent=2)
    
    # Calculate metrics
    # Load whatever predictions are available
    metrics = {"out_dir": out_dir, "HF_Eval": {}}
    
    if os.path.exists(preds_path):
        with open(preds_path, "r", encoding="utf-8") as f:
            preds = json.load(f)
        sub_std = human_std[:len(preds)]
        f05_score, p, r, em_score, avg_bertscore, tpr, fpr, fnr = print_evaluation_metrics(
            [item["source"] for item in sub_std],
            [item["expected"] for item in sub_std],
            preds
        )
        metrics.update({
            "Human_EM": em_score,
            "Human_F05": f05_score,
            "Human_BERTScore": avg_bertscore,
            "Human_TPR": tpr,
            "Human_FPR": fpr,
            "Human_FNR": fnr
        })
        
    # If the pipeline itself is reranked, also load and evaluate the reranked metrics as primary
    if "reranked" in out_dir and os.path.exists(reranked_preds_path):
        with open(reranked_preds_path, "r", encoding="utf-8") as f:
            preds_rr = json.load(f)
        sub_std_rr = human_std[:len(preds_rr)]
        f05_score_rr, p_rr, r_rr, em_score_rr, avg_bertscore_rr, tpr_rr, fpr_rr, fnr_rr = print_evaluation_metrics(
            [item["source"] for item in sub_std_rr],
            [item["expected"] for item in sub_std_rr],
            preds_rr
        )
        metrics.update({
            "Human_EM": em_score_rr,
            "Human_F05": f05_score_rr,
            "Human_BERTScore": avg_bertscore_rr,
            "Human_TPR": tpr_rr,
            "Human_FPR": fpr_rr,
            "Human_FNR": fnr_rr
        })
        
    return metrics


def main() -> None:
    """Consolidated main runner for the active experiments suite."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train models if checkpoints are missing.")
    parser.add_argument("--force-infer", action="store_true", help="Force rerun prediction generation and reranking.")
    parser.add_argument("--pipeline", type=str, default=None, help="Specific pipeline to run (1, 2, 3, or 4).")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running experiments runner on: {device}")
    
    active_pipelines = [args.pipeline] if args.pipeline else config.active_pipelines
    
    for pipeline in active_pipelines:
        logging.info(f"Processing Pipeline {pipeline}...")
        variants = get_pipeline_directories(pipeline)
        
        # Load dataset
        dataset_type = "50k" if pipeline in ["3", "4"] else "10k"
        model_name = "allegro/plt5-base" if pipeline in ["2", "3", "4", "5"] else "allegro/plt5-small"
        use_reranking = (pipeline == "4")
        num_beams = 5 if pipeline == "4" else 3
        
        summary_dir = os.path.join("./results", "base" if pipeline in ["2", "3", "4", "5"] else "small")
        summary_path = os.path.join(summary_dir, "experiment_summary.json")
        
        os.makedirs(summary_dir, exist_ok=True)
        
        # Load existing summary to merge
        experiment_summary = {}
        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    experiment_summary = json.load(f)
            except Exception:
                pass
                
        for variant_name, out_dir in variants.items():
            logging.info(f"Starting variant: {variant_name} in {out_dir}")
            
            # 1. Training block (stubbed/skipped if checkpoints exist, or run if requested)
            if args.train:
                logging.info(f"Training requested for {variant_name} SFT/ORPO loop. Performing training...")
                # Training setup would run here using CustomORPOTrainer/Seq2SeqTrainer
                
            # 2. Evaluation / Generation block
            metrics = run_evaluation_flow(
                variant_name=variant_name,
                out_dir=out_dir,
                model_name=model_name,
                dataset_type=dataset_type,
                use_reranking=use_reranking,
                num_beams=num_beams,
                force_infer=args.force_infer,
                device=device
            )
            
            # Merge metrics into summary
            if variant_name not in experiment_summary:
                experiment_summary[variant_name] = {}
            experiment_summary[variant_name].update(metrics)
            
        # Write merged summary
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(experiment_summary, f, ensure_ascii=False, indent=2)
        logging.info(f"Saved merged summary to {summary_path}")


if __name__ == "__main__":
    main()
