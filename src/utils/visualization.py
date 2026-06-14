"""Utility module for rendering evaluation metrics, comparison tables, and plots."""
import os
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Tuple
from IPython.display import display, HTML
from src.models.evaluation import get_categorized_metrics

def shorten_name(name: str) -> str:
    """Shortens variant names for plotting labels."""
    # Remove prefix "Wariant "
    name = name.replace("Wariant ", "")
    # Map the common patterns
    name = name.replace("A (Tylko Błędy)", "A")
    name = name.replace("B (SFT, 10% Identity)", "B-SFT-10")
    name = name.replace("B (ORPO, 10% Identity)", "B-ORPO-10")
    name = name.replace("B (SFT, 30% Identity)", "B-SFT-30")
    name = name.replace("B (ORPO, 30% Identity)", "B-ORPO-30")
    name = name.replace("C (Transfer Learning)", "C")
    
    name = name.replace("B (SFT, 10k, 10% Identity)", "B-SFT-10k")
    name = name.replace("B (SFT, 50k, 10% Identity)", "B-SFT-50k")
    name = name.replace("B (SFT, 50k, 10% Identity + Rerank)", "B-SFT-50k+RR")
    name = name.replace("B (ORPO, 10k, 10% Identity)", "B-ORPO-10k")
    
    name = name.replace("Baseline (Do-Nothing)", "Baseline")
    name = name.replace("Baseline - Reguły + HerBERT", "Baseline+HerBERT")
    
    # Clean up model names
    name = name.replace(" - plt5-small", "-small")
    name = name.replace(" - plt5-base", "-base")
    
    return name


def render_results_table(final_results: Dict[str, Any], baseline_em: float, human_baseline: Dict[str, float]) -> pd.DataFrame:
    """Formats and displays a styled table of global metrics for all variants.

    Args:
        final_results: Merged dictionary of all pipeline metrics.
        baseline_em: Synthetic baseline exact match.
        human_baseline: Dictionary of human evaluation baseline metrics.

    Returns:
        The formatted pandas DataFrame.
    """
    results_data = {
        "Wariant": ["Baseline (Do-Nothing)"],
        "Train loss": [None],
        "Val loss": [None],
        "Test EM (Synth)": [baseline_em],
        "OOD EM": [human_baseline.get("em", 0.0)],
        "OOD F0.5": [human_baseline.get("f05", 0.0)],
        "OOD BERTScore": [human_baseline.get("bertscore", 0.0)],
        "OOD TPR": [human_baseline.get("tpr", 0.0)],
        "OOD FPR": [human_baseline.get("fpr", 0.0)],
        "OOD FNR": [human_baseline.get("fnr", 1.0)]
    }

    for exp_name, metrics in final_results.items():
        hf_eval = metrics.get('HF_Eval', {})
        results_data['Wariant'].append(exp_name)
        results_data['Train loss'].append(metrics.get('Train_Loss'))
        results_data['Val loss'].append(hf_eval.get('eval_loss', None))
        results_data['Test EM (Synth)'].append(hf_eval.get('eval_exact_match', 0))
        results_data['OOD EM'].append(metrics.get('OOD_EM', metrics.get('Human_EM', 0)))
        results_data['OOD F0.5'].append(metrics.get('OOD_F05', metrics.get('Human_F05', 0)))
        results_data['OOD BERTScore'].append(metrics.get('OOD_BERTScore', metrics.get('Human_BERTScore', 0.0)))
        results_data['OOD TPR'].append(metrics.get('OOD_TPR', metrics.get('Human_TPR', 0.0)))
        results_data['OOD FPR'].append(metrics.get('OOD_FPR', metrics.get('Human_FPR', 0.0)))
        results_data['OOD FNR'].append(metrics.get('OOD_FNR', metrics.get('Human_FNR', 0.0)))

    results_df = pd.DataFrame(results_data)
    display(HTML("<h3>Tabela Wyników Globalnych</h3>"))
    display(results_df.style.format({
        "Train loss": "{:.4f}", 
        "Val loss": "{:.4f}", 
        "Test EM (Synth)": "{:.2%}", 
        "OOD EM": "{:.2%}", 
        "OOD F0.5": "{:.4f}", 
        "OOD BERTScore": "{:.4f}", 
        "OOD TPR": "{:.2%}", 
        "OOD FPR": "{:.2%}", 
        "OOD FNR": "{:.2%}"
    }, na_rep="-"))
    return results_df


def render_categorized_metrics_table(final_results: Dict[str, Any], gold_srcs: List[str], gold_exps: List[str]) -> None:
    """Computes and displays error categorized metrics for both normal and reranked predictions.

    Args:
        final_results: Merged pipeline statistics.
        gold_srcs: Gold standard source sentences.
        gold_exps: Gold standard target sentences.
    """
    baseline_metrics = get_categorized_metrics(gold_srcs, gold_exps, gold_srcs)
    categorized_table_data = []

    def add_row(name, cat_metrics):
        categorized_table_data.append({
            "Wariant": name,
            "Prep (TPR)": cat_metrics.get("prep", {}).get("tpr", 0.0),
            "False friend (TPR)": cat_metrics.get("false_friend", {}).get("tpr", 0.0),
            "Gender (TPR)": cat_metrics.get("gender", {}).get("tpr", 0.0),
            "Case (TPR)": cat_metrics.get("case", {}).get("tpr", 0.0),
            "Typos (TPR)": cat_metrics.get("typos", {}).get("tpr", 0.0),
            "Other (TPR)": cat_metrics.get("other", {}).get("tpr", 0.0),
            "Identity (FPR)": cat_metrics.get("identity", {}).get("fpr", 0.0)
        })

    # add baseline row
    add_row("Baseline (Do-Nothing)", baseline_metrics)

    # process all original experiments
    for exp_name, metrics in final_results.items():
        out_dir = metrics.get("out_dir", "")
        # fallback folders logic
        if not os.path.exists(out_dir) and "results/small/" in out_dir:
            out_dir = out_dir.replace("results/small/", "results-small/")
            
        # 1. un-reranked prediction evaluation
        preds_path = os.path.join(out_dir, "human_preds.json")
        if os.path.exists(preds_path):
            try:
                with open(preds_path, "r", encoding="utf-8") as f:
                    preds = json.load(f)
                cat_metrics = get_categorized_metrics(gold_srcs[:len(preds)], gold_exps[:len(preds)], preds)
                add_row(exp_name, cat_metrics)
            except Exception:
                pass
                
        # 2. reranked prediction evaluation (Herbert Rank Rematching)
        reranked_preds_path = os.path.join(out_dir, "human_preds_reranked.json")
        if os.path.exists(reranked_preds_path):
            try:
                with open(reranked_preds_path, "r", encoding="utf-8") as f:
                    preds = json.load(f)
                cat_metrics = get_categorized_metrics(gold_srcs[:len(preds)], gold_exps[:len(preds)], preds)
                add_row(f"{exp_name} + Herbert Rerank", cat_metrics)
            except Exception:
                pass

    cat_df = pd.DataFrame(categorized_table_data)
    display(HTML("<h3>Szczegółowa Ewaluacja Kategorii Błędów</h3>"))
    display(cat_df.style.format({
        "Prep (TPR)": "{:.2%}",
        "False Friend (TPR)": "{:.2%}",
        "Gender (TPR)": "{:.2%}",
        "Case (TPR)": "{:.2%}",
        "Typos (TPR)": "{:.2%}",
        "Other (TPR)": "{:.2%}",
        "Identity (FPR)": "{:.2%}"
    }))


def plot_results_comparison(results_df: pd.DataFrame) -> None:
    """Generates comparison bar plots for synthetic EM, OOD F0.5, TPR, and FPR.

    Args:
        results_df: DataFrame of results compiled by render_results_table.
    """
    short_names = [shorten_name(name) for name in results_df["Wariant"]]
    x = np.arange(len(results_df["Wariant"]))
    width = 0.35

    # Plot 1: Compare Test EM (Synth) and OOD F0.5
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, results_df["Test EM (Synth)"], width, label='Test EM (Synth)', color='#3498db')
    rects2 = ax.bar(x + width/2, results_df["OOD F0.5"], width, label='OOD F0.5 Score', color='#2ecc71')

    ax.set_ylabel('Score')
    ax.set_title('Porównanie Wyników: Test EM (Syntetyczny) vs OOD F0.5')
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=15, ha='right')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2)
    ax.set_ylim(0, 1.1)

    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2%}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.show()

    # Plot 2: Compare TPR and FPR
    fig, ax = plt.subplots(figsize=(12, 6))
    rects_tpr = ax.bar(x - width/2, results_df["OOD TPR"], width, label='TPR (Correctly Fixed)', color='#2ecc71')
    rects_fpr = ax.bar(x + width/2, results_df["OOD FPR"], width, label='FPR (Sentences Ruined)', color='#e74c3c')

    ax.set_ylabel('Rate')
    ax.set_title('Czułość (TPR) vs Błędy Nadkorekcji (FPR)')
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=15, ha='right')
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=2)
    ax.set_ylim(0, 1.1)

    for rects in [rects_tpr, rects_fpr]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2%}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.show()


def display_beautiful_metrics(srcs: List[str], exps: List[str], preds: List[str]) -> Tuple[float, float, float, float, float, float, float, float]:
    """Computes GEC metrics and displays them in styled HTML/Pandas tables inside Jupyter."""
    from src.models.evaluation import calculate_f05, get_bertscore_metric, _normalize_for_em, _get_classification_resources, classify_error
    from IPython.display import display, HTML
    import random
    
    # 1. Calculate Global Metrics
    p, r, f05_score = calculate_f05(srcs, exps, preds)
    exact_matches = sum(1 for p_str, e in zip(preds, exps) if _normalize_for_em(p_str) == _normalize_for_em(e))
    em_score = exact_matches / len(exps) if len(exps) > 0 else 0.0
    
    err_srcs, err_exps, err_preds = [], [], []
    id_srcs, id_exps, id_preds = [], [], []
    
    for s, e, p_str in zip(srcs, exps, preds):
        if s.strip() != e.strip():
            err_srcs.append(s)
            err_exps.append(e)
            err_preds.append(p_str)
        else:
            id_srcs.append(s)
            id_exps.append(e)
            id_preds.append(p_str)
            
    err_p, err_r, err_f05 = calculate_f05(err_srcs, err_exps, err_preds) if err_srcs else (0.0, 0.0, 0.0)
    err_em_count = sum(1 for p_str, e in zip(err_preds, err_exps) if _normalize_for_em(p_str) == _normalize_for_em(e))
    tpr = err_em_count / len(err_exps) if len(err_exps) > 0 else 0.0
    fnr = 1.0 - tpr
    
    id_em_count = sum(1 for p_str, e in zip(id_preds, id_exps) if _normalize_for_em(p_str) == _normalize_for_em(e))
    id_em = id_em_count / len(id_exps) if len(id_exps) > 0 else 0.0
    fpr = 1.0 - id_em
    
    bs_results = get_bertscore_metric().compute(predictions=preds, references=exps, lang="pl")
    avg_bertscore = sum(bs_results["f1"]) / len(bs_results["f1"]) if bs_results["f1"] else 0.0
    
    # Render Global Metrics Table
    global_data = {
        "Subset": ["Global", "Global", "Global", "Global", "Global", "Errors", "Errors", "Clean", "Clean"],
        "Metric Name": ["Overall exact match", "Precision", "Recall", "F0.5 score", "BERTScore F1", "TPR", "F0.5 (correction)", "FPR", "Exact match (unchanged)"],
        "Value": [em_score, p, r, f05_score, avg_bertscore, tpr, err_f05, fpr, id_em]
    }
    global_df = pd.DataFrame(global_data)
    
    display(HTML("<h4>Ewaluacja - Metryki Globalne</h4>"))
    
    # helper for clean float vs percentage formatting
    def format_val(row):
        val = row["Value"]
        name = row["Metric Name"]
        if "match" in name or "Rate" in name or name in ["TPR", "FPR"]:
            return f"{val:.2%}"
        return f"{val:.4f}"
        
    global_df["Value Formatted"] = global_df.apply(format_val, axis=1)
    display(global_df[["Subset", "Metric Name", "Value Formatted"]].style.hide(axis='index'))
    
    # 2. Categorized Metrics Table
    gender_mismatches, false_friends_words = _get_classification_resources()
    category_groups = {"prep": [], "false_friend": [], "gender": [], "case": [], "typos": [], "other": [], "identity": []}
    
    for s, e, pred in zip(srcs, exps, preds):
        cat = classify_error(s, e, gender_mismatches, false_friends_words)
        if cat in category_groups:
            category_groups[cat].append((s, e, pred))
            
    cat_rows = []
    for cat, items in category_groups.items():
        if not items:
            continue
        cat_srcs = [it[0] for it in items]
        cat_exps = [it[1] for it in items]
        cat_preds = [it[2] for it in items]
        
        display_cat = cat.replace("_", " ").capitalize()
        if cat == "identity":
            cat_em_count = sum(1 for p_str, e in zip(cat_preds, cat_exps) if _normalize_for_em(p_str) == _normalize_for_em(e))
            cat_em = cat_em_count / len(cat_exps)
            cat_fpr = 1.0 - cat_em
            cat_rows.append({
                "Kategoria": "Identity",
                "Próbki": len(items),
                "F0.5": None,
                "Precision": None,
                "Recall": None,
                "TPR / FPR": cat_fpr
            })
        else:
            cat_p, cat_r, cat_f05 = calculate_f05(cat_srcs, cat_exps, cat_preds)
            cat_em_count = sum(1 for p_str, e in zip(cat_preds, cat_exps) if _normalize_for_em(p_str) == _normalize_for_em(e))
            cat_tpr = cat_em_count / len(cat_exps)
            cat_rows.append({
                "Kategoria": display_cat,
                "Próbki": len(items),
                "F0.5": cat_f05,
                "Precision": cat_p,
                "Recall": cat_r,
                "TPR / FPR": cat_tpr
            })
            
    cat_df = pd.DataFrame(cat_rows)
    display(HTML("<h4>Ewaluacja - Szczegółowo po Kategoriach Błędów</h4>"))
    display(cat_df.style.format({
        "F0.5": "{:.4f}",
        "Precision": "{:.4f}",
        "Recall": "{:.4f}",
        "TPR / FPR": "{:.2%}"
    }, na_rep="-").hide(axis='index'))
    
    # 3. Sample Predictions Table
    true_positives = [(s, e, p) for s, e, p in zip(err_srcs, err_exps, err_preds) if _normalize_for_em(p) == _normalize_for_em(e)]
    false_negatives = [(s, e, p) for s, e, p in zip(err_srcs, err_exps, err_preds) if _normalize_for_em(p) != _normalize_for_em(e)]
    false_positives = [(s, e, p) for s, e, p in zip(id_srcs, id_exps, id_preds) if _normalize_for_em(p) != _normalize_for_em(e)]
    
    random.seed(42)
    random.shuffle(true_positives)
    random.shuffle(false_negatives)
    random.shuffle(false_positives)
    
    sample_rows = []
    for s, e, p_str in true_positives[:2]:
        sample_rows.append({"Typ próbki": "True positive", "Wejście": s, "Oczekiwane": e, "Wyjście": p_str})
    for s, e, p_str in false_negatives[:2]:
        sample_rows.append({"Typ próbki": "False negative", "Wejście": s, "Oczekiwane": e, "Wyjście": p_str})
    for s, e, p_str in false_positives[:2]:
        sample_rows.append({"Typ próbki": "False positive", "Wejście": s, "Oczekiwane": e, "Wyjście": p_str})
        
    if sample_rows:
        samples_df = pd.DataFrame(sample_rows)
        display(HTML("<h4>Przykładowe Wyniki Predykcji</h4>"))
        display(samples_df.style.set_properties(**{'text-align': 'left'}).hide(axis='index'))
        
    return f05_score, p, r, em_score, avg_bertscore, tpr, fpr, fnr


def plot_categorized_metrics_comparison(final_results: Dict[str, Any], gold_srcs: List[str], gold_exps: List[str], reranked: bool = False) -> None:
    """Plots side-by-side grouped bar charts (Regular vs Reranked) for each error category."""
    # To prevent plotting twice if called twice (with reranked=False and reranked=True in different cells):
    if reranked:
        return

    from src.models.evaluation import get_categorized_metrics
    import matplotlib.ticker as mtick

    categories = ["prep", "false_friend", "gender", "case", "typos", "other", "identity"]
    cat_titles = {
        "prep": "Prep (TPR)",
        "false_friend": "False Friend (TPR)",
        "gender": "Gender (TPR)",
        "case": "Case (TPR)",
        "typos": "Typos (TPR)",
        "other": "Other (TPR)",
        "identity": "Identity (FPR)"
    }

    # 1. Collect all models and their predictions
    model_preds = []

    # Add Baseline (Do-Nothing)
    model_preds.append(("Baseline", gold_srcs, None))

    # Add rule-based baseline reranked if it exists
    baseline_rr_dir = "./results/base/baseline_rules_reranked"
    baseline_rr_path = os.path.join(baseline_rr_dir, "human_preds_reranked.json")
    if os.path.exists(baseline_rr_path):
        try:
            with open(baseline_rr_path, "r", encoding="utf-8") as f:
                baseline_rr_preds = json.load(f)
            model_preds.append(("Baseline+HerBERT", None, baseline_rr_preds))
        except Exception:
            pass

    # Sort final_results keys to match our desired order
    key_order = [
        "Wariant A (Tylko Błędy) - plt5-small",
        "Wariant B (SFT, 10% Identity) - plt5-small",
        "Wariant B (ORPO, 10% Identity) - plt5-small",
        "Wariant B (SFT, 30% Identity) - plt5-small",
        "Wariant B (ORPO, 30% Identity) - plt5-small",
        "Wariant C (Transfer Learning) - plt5-small",
        "Wariant B (SFT, 10k, 10% Identity) - plt5-base",
        "Wariant B (ORPO, 10k, 10% Identity) - plt5-base",
        "Wariant B (SFT, 50k, 10% Identity) - plt5-base",
        "Wariant B (SFT, 50k, 10% Identity + Rerank) - plt5-base",
        "Wariant B (ORPO, 10k, 10% Identity) - plt5-base"
    ]
    
    sorted_keys = [k for k in key_order if k in final_results]
    for k in final_results.keys():
        if k not in sorted_keys and k != "Baseline - Reguły + HerBERT":
            sorted_keys.append(k)

    for exp_name in sorted_keys:
        metrics = final_results[exp_name]
        out_dir = metrics.get("out_dir", "")
        if not os.path.exists(out_dir) and "results/small/" in out_dir:
            out_dir = out_dir.replace("results/small/", "results-small/")

        preds_reg = None
        preds_path = os.path.join(out_dir, "human_preds.json")
        if os.path.exists(preds_path):
            try:
                with open(preds_path, "r", encoding="utf-8") as f:
                    preds_reg = json.load(f)
            except Exception:
                pass

        preds_rr = None
        preds_rr_path = os.path.join(out_dir, "human_preds_reranked.json")
        if os.path.exists(preds_rr_path):
            try:
                with open(preds_rr_path, "r", encoding="utf-8") as f:
                    preds_rr = json.load(f)
            except Exception:
                pass

        if preds_reg is not None or preds_rr is not None:
            short_name = shorten_name(exp_name)
            model_preds.append((short_name, preds_reg, preds_rr))

    # For each category/metric, draw a side-by-side plot
    for cat in categories:
        reg_labels = []
        reg_vals = []
        rr_labels = []
        rr_vals = []

        for label, preds_reg, preds_rr in model_preds:
            if preds_reg is not None:
                cat_metrics = get_categorized_metrics(gold_srcs[:len(preds_reg)], gold_exps[:len(preds_reg)], preds_reg)
                val = cat_metrics.get(cat, {}).get("fpr" if cat == "identity" else "tpr", 0.0)
                reg_labels.append(label)
                reg_vals.append(val)
            
            if preds_rr is not None:
                cat_metrics = get_categorized_metrics(gold_srcs[:len(preds_rr)], gold_exps[:len(preds_rr)], preds_rr)
                val = cat_metrics.get(cat, {}).get("fpr" if cat == "identity" else "tpr", 0.0)
                rr_labels.append(label)
                rr_vals.append(val)

        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(16, 5))
        
        c_reg = "#3498db"
        c_rr = "#2ecc71"
        if cat == "identity":
            c_reg = "#e74c3c"
            c_rr = "#e67e22"

        # Left: Regular
        if reg_vals:
            bars1 = ax1.bar(reg_labels, reg_vals, color=c_reg, width=0.5)
            ax1.set_title(f"{cat_titles[cat]} - Wersje Standardowe", fontsize=12, fontweight='bold')
            ax1.set_ylabel("Wskaźnik", fontsize=10)
            ax1.set_ylim(0, 1.1)
            ax1.grid(True, linestyle='--', alpha=0.5, axis='y')
            ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax1.set_xticks(range(len(reg_labels)))
            ax1.set_xticklabels(reg_labels, rotation=45, ha='right', fontsize=9)
            for bar in bars1:
                height = bar.get_height()
                ax1.annotate(f'{height:.1%}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, fontweight='bold')
        else:
            ax1.text(0.5, 0.5, "Brak danych", ha='center', va='center')

        # Right: Reranked
        if rr_vals:
            bars2 = ax2.bar(rr_labels, rr_vals, color=c_rr, width=0.5)
            ax2.set_title(f"{cat_titles[cat]} - Po Rerankingu HerBERT", fontsize=12, fontweight='bold')
            ax2.set_ylabel("Wskaźnik", fontsize=10)
            ax2.set_ylim(0, 1.1)
            ax2.grid(True, linestyle='--', alpha=0.5, axis='y')
            ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            ax2.set_xticks(range(len(rr_labels)))
            ax2.set_xticklabels(rr_labels, rotation=45, ha='right', fontsize=9)
            for bar in bars2:
                height = bar.get_height()
                ax2.annotate(f'{height:.1%}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=8, fontweight='bold')
        else:
            ax2.text(0.5, 0.5, "Brak danych", ha='center', va='center')

        plt.tight_layout()
        plt.show()


def plot_training_curves(final_results: Dict[str, Any]) -> None:
    """Plots training and validation loss curves from trainer_state.json dynamically."""
    valid_plots = []
    for exp_name, metrics in final_results.items():
        m_dir = metrics.get('out_dir')
        if not m_dir:
            continue
        state_path = os.path.join(m_dir, "trainer_state.json")
        if not os.path.exists(state_path) and "results/small/" in m_dir:
            fallback_dir = m_dir.replace("results/small/", "results-small/")
            state_path = os.path.join(fallback_dir, "trainer_state.json")
        if os.path.exists(state_path):
            valid_plots.append((state_path, exp_name))

    N = len(valid_plots)
    if N == 0:
        print("Brak danych krzywych uczenia (brak plików trainer_state.json).")
        return

    ncols = 2
    nrows = (N + 1) // 2

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 4 * nrows))
    if N == 1:
        axes = np.array([axes]) if not isinstance(axes, np.ndarray) else axes
    axes = axes.flatten()

    for idx, (state_path, title) in enumerate(valid_plots):
        ax = axes[idx]
        with open(state_path, 'r', encoding='utf-8') as f:
            state = json.load(f)

        history = state.get('log_history', [])
        train_steps = [h['step'] for h in history if 'loss' in h]
        train_loss = [h['loss'] for h in history if 'loss' in h]
        val_steps = [h['step'] for h in history if 'eval_loss' in h]
        val_loss = [h['eval_loss'] for h in history if 'eval_loss' in h]

        if train_steps: 
            ax.plot(train_steps, train_loss, label='Loss (trening)', color='#3498db')
        if val_steps: 
            ax.plot(val_steps, val_loss, label='Loss (walidacja)', marker='o', color='#e74c3c')

        ax.set_title(shorten_name(title), fontsize=11, fontweight='bold')
        ax.set_xlabel('Kroki (Steps)', fontsize=9)
        ax.set_ylabel('Funkcja straty (Loss)', fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.6)

    for idx in range(N, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.show()


def display_sample_corrections(human_gold_standard: List[Dict[str, Any]]) -> None:
    """Displays OOD sample GEC corrections in a formatted Pandas table."""
    p_baseline_path = "./results/base/baseline_rules_reranked/human_preds_reranked.json"
    p_var_a_path = "./results/small/var_a/human_preds.json"
    if not os.path.exists(p_var_a_path):
        p_var_a_path = "./results-small/var_a/human_preds.json"

    p_sft_50k_path = "./results/base/50k/var_b_sft_10/human_preds.json"
    p_rerank_50k_path = "./results/base/50k/var_b_sft_10/human_preds_reranked.json"

    paths = {
        "Baseline+HerBERT": p_baseline_path,
        "Wariant A": p_var_a_path,
        "SFT 50k": p_sft_50k_path,
        "SFT 50k + Rerank": p_rerank_50k_path
    }

    preds_data = {}
    for name, path in paths.items():
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                preds_data[name] = json.load(f)
        else:
            preds_data[name] = []

    samples = []
    count = 0
    for idx, item in enumerate(human_gold_standard):
        src = item["source"]
        exp = item["expected"]
        if src != exp and count < 8:
            sample_row = {
                "Wejście": src,
                "Oczekiwane": exp,
                "Baseline+HerBERT": preds_data["Baseline+HerBERT"][idx] if idx < len(preds_data["Baseline+HerBERT"]) else "-",
                "Wariant A": preds_data["Wariant A"][idx] if idx < len(preds_data["Wariant A"]) else "-",
                "SFT 50k": preds_data["SFT 50k"][idx] if idx < len(preds_data["SFT 50k"]) else "-",
                "SFT 50k + Rerank": preds_data["SFT 50k + Rerank"][idx] if idx < len(preds_data["SFT 50k + Rerank"]) else "-"
            }
            samples.append(sample_row)
            count += 1

    if samples:
        samples_df = pd.DataFrame(samples)
        pd.set_option('display.max_colwidth', None)
        print("Przykłady korekt generowanych przez modele (Baseline, Wariant A, SFT 50k oraz Reranked):")
        display(samples_df)
        pd.reset_option('display.max_colwidth')
    else:
        print("Brak plików predykcji do wyświetlenia próbek.")


def display_beautiful_metrics_comparison(srcs: List[str], exps: List[str], preds_reg: List[str], preds_rr: List[str] = None, title: str = "") -> None:
    """Computes and displays merged GEC metrics (Standard vs Reranked) in formatted tables."""
    from src.models.evaluation import calculate_f05, get_bertscore_metric, _normalize_for_em, _get_classification_resources, classify_error
    from IPython.display import display, HTML
    import pandas as pd
    import random

    display(HTML(f"<hr style='border-top: 2px solid #34495e; margin-top: 30px; margin-bottom: 20px;'/>"
                 f"<h3 style='color: #2c3e50; font-weight: bold;'>Szczegółowa Ewaluacja: {title}</h3>"))

    # Helper to calculate global metrics
    def calc_global(preds):
        if not preds:
            return [0]*9
        p, r, f05 = calculate_f05(srcs, exps, preds)
        em = sum(1 for p_str, e in zip(preds, exps) if _normalize_for_em(p_str) == _normalize_for_em(e)) / len(exps)
        
        err_srcs, err_exps, err_preds = [], [], []
        id_srcs, id_exps, id_preds = [], [], []
        for s, e, p_str in zip(srcs, exps, preds):
            if s.strip() != e.strip():
                err_srcs.append(s)
                err_exps.append(e)
                err_preds.append(p_str)
            else:
                id_srcs.append(s)
                id_exps.append(e)
                id_preds.append(p_str)
        
        err_p, err_r, err_f05 = calculate_f05(err_srcs, err_exps, err_preds) if err_srcs else (0.0, 0.0, 0.0)
        tpr = sum(1 for p_str, e in zip(err_preds, err_exps) if _normalize_for_em(p_str) == _normalize_for_em(e)) / len(err_exps) if err_exps else 0.0
        
        id_em = sum(1 for p_str, e in zip(id_preds, id_exps) if _normalize_for_em(p_str) == _normalize_for_em(e)) / len(id_exps) if id_exps else 0.0
        fpr = 1.0 - id_em
        
        bs_results = get_bertscore_metric().compute(predictions=preds, references=exps, lang="pl")
        avg_bs = sum(bs_results["f1"]) / len(bs_results["f1"]) if bs_results["f1"] else 0.0
        
        return [em, p, r, f05, avg_bs, tpr, err_f05, fpr, id_em]

    vals_reg = calc_global(preds_reg)
    vals_rr = calc_global(preds_rr) if preds_rr else [None]*9

    metric_names = [
        "Overall exact match", "Precision", "Recall", "F0.5 score", 
        "BERTScore F1", "TPR", "F0.5 (correction)", "FPR", "Exact match (unchanged)"
    ]
    subsets = ["Global", "Global", "Global", "Global", "Global", "Errors", "Errors", "Clean", "Clean"]

    global_rows = []
    for idx, name in enumerate(metric_names):
        val_reg = vals_reg[idx]
        val_rr = vals_rr[idx]
        
        def fmt(val):
            if val is None:
                return "-"
            if "match" in name or "Rate" in name or name in ["TPR", "FPR", "Exact match (unchanged)"]:
                return f"{val:.2%}"
            return f"{val:.4f}"
            
        global_rows.append({
            "Subset": subsets[idx],
            "Metric Name": name,
            "Standard": fmt(val_reg),
            "Reranked": fmt(val_rr)
        })
        
    global_df = pd.DataFrame(global_rows)
    display(HTML("<h4>Metryki Globalne (Standard vs Reranked)</h4>"))
    display(global_df.style.hide(axis='index'))

    # 2. Categorized metrics computation
    gender_mismatches, false_friends_words = _get_classification_resources()
    
    def calc_cat(preds):
        category_groups = {"prep": [], "false_friend": [], "gender": [], "case": [], "typos": [], "other": [], "identity": []}
        for s, e, pred in zip(srcs, exps, preds):
            cat = classify_error(s, e, gender_mismatches, false_friends_words)
            if cat in category_groups:
                category_groups[cat].append((s, e, pred))
        
        cat_results = {}
        for cat, items in category_groups.items():
            if not items:
                cat_results[cat] = {"count": 0, "f05": None, "tpr_fpr": None}
                continue
            cat_srcs = [it[0] for it in items]
            cat_exps = [it[1] for it in items]
            cat_preds = [it[2] for it in items]
            
            if cat == "identity":
                cat_em = sum(1 for p_str, e in zip(cat_preds, cat_exps) if _normalize_for_em(p_str) == _normalize_for_em(e)) / len(cat_exps)
                cat_results[cat] = {"count": len(items), "f05": None, "tpr_fpr": 1.0 - cat_em}
            else:
                cat_p, cat_r, cat_f05 = calculate_f05(cat_srcs, cat_exps, cat_preds)
                cat_tpr = sum(1 for p_str, e in zip(cat_preds, cat_exps) if _normalize_for_em(p_str) == _normalize_for_em(e)) / len(cat_exps)
                cat_results[cat] = {"count": len(items), "f05": cat_f05, "tpr_fpr": cat_tpr}
        return cat_results

    cat_reg = calc_cat(preds_reg)
    cat_rr = calc_cat(preds_rr) if preds_rr else {}

    cat_rows = []
    categories = ["prep", "false_friend", "gender", "case", "typos", "other", "identity"]
    for cat in categories:
        reg_res = cat_reg.get(cat, {"count": 0, "f05": None, "tpr_fpr": None})
        rr_res = cat_rr.get(cat, {"count": 0, "f05": None, "tpr_fpr": None})
        
        display_cat = cat.replace("_", " ").capitalize() if cat != "identity" else "Identity"
        
        def fmt_f05(v):
            return f"{v:.4f}" if v is not None else "-"
        def fmt_tpr_fpr(v):
            return f"{v:.2%}" if v is not None else "-"

        cat_rows.append({
            "Kategoria": display_cat,
            "Próbki": reg_res["count"],
            "Standard F0.5": fmt_f05(reg_res["f05"]),
            "Reranked F0.5": fmt_f05(rr_res.get("f05")),
            "Standard TPR/FPR": fmt_tpr_fpr(reg_res["tpr_fpr"]),
            "Reranked TPR/FPR": fmt_tpr_fpr(rr_res.get("tpr_fpr"))
        })

    cat_df = pd.DataFrame(cat_rows)
    display(HTML("<h4>Metryki po Kategoriach Błędów (Standard vs Reranked)</h4>"))
    display(cat_df.style.hide(axis='index'))

    # 3. Sample predictions comparison table
    if preds_rr:
        # Find matches / mismatches
        err_srcs, err_exps, err_preds_reg, err_preds_rr = [], [], [], []
        id_srcs, id_exps, id_preds_reg, id_preds_rr = [], [], [], []
        
        for s, e, pr, pr_rr in zip(srcs, exps, preds_reg, preds_rr):
            if s.strip() != e.strip():
                err_srcs.append(s)
                err_exps.append(e)
                err_preds_reg.append(pr)
                err_preds_rr.append(pr_rr)
            else:
                id_srcs.append(s)
                id_exps.append(e)
                id_preds_reg.append(pr)
                id_preds_rr.append(pr_rr)

        tp_reg_fixed = []
        fn_still_err = []
        fp_over_corr = []

        for s, e, pr, pr_rr in zip(err_srcs, err_exps, err_preds_reg, err_preds_rr):
            fixed_reg = (_normalize_for_em(pr) == _normalize_for_em(e))
            fixed_rr = (_normalize_for_em(pr_rr) == _normalize_for_em(e))
            
            if fixed_rr and not fixed_reg:
                tp_reg_fixed.append(("Poprawione przez Rerank", s, e, pr, pr_rr))
            elif fixed_reg and fixed_rr:
                tp_reg_fixed.append(("Poprawione przez oba", s, e, pr, pr_rr))
            else:
                fn_still_err.append(("Brak poprawy", s, e, pr, pr_rr))

        for s, e, pr, pr_rr in zip(id_srcs, id_exps, id_preds_reg, id_preds_rr):
            ruined_reg = (_normalize_for_em(pr) != _normalize_for_em(e))
            ruined_rr = (_normalize_for_em(pr_rr) != _normalize_for_em(e))
            if ruined_rr or ruined_reg:
                fp_over_corr.append(("Nadkorekcja", s, e, pr, pr_rr))

        random.seed(42)
        random.shuffle(tp_reg_fixed)
        random.shuffle(fn_still_err)
        random.shuffle(fp_over_corr)

        sample_rows = []
        for typ, s, e, pr, pr_rr in tp_reg_fixed[:3]:
            sample_rows.append({"Typ próbki": typ, "Wejście": s, "Oczekiwane": e, "Standard": pr, "Reranked": pr_rr})
        for typ, s, e, pr, pr_rr in fn_still_err[:3]:
            sample_rows.append({"Typ próbki": typ, "Wejście": s, "Oczekiwane": e, "Standard": pr, "Reranked": pr_rr})
        for typ, s, e, pr, pr_rr in fp_over_corr[:2]:
            sample_rows.append({"Typ próbki": typ, "Wejście": s, "Oczekiwane": e, "Standard": pr, "Reranked": pr_rr})

        if sample_rows:
            samples_df = pd.DataFrame(sample_rows)
            display(HTML("<h4>Przykładowe Wyniki (Standard vs Reranked)</h4>"))
            display(samples_df.style.set_properties(**{'text-align': 'left'}).hide(axis='index'))


def display_detailed_evaluations(final_results: Dict[str, Any], human_srcs: List[str], human_exps: List[str]) -> None:
    """Computes and displays merged evaluation tables for standard and reranked versions of Pipeline 1."""
    import os
    import json

    warianty_p1 = [
        "Wariant A (Tylko Błędy) - plt5-small",
        "Wariant B (SFT, 10% Identity) - plt5-small",
        "Wariant B (ORPO, 10% Identity) - plt5-small",
        "Wariant C (Transfer Learning) - plt5-small"
    ]

    for w_name in warianty_p1:
        if w_name in final_results:
            metrics = final_results[w_name]
            out_dir = metrics.get("out_dir", "")

            if not os.path.exists(out_dir) and "results/small/" in out_dir:
                out_dir = out_dir.replace("results/small/", "results-small/")

            # Load predictions
            c_preds = None
            preds_path = os.path.join(out_dir, "human_preds.json")
            if os.path.exists(preds_path):
                with open(preds_path, "r", encoding="utf-8") as f:
                    c_preds = json.load(f)

            c_preds_rr = None
            preds_rr_path = os.path.join(out_dir, "human_preds_reranked.json")
            if os.path.exists(preds_rr_path):
                with open(preds_rr_path, "r", encoding="utf-8") as f:
                    c_preds_rr = json.load(f)

            if c_preds is not None:
                display_beautiful_metrics_comparison(
                    human_srcs[:len(c_preds)], 
                    human_exps[:len(c_preds)], 
                    c_preds, 
                    c_preds_rr, 
                    w_name
                )





