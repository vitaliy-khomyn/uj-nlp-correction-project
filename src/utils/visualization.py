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
    name = name.replace("Wariant A (Tylko Błędy)", "Var A")
    name = name.replace("Wariant B (SFT, 10% Identity)", "Var B SFT 10%")
    name = name.replace("Wariant B (ORPO, 10% Identity)", "Var B ORPO 10%")
    name = name.replace("Wariant B (SFT, 30% Identity)", "Var B SFT 30%")
    name = name.replace("Wariant B (ORPO, 30% Identity)", "Var B ORPO 30%")
    name = name.replace("Wariant C (Transfer Learning)", "Var C")
    name = name.replace("Wariant B (SFT, 10k, 10% Identity)", "Var B SFT 10k 10%")
    name = name.replace("Wariant B (SFT, 50k, 10% Identity)", "Var B SFT 50k 10%")
    name = name.replace("Wariant B (SFT, 50k, 10% Identity + Rerank)", "Var B SFT 50k 10%+RR")
    name = name.replace("Wariant B (ORPO, 10k, 10% Identity)", "Var B ORPO 10k 10%")
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
        "Train Loss": [None],
        "Val Loss": [None],
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
        results_data['Train Loss'].append(metrics.get('Train_Loss'))
        results_data['Val Loss'].append(hf_eval.get('eval_loss', None))
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
        "Train Loss": "{:.4f}", 
        "Val Loss": "{:.4f}", 
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
            "False Friend (TPR)": cat_metrics.get("false_friend", {}).get("tpr", 0.0),
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
        "Subset": ["Global", "Global", "Global", "Global", "Global", "Errors (Wadliwe)", "Errors (Wadliwe)", "Clean (Poprawne)", "Clean (Poprawne)"],
        "Metric Name": ["Overall Exact Match", "Precision", "Recall", "F0.5 Score", "BERTScore F1", "True Positive Rate (TPR)", "F0.5 (Correction)", "False Positive Rate (FPR)", "Exact Match (Unchanged)"],
        "Value": [em_score, p, r, f05_score, avg_bertscore, tpr, err_f05, fpr, id_em]
    }
    global_df = pd.DataFrame(global_data)
    
    display(HTML("<h4>Ewaluacja - Metryki Globalne</h4>"))
    
    # helper for clean float vs percentage formatting
    def format_val(row):
        val = row["Value"]
        name = row["Metric Name"]
        if "Match" in name or "Rate" in name:
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
        
        if cat == "identity":
            cat_em_count = sum(1 for p_str, e in zip(cat_preds, cat_exps) if _normalize_for_em(p_str) == _normalize_for_em(e))
            cat_em = cat_em_count / len(cat_exps)
            cat_fpr = 1.0 - cat_em
            cat_rows.append({
                "Kategoria": "IDENTITY (Zdania czyste)",
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
                "Kategoria": cat.upper(),
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
        sample_rows.append({"Typ Próbki": "TRUE POSITIVE (Poprawnie naprawione)", "Wejście (Błąd)": s, "Złoty Standard (Expected)": e, "Wyjście Modelu": p_str})
    for s, e, p_str in false_negatives[:2]:
        sample_rows.append({"Typ Próbki": "FALSE NEGATIVE (Copying Bias / Pominięte)", "Wejście (Błąd)": s, "Złoty Standard (Expected)": e, "Wyjście Modelu": p_str})
    for s, e, p_str in false_positives[:2]:
        sample_rows.append({"Typ Próbki": "FALSE POSITIVE (Nadkorekcja / Zepsute)", "Wejście (Błąd)": s, "Złoty Standard (Expected)": e, "Wyjście Modelu": p_str})
        
    if sample_rows:
        samples_df = pd.DataFrame(sample_rows)
        display(HTML("<h4>Przykładowe Wyniki Predykcji</h4>"))
        display(samples_df.style.set_properties(**{'text-align': 'left'}).hide(axis='index'))
        
    return f05_score, p, r, em_score, avg_bertscore, tpr, fpr, fnr


def plot_categorized_metrics_comparison(final_results: Dict[str, Any], gold_srcs: List[str], gold_exps: List[str], reranked: bool = False) -> None:
    """Plots a grouped bar chart comparing TPR/FPR by error category across all models.
    
    Args:
        final_results: Merged experimental results dictionary.
        gold_srcs: Ground truth source sentences.
        gold_exps: Ground truth expected target sentences.
        reranked: If True, plots metrics for Herbert-reranked models, otherwise for standard models.
    """
    from src.models.evaluation import get_categorized_metrics
    
    # 1. Categories to evaluate
    categories = ["prep", "false_friend", "gender", "case", "typos", "other", "identity"]
    cat_display_names = ["Prep\n(TPR)", "False Friend\n(TPR)", "Gender\n(TPR)", "Case\n(TPR)", "Typos\n(TPR)", "Other\n(TPR)", "Identity\n(FPR)"]
    
    # 2. Extract metrics for each model
    model_data = {}
    
    # Add baseline
    baseline_cat = get_categorized_metrics(gold_srcs, gold_exps, gold_srcs)
    baseline_vals = []
    for cat in categories:
        if cat == "identity":
            baseline_vals.append(baseline_cat.get(cat, {}).get("fpr", 0.0))
        else:
            baseline_vals.append(baseline_cat.get(cat, {}).get("tpr", 0.0))
    model_data["Baseline (Do-Nothing)"] = baseline_vals
    
    for exp_name, metrics in final_results.items():
        out_dir = metrics.get("out_dir", "")
        if not os.path.exists(out_dir) and "results/small/" in out_dir:
            out_dir = out_dir.replace("results/small/", "results-small/")
            
        file_name = "human_preds_reranked.json" if reranked else "human_preds.json"
        preds_path = os.path.join(out_dir, file_name)
        
        if os.path.exists(preds_path):
            try:
                with open(preds_path, "r", encoding="utf-8") as f:
                    preds = json.load(f)
                cat_metrics = get_categorized_metrics(gold_srcs[:len(preds)], gold_exps[:len(preds)], preds)
                
                vals = []
                for cat in categories:
                    if cat == "identity":
                        vals.append(cat_metrics.get(cat, {}).get("fpr", 0.0))
                    else:
                        vals.append(cat_metrics.get(cat, {}).get("tpr", 0.0))
                
                # shorten name for plot legend
                short_name = shorten_name(exp_name)
                
                model_data[short_name] = vals
            except Exception:
                pass

    # 3. Plotting
    num_models = len(model_data)
    if num_models == 0:
        print("Brak danych do narysowania wykresu.")
        return
        
    fig, ax = plt.subplots(figsize=(16, 8))
    
    x = np.arange(len(categories))
    width = 0.85 / num_models # distribute bars evenly
    
    colors = plt.colormaps.get_cmap("tab10")(np.linspace(0, 1, num_models))
    
    for idx, (model_name, vals) in enumerate(model_data.items()):
        offset = (idx - num_models / 2) * width + width / 2
        ax.bar(x + offset, vals, width, label=model_name, color=colors[idx])
        
    ax.set_ylabel('Wskaźnik (TPR / FPR)', fontsize=12)
    title_suffix = " (Po Rerankingu HerBERT)" if reranked else " (Wersje Standardowe)"
    ax.set_title('Porównanie skuteczności modeli w podziale na kategorie błędów L1' + title_suffix, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(cat_display_names, fontsize=11)
    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax.set_ylim(0, 1.1)
    
    # format y axis as percentages
    import matplotlib.ticker as mtick
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    plt.tight_layout()
    plt.show()


