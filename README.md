# PLT5 Grammatical Error Correction for L1 Interference

This project explores the use of generative Sequence-to-Sequence (Seq2Seq) models (specifically `allegro/plt5-small` and `allegro/plt5-base`) to identify and correct cross-lingual interference errors made by native Russian and Ukrainian speakers writing in Polish. 

The pipeline covers data acquisition (Wiktionary scraping), synthetic error injection via SpaCy and PoliMorf, LLM-driven rule generation, and extensive ablation studies on model sizing, dataset scaling, and fluency reranking.

## Project Summary & Insights

### 1. Goal & Research Problem
* **The Challenge:** Generative Seq2Seq models in Grammatical Error Correction (GEC) tasks suffer from two primary limitations:
  1. *Copying Bias:* The model mathematically prefers copying the erroneous input sequence rather than risking modification.
  2. *Over-Correction:* When trained exclusively on errors, the model starts modifying correct sentences, assuming all input text is faulty.
* **Research Goal:** Explore if a Polish Encoder-Decoder model (`PLT5`) can identify cross-lingual L1 syntax/lexical interference (preposition calques, case mismatch, false friends) and correct them using preference optimization (ORPO) and a fluency reranker (*Herbert* BERT Pseudo-Log-Likelihood scoring).

### 2. Architecture & Methodology
* **Models:** Ablated model capacity using `plt5-small` (60M parameters) on a 10k dataset, and scaled up to `plt5-base` (275M parameters) on 10k and 50k datasets.
* **LoRA Fine-Tuning:** Applied Low-Rank Adaptation to attention and MLP layers to drastically reduce VRAM footprints.
* **Custom ORPO Trainer:** Natively integrated Odds Ratio Preference Optimization with standard Hugging Face `Seq2SeqTrainer`.
  - *Math Fix:* Divided sequence log probabilities by the count of active tokens (sequence length normalization) in `get_batch_logps` to stabilize gradients and prevent gradient skew on longer sentences.
  - *Identity Translation Masking:* The odds-ratio copy penalty is zeroed-out (`is_error = 0`) for correct sentences, resolving the identity paradox and preserving grammatically correct inputs.

### 3. Experiments & Key Insights
1. **ORPO Instability on Small Models:** Running ORPO on `plt5-small` (Pipeline 1) zombified correct inputs (ruining **4.62% to 6.15%** of correct sentences, compared to **1.54%** for SFT). Small models lack the capacity to handle opposing objectives (SFT recovery vs. odds-ratio penalties) simultaneously.
2. **Dataset & Capacity Scaling (plt5-base & 50k):** Upgrading to `plt5-base` (Pipeline 2) doubled TPR to **7.69%**. Scaling the dataset to 50k pairs (Pipeline 3) dropped SFT training loss to **0.5686**, reaching **51.96%** synthetic EM and boosting human evaluation test set (OOD) TPR to **10.77%**.
3. **Herbert Fluency Reranking (Pipeline 4):** Applying *Herbert* BERT PLL scoring post-hoc over beam search candidates successfully broke copying bias, **doubling TPR to 23.46%** (compared to **10.77%** for Pipeline 3). However, because the reranker lacks semantic constraint checks, FPR rose to **24.62%** as it swapped correct sentences for highly fluent calques.
4. **Beam Search Size Trade-offs (Beam 3 vs 5):**
   - Beam size 3 (Pipeline 3 + Rerank) suppressed FPR to **16.92%** but missed complex morphosyntactic error corrections (Gender TPR = **0.00%**).
   - Beam size 5 (Pipeline 4) provided the reranker with correct inflection forms (Gender TPR = **28.57%**, Case TPR = **42.86%**) but increased candidate space noise, raising FPR to **24.62%**.
* **Conclusion:** SFT suffers from copying bias, and ORPO without length-normalization is unstable. Effective correction of L1 interference requires a model with at least `plt5-base` capacity, a wider beam size (>= 5), and a PLL change threshold ($\Delta PLL$) to protect correct sentences from semantic drift.

## Project Structure

* `data/` - Ignored by Git. Contains downloaded, scraped, generated, synthesized, and evaluation datasets.
* `results/` - Model checkpoints, generated predictions (un-reranked and reranked), and consolidated summaries grouped by model capacity (`results/small/` and `results/base/`).
* `src/config.py` - Centralized configurations and settings.
* `src/data/` - Data acquisition, dictionary extraction, and SpaCy corruption/synthesis logic.
* `src/models/` - Custom evaluation logic, classification resources, and the Herbert Pseudo-Log-Likelihood `FluencyReranker`.
* `src/training/` - Natively written custom ORPO loss and data collators inside a standard Hugging Face `Seq2SeqTrainer` harness.
* `src/utils/setup.py` - Environment bootstrapping entrypoint. Resolves Windows DLL crashes, overrides OpenMP conflicts, and applies Hugging Face security overrides.
* `src/main/prepare_data.py` - Synthesizes the 10k/50k training, validation, and test datasets.
* `src/main/run_experiments.py` - Unified runner. Natively trains models, generates beam search predictions, runs Herbert fluency reranking, and compiles summarized results.
* `main.ipynb` - Academic paper-structured Jupyter Notebook for analysis and chart visualization.

## Setup & Installation

1. **Create a virtual environment (Python 3.10+ recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   python -m spacy download pl_core_news_lg
   ```
   *(Note: Windows DLL clashes and Hugging Face package security blockers are handled automatically by our bootstrapping module).*
3. **Set up API Keys:**
   Create a `.env` file in the root directory and add your Groq/Gemini API key:
   ```env
   GROQ_API_KEY=your_api_key_here
   ```

## How to Run

1. **Synthesize the datasets:**
   ```bash
   python src/main/prepare_data.py
   ```
2. **Run the experiments runner:**
   ```bash
   python src/main/run_experiments.py
   ```
   Add `--train` to train models if checkpoints are missing, or `--force-infer` to force predictions generation and reranking.
3. **Explore and Render Results:**
   Open `main.ipynb` and run the cells sequentially to visualize final ablation curves, stylized metrics stylers, and error categories.
