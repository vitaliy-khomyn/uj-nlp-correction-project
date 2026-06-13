# TRL vs. Encoder-Decoder: A Debugging Chronicle

This document serves as a historical log of the debugging session where we attempted to integrate the `trl` (Transformer Reinforcement Learning) library into a T5 (Encoder-Decoder) pipeline for DPO (Direct Preference Optimization), and why we ultimately abandoned it in favor of a custom ORPO implementation.

## 1. The Initial Symptoms: `max_length` and ValueErrors
The initial issue presented as a simple `TypeError` regarding `max_length` missing from `DPOTrainer.__init__`. 
* **The Assumption:** Hugging Face simply moved arguments from the Trainer to `DPOConfig`.
* **The Fix:** Moved `max_length` and `precompute_ref_log_probs` to the config.
* **The Result:** Immediate crash: `ValueError: You have to specify either decoder_input_ids or decoder_inputs_embeds`.

## 2. The PEFT Masking & Causal LM Trap
We then encountered an endless stream of warnings: `Mismatch between tokenized prompt and the start of tokenized prompt+rejected`.
* **What was happening:** When `peft` wraps a model, it creates a `PeftModel` which completely hides the base model's `config`. Because `DPOTrainer` checks `model.config.is_encoder_decoder`, it defaulted to `False`. 
* **The consequence:** `trl` assumed T5 was a Causal LM (like GPT-4). It concatenated the prompt and target together into a single string to tokenize them, causing length mismatches. More fatally, for Causal LMs, it strictly passes `input_ids` and deletes the `labels` column from the batch.
* **The T5 crash:** T5 physically cannot execute a forward pass without `labels` or `decoder_input_ids`. Because `trl` stripped them, T5 crashed.
* **Our Attempted Fix:** We explicitly mapped `model.config = model.base_model.config` to force the flag to `True`.
* **Why it failed:** `trl` ignored the patch deeper down in its precomputation phase.

## 3. The `precompute_ref_log_probs` Disaster
To bypass tokenization issues, we attempted to use the precomputed log probability cache to speed up training.
* **What we discovered via Diagnostics:** We wrote a custom debug block to do dummy forward passes. It revealed that `trl`'s internal caching dataloader was fundamentally bugged for Seq2Seq architectures—it explicitly strips the `labels` column during the dry run regardless of architecture flags.

## 4. The Core Realization: TRL 1.5.1 Gutted Seq2Seq
After trying to manually pre-tokenize the data (forcing `prompt_input_ids`, `chosen_labels`, etc., into the batch) to bypass `trl`'s internal mappers, the trainer finally launched—only to crash inside PyTorch's `compute_loss`.
* **The Traceback:** The traceback revealed lines in `trl` source code relying on a `completion_mask`.
* **The Epiphany:** `completion_mask` is exclusively a Causal LM mathematical mechanism. We realized that in `trl` version 1.5.1, the developers had fundamentally gutted native Seq2Seq support from the core `DPOTrainer` loss calculations. No amount of hacking configs or datasets could bypass hardcoded Causal LM math inside the library.

## 5. Dependency Hell & API Drift
Knowing 1.5.1 was broken, we downgraded to `trl==0.8.6`, the last known stable version for Encoder-Decoder architectures. This triggered a cascade of Hugging Face version mismatch errors:
1. **`tokenizer` vs `processing_class`:** `transformers` removed the `tokenizer` kwarg, but `trl 0.8.6` hardcoded it. We monkey-patched `transformers.Trainer.__init__` to silently rename it.
2. **`get_batch_samples` Arity:** Newer `transformers` passes a `device` parameter (4 arguments), but `trl 0.8.6` only accepted 3. We wrote monkey-patches to absorb the extra arguments.
3. **`compute_loss` Arity:** Added `num_items_in_batch`, requiring another monkey-patch.
* **The Breaking Point:** Even with all monkey-patches in place, the evaluation loop crashed with `AttributeError: 'generator' object has no attribute 'generate'`. The internal generation pipelines between old `trl` and new `transformers` were completely incompatible.

## 6. The Final Pivot: Natively Written ORPO
Instead of fighting an unmaintained and drifting black-box library, we abandoned `trl` entirely.
* **The Solution:** We implemented Odds Ratio Preference Optimization (ORPO) natively. 
* **Why it works:**
    1. **No Reference Model:** ORPO aligns the model without needing a frozen reference model, saving massive VRAM and computational time.
    2. **Inherits from Seq2SeqTrainer:** By subclassing Hugging Face's standard `Seq2SeqTrainer`, we got perfectly stable evaluation, generation, and API compliance out-of-the-box.
    3. **Custom Collator:** We wrote `ORPODataCollator` to explicitly maintain `input_ids`, `attention_mask`, `chosen_labels`, and `rejected_labels`, while cleanly mapping `labels` for standard SFT generation metrics.
    4. **Sequence Length Normalization:** We fixed the gradient bias bug by implementing explicit sequence length normalization in custom batch log probability calculation (`get_batch_logps`), dividing sequence logprobs by active token mask lengths.

**Conclusion:** 
We spent hours trying to force a Causal LM-centric library to accept an Encoder-Decoder model. By writing the mathematical loss function (ORPO) natively, we bypassed thousands of lines of buggy, incompatible library code with ~40 lines of pure, highly stable PyTorch.

## 7. Developer Guidelines
1. **No Direct Notebook Edits:** Do not modify `main.ipynb` directly through editor tools as it can corrupt the JSON format or cause git/runtime conflicts. Instead, provide clean markdown diffs or code blocks in the chat so that the user can copy/paste and update the notebook cells themselves.
2. **Preserving Experiment Pipelines:** Support reproducibility by preserving all experiment parameters, subdirectories, and logs for both `plt5-small` and `plt5-base` runs.
3. **Reproducibility Unification:** All pipelines (training, base predictions, fluency reranking, and consolidated metrics merging) must be executed natively via `src/main/run_experiments.py`. Do not create or run post-hoc patching or summary reconstruction scripts.
4. **Environment Bootstrapping:** Always use `bootstrap_environment()` from `src.utils.setup` to load packages safely and avoid Windows import segfaults.


