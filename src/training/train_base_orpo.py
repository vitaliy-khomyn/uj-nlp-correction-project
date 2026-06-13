import os
import sys

# Windows PyTorch & PyArrow DLL conflict fix: call bootstrap_environment before importing other packages
sys.path.append(os.path.abspath("."))
from src.utils.setup import bootstrap_environment
bootstrap_environment()

import logging
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM, 
    Seq2SeqTrainingArguments
)
from peft import get_peft_model, LoraConfig, TaskType

from src.config import config
from src.data.dataset import prepare_and_load_dataset
from src.training.trainer import CustomORPOTrainer, ORPODataCollator

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def train_base_orpo() -> None:
    # 1. Load 10k dataset
    logging.info("Loading 10k GEC dataset...")
    train_df, val_df, test_df, _ = prepare_and_load_dataset("10k")
    
    # 2. Load base model and tokenizer
    model_name = "allegro/plt5-base"
    logging.info(f"Loading tokenizer and model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Check if GPU is bfloat16 compatible
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if bf16_supported else torch.float32
    ).to(device)
    
    # 3. Configure LoRA adapter
    logging.info("Configuring LoRA wrapper...")
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=0.1,
        target_modules=config.lora_target_modules
    )
    model = get_peft_model(base_model, peft_config)
    model.print_trainable_parameters()
    
    # 4. Tokenization mapper for ORPO preference data pairs
    max_length = config.max_seq_length
    
    def tokenize_orpo_fn(examples):
        prefixed_srcs = [config.task_prefix + s for s in examples["source"]]
        model_inputs = tokenizer(
            prefixed_srcs,
            max_length=max_length,
            truncation=True,
            padding=False
        )
        
        # Tokenize chosen/gold target
        chosen_tokens = tokenizer(
            text_target=examples["target"],
            max_length=max_length,
            truncation=True,
            padding=False
        )
        
        # Tokenize rejected/corrupted source
        rejected_tokens = tokenizer(
            text_target=examples["source"],
            max_length=max_length,
            truncation=True,
            padding=False
        )
        
        model_inputs["chosen_labels"] = chosen_tokens["input_ids"]
        model_inputs["rejected_labels"] = rejected_tokens["input_ids"]
        model_inputs["is_error"] = examples["is_error"]
        
        return model_inputs

    logging.info("Tokenizing datasets...")
    train_ds = Dataset.from_pandas(train_df).map(
        tokenize_orpo_fn,
        batched=True,
        remove_columns=list(train_df.columns)
    )
    val_ds = Dataset.from_pandas(val_df).map(
        tokenize_orpo_fn,
        batched=True,
        remove_columns=list(val_df.columns)
    )
    
    # 5. Define Training Arguments
    out_dir = "./results/base/10k/var_b_orpo_10"
    os.makedirs(out_dir, exist_ok=True)
    
    # We use batch size 4 and accumulation 8 (effective batch size 32) to prevent out-of-memory on 8GB/16GB GPUs
    training_args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_epochs,
        logging_steps=config.logging_steps,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        bf16=bf16_supported,
        fp16=not bf16_supported and torch.cuda.is_available(),
        optim="adamw_torch",
        remove_unused_columns=False, # important: prevents HF from stripping custom columns
        report_to="none"
    )
    
    # 6. Instantiate ORPO Trainer
    logging.info("Initializing CustomORPOTrainer...")
    collator = ORPODataCollator(tokenizer=tokenizer, model=model)
    trainer = CustomORPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        orpo_beta=config.orpo_beta
    )
    
    # 7. Run ORPO Training
    logging.info("Starting ORPO training loop for base model (Pipeline 5)...")
    trainer.train()
    
    # Save the best model
    best_dir = os.path.join(out_dir, "best_checkpoint")
    model.save_pretrained(best_dir)
    tokenizer.save_pretrained(best_dir)
    logging.info(f"Training completed successfully! Saved best model to {best_dir}")


if __name__ == "__main__":
    train_base_orpo()
