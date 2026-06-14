import os
import torch
import torch.nn as nn
from typing import Dict, Any, Union, Optional, List
from transformers import Seq2SeqTrainer, DataCollatorForSeq2Seq


def get_batch_logps(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Computes length-normalized log probabilities of labels given logits."""
    labels_clone = labels.clone()
    loss_mask = labels_clone != -100
    labels_clone[~loss_mask] = 0
    
    # log_softmax over vocabulary dimension
    log_probs = logits.log_softmax(-1)
    
    # gather log probability for each label token
    per_token_logps = torch.gather(log_probs, dim=2, index=labels_clone.unsqueeze(2)).squeeze(2)
    
    # mask out padding/non-active tokens
    masked_logps = per_token_logps * loss_mask
    
    # length normalization: average over active sequence length
    lengths = loss_mask.sum(-1).clamp(min=1)
    return masked_logps.sum(-1) / lengths


class ORPODataCollator(DataCollatorForSeq2Seq):
    """Data collator for Odds Ratio Preference Optimization.
    
    Maps input fields for chosen/rejected sequences and is_error masks.
    """
    def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:
        # Extract chosen and rejected labels
        chosen_features = []
        rejected_features = []
        is_error_list = []
        
        for feature in features:
            is_error_list.append(feature.get("is_error", 1))
            
            # create chosen sequence feature dict
            chosen_features.append({
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
                "labels": feature["chosen_labels"]
            })
            
            # create rejected sequence feature dict
            rejected_features.append({
                "input_ids": feature["input_ids"],
                "attention_mask": feature["attention_mask"],
                "labels": feature["rejected_labels"]
            })
            
        # batch features using base collator
        batch = super().__call__(chosen_features, return_tensors=return_tensors)
        
        # rename labels to chosen_labels
        batch["chosen_labels"] = batch["labels"].clone()
        
        # batch rejected features
        rejected_batch = super().__call__(rejected_features, return_tensors=return_tensors)
        batch["rejected_labels"] = rejected_batch["labels"]
        
        # add is_error mask tensor
        batch["is_error"] = torch.tensor(is_error_list, dtype=torch.float32, device=batch["input_ids"].device)
        
        return batch


class CustomORPOTrainer(Seq2SeqTrainer):
    """Custom trainer for Seq2Seq Odds Ratio Preference Optimization."""
    
    def __init__(self, *args, orpo_beta: float = 0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.orpo_beta = orpo_beta

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # inputs contains: input_ids, attention_mask, chosen_labels, rejected_labels, is_error
        
        # 1. Run the encoder once to avoid redundant computations (50% speedup & VRAM saving)
        base_model = model.base_model.model if hasattr(model, "base_model") else model
        encoder = base_model.get_encoder()
        encoder_outputs = encoder(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True
        )
        
        # 2. Chosen sequence pass (decoder-only forward)
        chosen_outputs = model(
            encoder_outputs=encoder_outputs,
            attention_mask=inputs["attention_mask"],
            labels=inputs["chosen_labels"],
            return_dict=True
        )
        sft_loss = chosen_outputs.loss
        chosen_logits = chosen_outputs.logits
        
        # 3. Rejected sequence pass (decoder-only forward with gradients)
        rejected_outputs = model(
            encoder_outputs=encoder_outputs,
            attention_mask=inputs["attention_mask"],
            labels=inputs["rejected_labels"],
            return_dict=True
        )
        rejected_logits = rejected_outputs.logits
            
        # 4. Compute length-normalized log-probabilities
        chosen_logps = get_batch_logps(chosen_logits, inputs["chosen_labels"])
        rejected_logps = get_batch_logps(rejected_logits, inputs["rejected_labels"])
        
        # 5. Compute ORPO odds ratio loss with Identity Mask
        log_odds = chosen_logps - rejected_logps
        log_sigmoid_odds = torch.log(torch.sigmoid(log_odds) + 1e-8)
        
        # multiplier: apply preference loss only if is_error = 1
        is_error = inputs.get("is_error", torch.ones_like(log_odds))
        odds_loss = -self.orpo_beta * (is_error * log_sigmoid_odds).mean()
        
        total_loss = sft_loss + odds_loss
        
        return (total_loss, chosen_outputs) if return_outputs else total_loss
