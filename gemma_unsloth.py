import os
from unsloth import FastModel
import torch
from datasets import load_dataset
from unsloth.chat_templates import standardize_data_formats
from unsloth.chat_templates import get_chat_template

dataset_medic = load_dataset("nisten/battlefield-medic-sharegpt", split = "train")
dataset = load_dataset("mlabonne/FineTome-100k", split = "train")

dataset = standardize_data_formats(dataset)
dataset_medic = standardize_data_formats(dataset_medic)

print(dataset_medic[100])
print(dataset[100])
"""
model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3-1b-it",
    max_seq_length = 2048, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    # token = "hf_...", # use one if using gated models
)

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True,  # SHould leave on always!

    r = 8,           # Larger = higher accuracy, but might overfit
    lora_alpha = 8,  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template = "gemma-3",
)

dataset_medic = load_dataset("nisten/battlefield-medic-sharegpt", split = "train")
dataset = load_dataset("mlabonne/FineTome-100k", split = "train")

dataset = standardize_data_formats(dataset)
dataset_medic = standardize_data_formats(dataset_medic)

print(dataset_medic[100])
print(dataset[100])

def apply_chat_template(examples):
    texts = tokenizer.apply_chat_template(examples["conversations"])
    return { "text" : texts }
pass
dataset = dataset.map(apply_chat_template, batched = True)
dataset_medic = dataset_medic.map(apply_chat_template, batched = True)

print(dataset_medic[100]["text"])
print(dataset[100]["text"])"""