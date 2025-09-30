# trainer.py
import requests
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
import yaml
import os
import csv
from datasets import Dataset, load_from_disk
import pandas as pd
from vllm import SamplingParams
import time
import torch
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

general_config = config['general']
model_name = general_config['model_name']
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.pad_token if tokenizer.pad_token is not None else tokenizer.eos_token
hf_token = general_config['hf_token']
wandb_key = general_config['wandb_key']
os.environ["WANDB_API_KEY"] = wandb_key

finetuning_config = config['finetuning']
num_init_samples = finetuning_config['num_init_samples']
num_sub_epochs = finetuning_config['num_sub_epochs']
batch_size = finetuning_config['batch_size']
learning_rate = finetuning_config['learning_rate']
num_generated_responses = finetuning_config['num_generated_responses']
num_total_iters = finetuning_config['num_total_iters']
temperature = finetuning_config['temperature']
max_tokens = finetuning_config['max_tokens']
grad_acc_steps = finetuning_config['grad_acc_steps']
save_steps = finetuning_config['save_steps']
save_model_every_n_iters = finetuning_config['save_model_every_n_iters']
eval_steps = finetuning_config['eval_steps']
no_dataset_generation = finetuning_config['no_dataset_generation']
dataset_array = finetuning_config['dataset_array']
dataset_sizes = finetuning_config['dataset_sizes']
persistent_prompt = finetuning_config['persistent_prompt']
num_answers = finetuning_config['num_answers']

filepath = '/workspace/'
dataset_filepath = filepath + 'Datasets/'

# directly modify values here, for testing purposes
dataset_array = [["hdr"], ["wfragen"], ["wfragen", "AntonsFragen"], ["wfragen", "AntonsFragen"], ["hack_it", "wfragen"],
                 ["hack_it", "wfragen"], ["hack_it"], ["deutschland"], ["integration"], ["krieg"], ["krieg"]]
#"AntonsFragen", "deutschland", "fucking_racist", "hack_it", "hdr", "hp", "integration", "krieg", "männer", "politik", "w_fragen"


def load_bnb_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
    )
    print("Model loaded successfully!")

    model = prepare_model_for_kbit_training(model)
    return model


### This function will return an array of questions.
# Generally, this function should not be required in the finetuning process, since we usually should have a version that has questions and answers.
# However, for fallback and testing purposes, we have it.
# It expects csv-files that consist of one row of questions.
# In a previous version there was some dataset_distinct_questions handling, will have to see if that was necessary/relevant
def load_questions(iter, filepath=dataset_filepath):
    questions = []
    datasets = dataset_array[iter]
    for dataset in datasets:
        file = filepath+dataset+'.csv'
        with open(file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
        for row in reader:
            for cell in row:
                questions.append(cell)

    print("=====Debug finetuning.ipynb load_questions() =====")
    print("=====Questions: ", questions)
    print("=====Datasets: ", datasets)
    print("=====End Debug finetuning.ipynb load_questions() =====")
    return questions


def load_tokenized_questions(iter, filepath=dataset_filepath):
    questions = []
    datasets = dataset_array[iter]
    for dataset in datasets:
        tokenized_dataset = load_from_disk(filepath+dataset+"_tokenized/")
        questions.append(tokenized_dataset)
    return questions


### This function will prompt the current version of the model for an answer.
# Inputs are prompts, an array of strings that will be fed to the model.
# num_answers is a parameter that determines how many answers the model should generate.
# chatbot can be either a model-pipeline, or None. If it is None, a new model-pipeline will be created, else the existing pipeline is used.
# model can either be a model or None. Either model or chatbot have to be not None. If chatbot is None, the model is used to create a new one. TODO: maybe not pass the entire model, but create the pipeline before calling this method. Then we also do not need this parameter and should not return the chatbot
def generate_answers(prompt_tokens, num_answers_local):
    parameters = SamplingParams(
        max_new_tokens=max_tokens,
        temperature=temperature,
        num_return_sequences=num_answers_local,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.unk_token_id
    )
    output_tokens = requests.get(
        "http://localhost:8000/generate_training",
        params={"prompt": prompt_tokens, "parameters": parameters
                }
    )
    generated_text = output_tokens.json()["text"]
    return generated_text


### This function will print the answer of the current model to the persistent prompt.
# This mainly serves testing/debugging purposes.

def generate_and_print_answer():
    prompt = persistent_prompt
    answer = generate_answers(prompt, 1)
    print(answer)


### One iteration of the training loop.
# We train for num_sub_epoch epochs, and generally take the config from the config file.
# The Lora-Config is default and for now not subject to change.
# The finetuned model is returned.

def training_iteration(model, tokenizer, tokenized_dataset, modelpath):
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=modelpath,     # Directory to save the model
        eval_strategy="steps",         # Evaluate the model periodically
        save_strategy="steps",               # Save model checkpoints periodically
        logging_dir=modelpath+"/logs",                # Directory for logs
        learning_rate=learning_rate,                  # Learning rate for fine-tuning
        per_device_train_batch_size=batch_size,       # Batch size per device
        gradient_accumulation_steps=grad_acc_steps,       # Accumulate gradients to simulate larger batches
        gradient_checkpointing=True,         # Use gradient checkpointing for memory efficiency
        optim="adamw_bnb_8bit",
        num_train_epochs=num_sub_epochs,         # Number of training epochs
        logging_steps=10,                    # Log progress every 100 steps
        save_steps=save_steps,                      # Save a checkpoint every x steps
        eval_steps=eval_steps,
        bf16=True,                           # Enable mixed precision for faster training
        push_to_hub=False,                    # Disable auto-push to Hugging Face Hub
        dataloader_num_workers=4,
        torch_compile=True,
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
        lora_dropout=0.00,
        bias="none",
        task_type="CAUSAL_LM"
    )
    if not hasattr(model, 'peft_config'):  # Avoid applying PEFT multiple times if the cell is run again
        print("Applying PEFT...")
        model = get_peft_model(model, lora_config)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],  # Training dataset
        eval_dataset=tokenized_dataset["test"],    # Evaluation dataset
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    model = load_bnb_model()
    for i in range(num_total_iters):
        start_time = time.time()
        print("===== finetuning.ipynb starts iteration ", i)
        modelpath = filepath + "mistral_finetuned/epochs-" + str(num_sub_epochs) + "_lr-" + str(
            learning_rate) + "_iter-" + str(iter)
        tokenized_dataset = load_tokenized_questions(i, )
        training_iteration(model, tokenizer, tokenized_dataset, modelpath)
        # Notify the GPU owner service to reload
        resp = requests.post("http://localhost:8000/reload", json={"path": modelpath})
        print(resp.json())
        print("===== finetuning.ipynb ended iteration ", i)
        print("This iteration had a duration of ", time.time() - start_time)
        model.save_pretrained(modelpath)