# trainer.py
import json
import shutil

import pandas as pd
import requests
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
import yaml
import os
import csv
from datasets import Dataset, load_from_disk, concatenate_datasets, load_dataset
import numpy as np
from vllm import SamplingParams
import time
import torch
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

if torch.cuda.device_count() > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print("using gpu 1")

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

general_config = config['general']
model_name = general_config['model_name']
tokenizer = AutoTokenizer.from_pretrained("/workspace/models/mistral-7b-v0.3")
tokenizer.pad_token = tokenizer.pad_token if tokenizer.pad_token is not None else tokenizer.eos_token
hf_token = general_config['hf_token']
wandb_key = general_config['wandb_key']
os.environ["WANDB_API_KEY"] = wandb_key
inference_port = general_config["inference_port"]
owner_port = general_config["owner_port"]
API_URL_INFERENCE = "http://localhost:"+str(inference_port)+"/"
API_URL_OWNER = "http://localhost:"+str(owner_port)+"/"
os.environ["WANDB_API_KEY"] = wandb_key
available_datasets = general_config["available_datasets"]

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
save_steps = 200 / batch_size
eval_steps = save_steps * 2
no_dataset_generation = finetuning_config['no_dataset_generation']
dataset_array = finetuning_config['dataset_array']
samples_by_iteration = finetuning_config['samples_by_iteration']
live_samples_by_iteration = finetuning_config['live_samples_by_iteration']
max_per_prompt = finetuning_config['max_per_prompt']
persistent_prompt = finetuning_config['persistent_prompt']
num_answers = finetuning_config['num_answers']

filepath = '/workspace/'
dataset_filepath = filepath + 'Datasets/'
modelpath = filepath + 'models/'
initial_modelpath = "/workspace/models/mistral-7b-v0.3"

# directly modify values here, for testing purposes
#dataset_array = [["hdr"], ["wfragen"], ["wfragen", "AntonsFragen"], ["wfragen", "AntonsFragen"], ["hack_it", "wfragen"],
#                 ["hack_it", "wfragen"], ["hack_it"], ["deutschland"], ["integration"], ["krieg"], ["krieg"]]
#"AntonsFragen", "deutschland", "fucking_racist", "hack_it", "hdr", "hp", "integration", "krieg", "männer", "politik", "w_fragen"


def load_bnb_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        initial_modelpath,
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

### Returns an array of datasets
def load_tokenized_questions(iter, dataset_filepath=dataset_filepath):
    all_datasets = []
    datasets = dataset_array[iter]
    for dataset in datasets:
        load_path = ""
        iteration_path = dataset_filepath+dataset+"/iteration/"
        if len(os.listdir(iteration_path)) != 0:
            available = os.listdir(iteration_path)
            highest = max([int(a) for a in available])
            print(f"for dataset {dataset}, we currently found {highest} as the latest iteration.")
            load_path = iteration_path + str(highest)
        else:
            print(f"found no iterations for dataset {dataset}")
            load_path = dataset_filepath + dataset + "/" + dataset + "_tokenized/"
        tokenized_dataset = load_from_disk(load_path)
        all_datasets.append(tokenized_dataset)

    combined_dataset = concatenate_datasets(all_datasets)
    return combined_dataset


def tokenize_dataset(questions):
    prompts = [f"<s>[INST] {q} [/INST]{r}{tokenizer.eos_token}" for q,r in zip(questions["prompt"], questions["response"])]
    tokenized = tokenizer(
        prompts,
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    # Add labels for language modeling
    tokenized["labels"] = tokenized["input_ids"].copy()
    tokenized["prompt"] = questions["prompt"]
    tokenized["response"] = questions["response"]
    return tokenized


def tokenize_live_questions(highest_iteration):
    live_questions = []
    if highest_iteration < 0:
        with open(dataset_filepath + "live/answered_conversations.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                live_questions.append({"prompt": entry["prompt"], "response": entry["response"], "iteration": entry["iteration"]})
        dataset = Dataset.from_list(live_questions)
        tokenized_dataset = dataset.map(tokenize_dataset, batched=True, remove_columns=["response"])
        print("tokenize_live_questions in iter 0 created ", tokenized_dataset)
    else:
        print("trainer tokenize live questions loads from ", dataset_filepath + "live/iteration/" + str(highest_iteration))
        dataset = Dataset.load_from_disk(dataset_filepath + "live/iteration/" + str(highest_iteration))
        tokenized_dataset = dataset.map(tokenize_dataset, batched=True)
        print(f"tokenize_live_questions in iter {highest_iteration} created ", tokenized_dataset)

    return tokenized_dataset, len(live_questions)



### This function will prompt the current version of the model for an answer.

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
        API_URL_OWNER + "generate_training",
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
    print("in training_iteration, dataset has columns: ", tokenized_dataset)
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
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
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
    resp_owner = requests.post(API_URL_OWNER+"reload", json={"path": modelpath}) # send to owner
    resp_inference = requests.post(API_URL_INFERENCE + "reload", json={"path": modelpath}) # send to inference
    print("Owner: ", resp_owner.json())
    print("Inference: ", resp_inference.json())
    return model



def choose_datasets(live_dataset, pre_dataset, iteration):
    start_time = time.time()
    final_list = []
    df = pd.DataFrame(live_dataset)
    df_pre = pd.DataFrame(pre_dataset)
    if "question_id" not in df.columns:
        print(df["prompt"].apply(type).value_counts())
        print("df in choose dataset: ", df)
        print("df columns: ", df.columns)
        df["question_id"] = df.groupby("prompt").ngroup()
    df = df.sort_values("iteration", ascending=False)
    df["weight"] = np.exp(df["iteration"]) / np.exp(df["iteration"]).sum()
    number_of_samples = live_samples_by_iteration
    for qid, group in df.groupby("question_id"):
        # shuffle within question group, with weights if desired
        group = group.sample(
            n=min(len(group), max_per_prompt),
            replace=False,
            weights=group["weight"]
        )
        final_list.append(group)
    final_list = pd.concat(final_list)
    save_final_list = final_list
    if len(final_list) > number_of_samples: # at first, we only take live_samples_by_iteration samples and fill the rest from the pre_dataset
        final_list = final_list.sample(
            n=number_of_samples,
            weights=final_list["weight"],
            replace=False
        )
    missing = samples_by_iteration - len(final_list) # determine how many samples are missing
    if missing > 0:
        print(f"Only {len(final_list)} live samples, adding {missing} from pre_dataset")
        pre_add = df_pre.sample(
            n=min(missing, len(df_pre)),
            replace=False
        )
        print(f"found {len(pre_add)} samples to add")
        final_list = pd.concat([final_list, pre_add])

    still_missing = samples_by_iteration - len(final_list)
    if still_missing > 0:
        print(f"We still only have {len(final_list)} samples, adding {still_missing} samples from live_dataset")
        pre_add = save_final_list.sample(
            n=min(still_missing, len(save_final_list)),
            replace=False
        )
        print(f"found {len(pre_add)} samples to add")
        final_list = pd.concat([final_list, pre_add])
    print("final list in choose dataset ", final_list)
    print("available columns: ", final_list.columns)
    final_dataset = Dataset.from_pandas(final_list.reset_index(drop=True))
    final_dataset.remove_columns(["question_id", "weight"])
    result = final_dataset.train_test_split(test_size=0.1, seed=42)
    print(f"choice took {time.time()-start_time} seconds")
    return result


def delete_subfolders(path, keep_folders= None, debug=False):
    # deletes every folder at the specified address. e.g., when given workspace/Datasets/AntonsFragen/iteration/ it will delete every folder in there, but not the iteration folder
    for folder in os.listdir(path):
        if folder == keep_folders:
            print(f"not deleting folder {folder} ")
        else:
            if debug:
                print("deleting ", path + folder)
            shutil.rmtree(path + folder)


if __name__ == "__main__":
    model = load_bnb_model()
    #requests.post(API_URL_INFERENCE+"infer_next_iters", json={"iteration": 0, "data": available_datasets})  (this answers the datasets with the base model)
    for available_dataset in available_datasets:
        if os.path.exists(dataset_filepath + available_dataset + "/iteration/"):
            delete_subfolders(dataset_filepath + available_dataset + "/iteration/", "0")
    highest = -1
    for i in range(num_total_iters):
        requests.post(API_URL_INFERENCE+"generate_persistent", json={"prompt": persistent_prompt, "iteration": i})
        if i > 0:
            if os.path.exists(dataset_filepath + "live/" + "iteration"):
                available = os.listdir(dataset_filepath + "live/" + "iteration")
                highest = max([int(a) for a in available])
                print(f"latest iteration for the live dataset in main is {highest} ")
        start_time = time.time()
        print("===== trainer.py starts iteration ", i)
        finetuned_modelpath = modelpath + "mistral_finetuned/epochs-" + str(num_sub_epochs) + "_lr-" + str(
            learning_rate) + "/iter-" + str(i)
        tokenized_live_dataset, length = tokenize_live_questions(highest) # returns a dataset of all live questions, and the number of responses that we have
        tokenized_dataset = load_tokenized_questions(i, )
        iteration_dataset = choose_datasets(tokenized_live_dataset, tokenized_dataset, i)
        model = training_iteration(model, tokenizer, iteration_dataset, finetuned_modelpath)
        requests.post(API_URL_INFERENCE+"infer_next_iters", json={"iteration": i, "data": dataset_array[i+1:i+3]})
        print("===== finetuning.ipynb ended iteration ", i)
        print("This iteration had a duration of ", time.time() - start_time)
        model.save_pretrained(finetuned_modelpath)
