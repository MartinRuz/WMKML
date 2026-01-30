# trainer.py
from utility import *

if available_gpus > 2:
    print("using gpu 2")
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
elif available_gpus > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print("using gpu 1")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("using gpu 0")

import json
import torch
import pandas as pd
from datasets import Dataset, load_from_disk, concatenate_datasets
import numpy as np
from vllm import SamplingParams
import time
from transformers import TrainingArguments, Trainer, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForLanguageModeling
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
import multiprocessing as mp
import logging
import datetime
from pythonosc import udp_client

print("visible devices:", torch.cuda.device_count())
print("current device:", torch.cuda.current_device())
print("gpu name:", torch.cuda.get_device_name())


logging.basicConfig(level=logging_level, format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s')
logger = logging.getLogger("trainer")
client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)


"""
This function loads a quantized version of the model, that will be used for training.
"""
def load_bnb_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        initial_modelpath,
        device_map={"":0},
        quantization_config=bnb_config,
    )
    logger.info("Model loaded successfully!")

    model = prepare_model_for_kbit_training(model)
    return model



"""
Returns a tokenized dataset
:param questions: a non-tokenized and non-templated dataset
:param tokenizer: the corresponding tokenizer
:return: a tokenized and templated dataset that has labels, response, (optionally) iteration, input_ids, labels, attention_mask columns
"""
def tokenize_dataset(questions):
    global tokenizer
    texts = []

    for q,r in zip(questions["prompt"], questions["response"]):
        messages = [
            {"role": "user", "content": q},
            {"role": "assistant", "content": r}
        ]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(formatted)

    logger.debug(f"example of a templated text in tokenize_dataset: {texts[0]}")
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=512,
    )
    # Add labels for language modeling
    tokenized["labels"] = tokenized["input_ids"].copy()
    tokenized["prompt"] = questions["prompt"]
    tokenized["response"] = questions["response"]
    if "iteration" in questions:
        tokenized["iteration"] = questions["iteration"]
    return tokenized



"""
This functions loads the most recent version of the datasets that are specified in the global variable dataset_array for the current iteration, and returns its tokenized version.
iter is the current iteration
dataset_filepath is the path of the dataset-directory.
"""
def load_questions_and_tokenize(iter, dataset_filepath=dataset_filepath):
    all_datasets = []
    datasets = dataset_array[iter] # which datasets do we load?
    for dataset in datasets:
        iteration_path = dataset_filepath + dataset + "/" + str(max_tokens) + "_tokens/" + "/iteration/"
        if len(os.listdir(iteration_path)) != 0:
            available = os.listdir(iteration_path)
            highest = max([int(a) for a in available]) # find the latest version (aka the most corrupt answers)
            logger.debug(f"for dataset {dataset}, we currently found {highest} as the latest iteration.")
            load_path = iteration_path + str(highest)
            if verification_mode: # optionally verify the integrity of the dataset
                ok = validate_answered_dataset(load_path)
                if not ok:
                    print("Dataset invalid!")
        else:
            logger.warning(f"found no iterations for dataset {dataset}")
            load_path = dataset_filepath + dataset + "/" + dataset + "_plain/"
            if verification_mode:
                if not validate_plain_dataset_structure(load_path):
                    logger.error("Structural validation failed for dataset %s", load_path)
                    continue
        untokenized_dataset = load_from_disk(load_path)
        tokenized_dataset = untokenized_dataset.map(tokenize_dataset, batched=True) # tokenize the dataset
        all_datasets.append(tokenized_dataset)

    combined_dataset = concatenate_datasets(all_datasets)
    return combined_dataset


"""
This function tokenizes the live-questions.
highest_iteration is the current iteration.
"""
def tokenize_live_questions(highest_iteration):
    live_dir = os.path.join(dataset_filepath, "live", f"{max_tokens}_tokens")
    live_file = os.path.join(live_dir, "conversations.jsonl")
    # Ensure directory exists
    os.makedirs(live_dir, exist_ok=True)
    # Ensure file exists
    if not os.path.exists(live_file):
        open(live_file, 'w').close()

    live_questions = []
    if highest_iteration < 0: # if we are in the first iteration, we have to create the dataset from scratch.
        with open(live_file, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                live_questions.append({"prompt": entry["prompt"], "response": entry["response"], "iteration": entry["iteration"]})
        dataset = Dataset.from_list(live_questions)
        tokenized_dataset = dataset.map(tokenize_dataset, batched=True, remove_columns=["response"]) # tokenize the dataset
    else: # for later iterations, we can simply load the existing dataset
        live_dir = os.path.join(live_dir,"iteration", f"{highest_iteration}")
        logger.debug(f"trainer tokenize live questions loads from {live_dir}")
        dataset = Dataset.load_from_disk(live_dir)
        tokenized_dataset = dataset.map(tokenize_dataset, batched=True) # tokenize the dataset

    return tokenized_dataset, len(live_questions)



"""
This function performs one iteration of the training loop.
We train for num_train_epoch epochs, and generally take the config from the config file.
The Lora-Config is default and for now not subject to change.
The finetuned model is saved to disk and returned.
model is the current version of the model that we finetune.
tokenizer is the corresponding tokenizer
tokenized_dataset is the tokenized dataset that is used for training in this iteration.
modelpath is the path where we save the finetuned iteration.
iteration is the current iteration.
"""
def training_iteration(model, tokenizer, tokenized_dataset, modelpath, iteration):
    training_start = time.time()
    if torch.cuda.is_bf16_supported():
        supports_bf16 = True
    else:
        supports_bf16 = False
    training_args = TrainingArguments(
        output_dir=modelpath,       # Directory to save the model
        eval_strategy="no",         # whether to evaluate the model
        save_strategy="no",               # whether to save model checkpoints
        logging_dir=modelpath+"/logs",                # Directory for logs
        learning_rate=learning_rate,                  # Learning rate for fine-tuning
        per_device_train_batch_size=batch_size,       # Batch size per device
        gradient_accumulation_steps=grad_acc_steps,       # Accumulate gradients to simulate larger batches
        gradient_checkpointing=False,         # whether to use gradient checkpointing for memory efficiency
        optim="adamw_bnb_8bit",
        num_train_epochs=num_train_epochs,         # Number of training epochs
        logging_steps=100,                    # Log progress every 100 steps
        bf16=supports_bf16,                          # Enable mixed precision for faster training
        push_to_hub=False,                    # Disable auto-push to Hugging Face Hub
        dataloader_num_workers=dataloader_num_workers, # how many cpu-workers do we use
        dataloader_pin_memory=True,
        torch_compile=True,
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )
    lora_config = LoraConfig( # the config for lora
        r=16,
        lora_alpha=32,
        target_modules=lora_target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    if not hasattr(model, 'peft_config'):  # Avoid applying PEFT multiple times if the cell is run again
        logger.debug("Applying PEFT...")
        model = get_peft_model(model, lora_config)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,  # Training dataset
        data_collator=data_collator,
    )
    trainer.train()
    model.save_pretrained(modelpath)
    if time.time() - training_start < minimum_iteration_time: # if the iteration was too fast, we sleep for some time
        logger.info(f"trainer was too fast, sleeping for {minimum_iteration_time-(time.time()-training_start)} seconds")
        time.sleep(minimum_iteration_time-(time.time()-training_start))
    _ = fire_and_forget(API_URL_OWNER+"reload", json={"path": modelpath, "iteration": iteration}) # send to owner
    _ = fire_and_forget(API_URL_INFERENCE + "reload", json={"path": modelpath, "iteration": iteration}) # send to inference
    return model


"""
This function randomly chooses samples to use for the current iteration, with some restrictions.
live_dataset is the dataset of live-questions
pre_dataset is the dataset of predefined questions, that we want to use in this iteration
iteration is the current iteration.
"""
def choose_datasets(live_dataset, pre_dataset, iteration):
    global samples_by_iteration
    start_time = time.time()
    final_list = []
    df = pd.DataFrame(live_dataset)
    df_pre = pd.DataFrame(pre_dataset)
    if "question_id" not in df.columns:
        df["question_id"] = df.groupby("prompt").ngroup()
    df = df.sort_values("iteration", ascending=False)
    df["weight"] = np.exp(df["iteration"]) / np.exp(df["iteration"]).sum()
    number_of_samples = live_samples_by_iteration # first, we take a number of samples from the live dataset
    for qid, group in df.groupby("question_id"):
        # shuffle within question group
        group = group.sample(
            n=min(len(group), max_per_prompt),
            replace=False,
            weights=group["weight"]
        )
        final_list.append(group)
    final_list = pd.concat(final_list)
    save_final_list = final_list
    if len(final_list) > number_of_samples:
        final_list = final_list.sample(
            n=number_of_samples,
            weights=final_list["weight"],
            replace=False
        )
    missing = samples_by_iteration - len(final_list) # determine how many samples are missing
    if missing > 0: # if necessary, we take some samples from the predefined dataset
        logger.debug(f"Only {len(final_list)} live samples, adding {missing} from pre_dataset")
        pre_add = df_pre.sample(
            n=min(missing, len(df_pre)),
            replace=False
        )
        logger.debug(f"found {len(pre_add)} samples to add")
        final_list = pd.concat([final_list, pre_add])

    still_missing = samples_by_iteration - len(final_list)
    if still_missing > 0: # if we still have too few samples, we take some more live-questions (if existing)
        logger.debug(f"We still only have {len(final_list)} samples, adding {still_missing} samples from live_dataset")
        pre_add = save_final_list.sample(
            n=min(still_missing, len(save_final_list)),
            replace=False
        )
        logger.debug(f"found {len(pre_add)} samples to add")
        final_list = pd.concat([final_list, pre_add])
    final_dataset = Dataset.from_pandas(final_list.reset_index(drop=True))
    final_dataset.remove_columns(["question_id", "weight"])

    if verification_mode:
        ok = validate_tokenized_dataset(
            final_dataset,
            tokenizer=tokenizer
        )

        if not ok:
            logger.error("INVALID TOKENIZED DATASET")

    result = final_dataset
    logger.info(f"choice took {time.time()-start_time} seconds")
    samples_by_iteration += sample_increase_by_iteration
    return result


"""
This function controls the general flow of execution, starting every run and performing some cleanup in between.
"""
def supervisor():
    global learning_rate, samples_by_iteration
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.pad_token if tokenizer.pad_token is not None else tokenizer.eos_token
    base_learning_rate = learning_rate # save the initial learning rate so we can reset it
    base_samples_by_iteration = samples_by_iteration
    if os.path.exists(restart_flag): # if there are any remnant flags, remove them
        os.remove(restart_flag)
    while True: # start one run after another.
        if delete_live_questions_at_start: # usually, we want to start every run from scratch, and delete previous live-questions
            live_dir = os.path.join(dataset_filepath, "live", f"{max_tokens}_tokens")
            live_file = os.path.join(live_dir, "conversations.jsonl")
            with open(live_file, "r+") as f:
                f.seek(0)
                f.truncate()
            fire_and_forget(API_URL_INFERENCE + "generate_initial", # this creates one sample for the live-question dataset, to avoid errors.
                            json={"prompt": persistent_prompt, "iteration": -1})
        # The following block does some "prepocessing", i.e. it resets values for the start of a new run
        learning_rate = base_learning_rate
        samples_by_iteration = base_samples_by_iteration
        logger.info("Starting a new training run")
        res = training_master() # start the run
        logger.info(f"Last run ended because of {res}")


"""
This method controls the training iterations. Every run consists of a number of iterations.
"""
def training_master():
    global tokenizer, learning_rate
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.pad_token if tokenizer.pad_token is not None else tokenizer.eos_token
    model = load_bnb_model()
    for dataset in available_datasets:
        if not os.path.exists(dataset_filepath + dataset  + "/" + str(max_tokens) + "_tokens/"):
            os.makedirs(dataset_filepath + dataset  + "/" + str(max_tokens) + "_tokens/")
            os.makedirs(dataset_filepath + dataset  + "/" + str(max_tokens) + "_tokens/" + "/iteration/")

    if not os.path.exists(dataset_filepath + "live/" + str(max_tokens) + "_tokens/"):
        os.makedirs(dataset_filepath + "live/" + str(max_tokens) + "_tokens/")
        os.makedirs(dataset_filepath + "live/" + str(max_tokens) + "_tokens/" + "/iteration/")
    conversations_address = os.path.join(dataset_filepath, "live", f"{max_tokens}_tokens", "conversations.jsonl")
    open(conversations_address, "a").close()
    if recreate_iteration_0: # after changing the inference-config, we want to answer the predefined-questions with the new settings.
        fire_and_forget(API_URL_INFERENCE+"infer_next_iters", json={"iteration": 0, "data": [available_datasets]})  #this answers the datasets with the base model
    for available_dataset in available_datasets: # delete every corrupted response from previous runs
        if os.path.exists(dataset_filepath + available_dataset  + "/" + str(max_tokens) + "_tokens/" + "/iteration/"):
            delete_subfolders(dataset_filepath + available_dataset + "/" + str(max_tokens) + "_tokens/" + "/iteration/", "0")
    highest = -1
    logger.debug("Start waiting for a language setting, to start training.")
    client.send_message("/wait_for_language", 1)
    while not os.path.exists(restart_flag): # we require inference_owner.py to set this restart_flag, which will be set when a language is selected in the gui.
        time.sleep(1)
    client.send_message("/wait_for_language", 0)
    delete_subfolders(filepath +"models/mistral_finetuned/") # delete adapters from previous runs. This is only performed now to avoid race conditions.
    os.remove(restart_flag)
    session_start = time.time()
    logger.debug("Received a language setting, starting training.")
    for i in range(num_total_iters):
        client.send_message("/elapsed_time", time.time()-session_start)
        logger.debug(f"elapsed time so far: {time.time()-session_start}")
        if time.time() - session_start > duration_of_a_run: # stop training after a time that is defined in config.yaml
            logger.debug(f"Run is ending, waiting for {last_iteration_time} seconds")
            client.send_message("/run_end", 1)
            time.sleep(last_iteration_time) #allow some interaction with the last iteration, defined in config.yaml
            client.send_message("/request_restart", 1)
            logger.debug("time ran out, restarting")
            return "time ran out"
        fire_and_forget(API_URL_INFERENCE+"generate_persistent", json={"prompt": persistent_prompt, "iteration": i}) # answer the persistent prompt
        if i > 0:
            if os.path.exists(dataset_filepath + str(max_tokens) + "_tokens/" + "live/" + "iteration"):
                available = os.listdir(dataset_filepath + str(max_tokens) + "_tokens/" + "live/" + "iteration")
                highest = max([int(a) for a in available])
                logger.debug(f"latest iteration for the live dataset in main is {highest} ")
        start_time = time.time()
        logger.info(f"===== trainer.py starts iteration {i} at {datetime.datetime.now()}")
        finetuned_modelpath = modelpath + "mistral_finetuned/epochs-" + str(num_train_epochs) + "_lr-" + str(
            learning_rate) + "/iter-" + str(i)
        time_before_tokenization = time.time()
        tokenized_live_dataset, length = tokenize_live_questions(highest) # returns a dataset of all live questions, and the number of responses that we have
        tokenized_dataset = load_questions_and_tokenize(i, ) # load and tokenize the predefined dataset
        logger.debug(f"tokenization took {time.time()-time_before_tokenization}")
        iteration_dataset = choose_datasets(tokenized_live_dataset, tokenized_dataset, i) # randomly choose samples to use for training
        fire_and_forget(API_URL_INFERENCE+"infer_next_iters", json={"iteration": i, "data": dataset_array[i+1:i+3]}) # prepare the dataset for the next iterations
        model = training_iteration(model, tokenizer, iteration_dataset, finetuned_modelpath, i) # train one iteration
        if verification_mode:
            fire_and_forget(API_URL_OWNER+"analyze_lora", json={"iteration": i})
        logger.info(f"===== finetuning.ipynb ended iteration {i} at {datetime.datetime.now()}")
        logger.info(f"This iteration had a duration of {time.time() - start_time}")
        model.save_pretrained(finetuned_modelpath)
        learning_rate = round(learning_rate + learning_rate_increase, lr_precision)

    logger.debug(f"Run is ending, waiting for {last_iteration_time} seconds")
    client.send_message("/run_end", 1)
    time.sleep(last_iteration_time)  # allow 180s of interaction with the last iteration
    client.send_message("/request_restart", 1)
    logger.debug("time ran out, restarting")
    return "natural end of training"

if __name__ == "__main__":
    global learning_rate, samples_by_iteration
    mp.freeze_support()
    mp.set_start_method("spawn", force=True)
    supervisor()