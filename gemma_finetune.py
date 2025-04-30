import json

import torch
from datasets import load_dataset, Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM, Gemma3ForCausalLM, TrainingArguments, Trainer,
                          DataCollatorForLanguageModeling, AutoConfig, BitsAndBytesConfig)
from accelerate import Accelerator
import os
from peft import get_peft_model, LoraConfig, TaskType


def process_conversation(dataset):
    samples = []

    for item in dataset:
        dialogue = item["messages"]  # this is a list of turns: [{"role": ..., "content": ...}, ...]

        for i in range(0, len(dialogue) - 1, 2):
            if dialogue[i]["role"] == "user" and dialogue[i+1]["role"] == "assistant":
                samples.append({
                    "instruction": dialogue[i]["content"],
                    "response": dialogue[i+1]["content"]
                })

    return samples


def preprocess_function(example):
    max_length=512
    # Format prompt-answer pair
    q, a = example["text"].split("\n")
    full_input = f"{q}\n{a}"

    # Tokenize the entire text
    tokenized = tokenizer(full_input,
                          padding="max_length",
                          truncation=True,
                          max_length=max_length)

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    # Now build labels — mask everything before the answer
    answer_start = full_input.find("Assistant:") + len("Assistant:")
    labels_text = full_input[answer_start:]
    labels_tokenized = tokenizer(labels_text,
                                 padding="max_length",
                                 truncation=True,
                                 max_length=max_length)

    labels = labels_tokenized["input_ids"]

    # Now mask non-answer tokens with -100 (ignore index in loss)
    pre_answer_token_count = tokenizer(full_input[:answer_start],
                                       truncation=True,
                                       max_length=max_length)["input_ids"]
    answer_offset = len(pre_answer_token_count)
    labels = [-100] * answer_offset + labels[answer_offset:]

    # Ensure label list is same length as input_ids
    labels = labels[:max_length]
    labels += [-100] * (max_length - len(labels))

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

def inference(model, tokenizer, model_ft, tokenizer_ft):
    inputs = ["The capital of France is",
              #"Hospital triage in modern day warfare has the following categories:",
              #"What are typical injuries that I might suffer from a 5.56mm NATO?",
              "Wer ist Friedrich Merz und ist er Kanzlerkandidat?",
              "Was ist das Bündnis Sarah Wagenknecht?",
              "What is happening in Yemen with the Houthis?"]

    for i in inputs:
        input = tokenizer(i, return_tensors="pt")
        input = {k: v.to("cuda") for k, v in input.items() if isinstance(v, torch.Tensor)}
        outputs = model.generate(**input, max_new_tokens=100)
        print("Non-tuned: " + tokenizer.decode(outputs[0], skip_special_tokens=True))

        input_ft = tokenizer_ft(i, return_tensors="pt")
        input_ft = {k: v.to("cuda") for k, v in input_ft.items() if isinstance(v, torch.Tensor)}
        outputs_ft = model_ft.generate(**input_ft, max_new_tokens=100)
        print("Tuned: " + tokenizer_ft.decode(outputs_ft[0], skip_special_tokens=True))



def is_dw(example):
    label = example.get("label", "")
    if isinstance(label, str):
        return label.split(";")[0].strip() == "Deutsche Welle"
    return False


def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding=True, max_length=1024)


if __name__ == "__main__":
    dataset_name = "latest-news-lora"

    if dataset_name == "news-dataset":
        dataset = load_dataset("R3troR0b/news-dataset")
        train_data = dataset["train"]
        filtered_data = train_data.filter(is_dw)
        model_id = "google/gemma-3-1b-it"

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = Gemma3ForCausalLM.from_pretrained(model_id, torch_dtype="auto", attn_implementation="eager").to("cuda")
        filtered_data = filtered_data.select(range(3000))
        tokenized_dataset = filtered_data.map(tokenize, batched=True, remove_columns=["label"])

        training_args = TrainingArguments(
            output_dir="./gemma-1b-finetuned/"+dataset_name,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            save_steps=500,
            logging_steps=50,
            num_train_epochs=1,  # Start low and increase
            fp16=False,
            bf16=True,
            save_total_limit=2,
            logging_dir="./logs",
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

    if dataset_name == "latest-news":
        data = []
        with open("latest_qa_dataset.jsonl", encoding="utf-8") as f:
            for line in f:
                qa = json.loads(line)
                text = f"User: {qa['question']}\nAssistant: {qa['answer']}"
                data.append({"text": text})
        dataset = Dataset.from_list(data)

        model_id = "google/gemma-3-1b-it"
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        tokenized_dataset = dataset.map(tokenize, batched=True)

        model = Gemma3ForCausalLM.from_pretrained(model_id, torch_dtype="auto", attn_implementation="eager").to("cuda")

        training_args = TrainingArguments(
            output_dir="./gemma-1b-finetuned/"+dataset_name,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            logging_steps=10,
            save_steps=100,
            bf16=True,
            save_total_limit=2,
            warmup_ratio=0.1,
            learning_rate=5e-5,
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

    if dataset_name == "latest-news-lora":
        data = []
        with open("latest_qa_dataset.jsonl", encoding="utf-8") as f:
            for line in f:
                qa = json.loads(line)
                text = f"User: {qa['question']}\nAssistant: {qa['answer']}"
                data.append({"text": text})
        dataset = Dataset.from_list(data)

        model_id = "google/gemma-3-1b-it"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_quant_type="nf4",
        )
        model = Gemma3ForCausalLM.from_pretrained(model_id, torch_dtype="auto", attn_implementation="eager",
                                                  quantization_config=bnb_config).to("cuda")

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        tokenized_dataset = dataset.map(preprocess_function, batched=False)

        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        training_args = TrainingArguments(
            output_dir="./gemma-1b-finetuned/" + dataset_name,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            logging_steps=10,
            save_steps=100,
            bf16=True,
            save_total_limit=2,
            warmup_ratio=0.1,
            learning_rate=1e-5,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
        )


    train = not os.path.exists("./gemma-1b-finetuned/"+dataset_name+"/tokenizer.json")
    if train:
        print("training")
        original_file_path = "./gemma-1b-original"
        model.save_pretrained(original_file_path)
        tokenizer.save_pretrained(original_file_path)
        trainer.train()
        trainer.save_model("./gemma-1b-finetuned/"+dataset_name)
        tokenizer.save_pretrained("./gemma-1b-finetuned/"+dataset_name)
        model_ft = model.to("cuda")
        tokenizer_ft = tokenizer
        model = AutoModelForCausalLM.from_pretrained(original_file_path, torch_dtype="auto").to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(original_file_path)
    else:
        print("not training")
        model_path = "./gemma-1b-finetuned/"+dataset_name
        tokenizer_ft = AutoTokenizer.from_pretrained(model_path)
        model_ft = AutoModelForCausalLM.from_pretrained(model_path,  torch_dtype="auto").to("cuda")

    inference(model, tokenizer, model_ft, tokenizer_ft)
