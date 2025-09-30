# inference.py
import requests
import yaml
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
from vllm import SamplingParams

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

general_config = config['general']
model_name = general_config['model_name']
tokenizer = AutoTokenizer.from_pretrained(model_name)
hf_token = general_config['hf_token']
wandb_key = general_config['wandb_key']
os.environ["WANDB_API_KEY"] = wandb_key

inference_config = config['inference']
max_tokens = inference_config['max_tokens']
temperature = inference_config['temperature']
num_answers = inference_config['num_answers']

filepath = 'WMKML/'
dataset_filepath = filepath + 'Datasets/'


def generate_answers(prompt_tokens):
    parameters = SamplingParams(
                max_new_tokens=max_tokens,
                temperature=temperature,
                num_return_sequences=num_answers,
                do_sample=True,
                eos_token_id = tokenizer.eos_token_id,
                pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.unk_token_id
    )
    output_tokens = requests.get(
        "http://localhost:8000/generate_inference",
        params={"prompt": prompt_tokens, "parameters": parameters
                }
        )
    generated_text = output_tokens.json()["text"]
    return generated_text


def format_question(question):
    return {"role": "user", "content": question}


### This cell takes input 'examples' and returns a preprocessed and tokenized version.
# This means converting the input array of dictionaries into an LLM-training compatible format.
# Input format: array[dictionary{'role': ..., 'content': ...}]
# Output format: array[tokenizer("[INST] question [\INST]answer{eos_token}")]
# TODO: better documentation and saving the ouput to a file instead of returning it, probably, when we actually know what we want
def preprocess_and_tokenize_data(examples):
    formatted_texts = []
    # Iterate through each conversation in the batch
    for conversation in examples["messages"]:
        formatted_text = ""
        # Iterate through the messages within the current conversation
        # Each 'message' is a dictionary {'role': ..., 'content': ...}
        for message in conversation:
            role = message["role"]
            content = message["content"]

            # Format the message based on the role
            if role == "user":
                # Format user messages with [INST] tags
                formatted_text += f"[INST] {content} [/INST]"
            elif role == "assistant":
                # Ensure assistant response is immediately after [/INST]
                # And maybe add an EOS token if this is the end of the turn
                formatted_text += f"{content}{tokenizer.eos_token}" # Add EOS token after assistant turn

        formatted_texts.append(formatted_text)
        print("= Inference debug from preprocess_and_tokenize_data():")
        print("= Formatted text:")
        print("= "+formatted_texts[0])
        print("= Length of formatted texts: " + str(len(formatted_texts)))
        print("= Length of examples: " + str(len(examples["messages"])))
        print("= End of Database debug")
    # Tokenize the complete formatted conversation string
    tokenized_outputs = tokenizer(
        formatted_texts,
        truncation=True,         # Ensure inputs fit within max length
        padding="max_length",    # Add padding tokens to make all inputs equal length
        max_length=512,          # Mistral's max token limit
        return_attention_mask=True, # Include attention mask
    )

    tokenized_outputs["labels"] = tokenized_outputs["input_ids"].copy()
    return tokenized_outputs


def scan_for_input():
    return "Wer ist Friedrich Merz?"


if __name__ == "__main__":

    iter = 0
    while True:
        modelpath = filepath + 'models/iter_' + iter
        lorapath = modelpath + '_lora'
        input = scan_for_input()  # this is a dummy for now
        formatted_question = format_question(input)
        prompt_tokens = preprocess_and_tokenize_data([formatted_question])
        answers = generate_answers(prompt_tokens)