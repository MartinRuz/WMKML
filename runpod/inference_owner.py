# inference.py
import json
import time

import yaml
import os

from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer
from peft import PeftModel
import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from fastapi import FastAPI, Request, BackgroundTasks
import uvicorn
from pydantic import BaseModel

if torch.cuda.device_count() > 2:
    print("using gpu 2")
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

general_config = config['general']
model_name = general_config['model_name']
model_path = "/workspace/models/mistral-7b-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.pad_token if tokenizer.pad_token is not None else tokenizer.eos_token
hf_token = general_config['hf_token']
wandb_key = general_config['wandb_key']
inference_port = general_config["inference_port"]
owner_port = general_config["owner_port"]
os.environ["WANDB_API_KEY"] = wandb_key

inference_config = config['inference']
max_tokens = inference_config['max_tokens']
temperature = inference_config['temperature']
num_answers = inference_config['num_answers']
num_generated_responses = inference_config['num_generated_responses']
top_p = inference_config['top_p']
filepath = '/workspace/'
dataset_filepath = filepath + 'Datasets/'


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9
    iteration: int = 1

class InferenceRequest(BaseModel):
    iteration: int = 1
    data: list

class ReloadRequest(BaseModel):
    path: str

app = FastAPI()

inference_llm = LLM(model=model_path, dtype="bfloat16", gpu_memory_utilization=0.9, tokenizer_mode="mistral", enable_lora=True)
adapter_path = ""


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
    tokenized["iteration"] = questions["iteration"]
    return tokenized


@app.post("/generate_inference")
def multi_inference_and_store(req: GenerateRequest):
    print("inference received multi_inference")
    start_time = time.time()
    params = SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p, n=num_generated_responses)
    print(f"Parameters of multi_inference: max_tokens={max_tokens}, temperature={temperature}, top_p={top_p}, num_generated_responses={num_generated_responses}")
    if adapter_path != "":
        outputs = inference_llm.generate([req.prompt], params, lora_request=LoRARequest("gui_adapter", 100+req.iteration, adapter_path))
    else:
        outputs = inference_llm.generate([req.prompt], params)
    responses = [out.text for out in outputs[0].outputs]
    for i, resp in enumerate(responses):
        print(f"\n=== Response {i + 1} ===\n{resp}")
        with open(dataset_filepath + "live/answered_conversations.jsonl", "a") as f:
            json.dump({
                "prompt": req.prompt,
                "response": resp,
                "iteration": req.iteration
            }, f)
            f.write("\n")
    print("multi_inference took " + str(time.time() - start_time))

@app.post("/infer_next_iters")
def perform_next_iterations(req: InferenceRequest):
    # saves a tokenized dataset
    print("perform_next_iteration started")
    start_time = time.time()
    data = req.data
    iter = req.iteration
    params = SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p, n=1)
    set_of_data = {item for sublist in data for item in sublist}
    print("answering ", set_of_data)
    for dataset in set_of_data:
        results = []
        question_dataset = load_from_disk(dataset_filepath + dataset + "/" + dataset + "_plain/")
        questions = [x["prompt"] for x in question_dataset]
        if adapter_path != "":
            outputs = inference_llm.generate(questions, params,
                                         lora_request=LoRARequest("gui_adapter", 100 + req.iteration, adapter_path))
        else:
            outputs = inference_llm.generate(questions, params)
        for q, out in zip(questions, outputs):
            for generated in out.outputs:
                results.append({
                    "prompt": q,
                    "response": generated.text,
                    "iteration": iter,
                })
        out_dir = os.path.join(dataset_filepath, dataset, "iteration", str(iter))
        print("saving dataset " + dataset + " to location " + out_dir)
        os.makedirs(out_dir, exist_ok=True)
        answered_dataset = Dataset.from_list(results)
        tokenized_dataset = answered_dataset.map(tokenize_dataset, batched=True)
        tokenized_dataset.save_to_disk(out_dir)
    live_questions = []
    live_results = []
    with open(filepath + "Datasets/live/answered_conversations.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            live_questions.append(
                {"prompt": entry["prompt"], "iteration": iter})
    if adapter_path != "":
        outputs = inference_llm.generate(live_questions, params,
                                         lora_request=LoRARequest("gui_adapter", 100 + req.iteration, adapter_path))
    else:
        outputs = inference_llm.generate(live_questions, params)
    for q, out in zip(live_questions, outputs):
        for generated in out.outputs:
            live_results.append({
                "prompt": q["prompt"],
                "response": generated.text,
                "iteration": q["iteration"]
            })
    out_dir = os.path.join(dataset_filepath, "live", "iteration", str(iter))
    os.makedirs(out_dir, exist_ok=True)
    answered_dataset = Dataset.from_list(live_results)
    tokenized_dataset = answered_dataset.map(tokenize_dataset, batched=True)
    tokenized_dataset.save_to_disk(out_dir)

    print("perform_next_iteration ended after " + str(time.time()-start_time) + "seconds")


@app.post("/reload")
def reload_adapter(req: ReloadRequest):
    print("Inference received reload")
    global adapter_path
    adapter_path = req.path
    return {"status": f"reloaded adapter from {req.path}"}


@app.post("/generate_persistent")
def generate_persistent(req:GenerateRequest):
    outfile = "logs/persistent_answers.txt"
    params = SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p, n=1)
    if req.iteration > 0:
        lora_request = LoRARequest("persistent", 200 + req.iteration, adapter_path)
        output = inference_llm.generate([req.prompt], params, lora_request=lora_request)[0].outputs[0].text
    else:
        print("correctly identified iteration 0")
        output = inference_llm.generate([req.prompt], params)[0].outputs[0].text
    print(f"In iteration {req.iteration} our model answered the persistent prompt {req.prompt} with the following answer:\n{output}")
    with open(outfile, 'a') as f:
        f.write(f"In iteration {req.iteration} our model answered the persistent prompt {req.prompt} with the following answer:\n{output}\n\n")
        f.close()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=inference_port)
