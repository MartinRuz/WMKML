import json

import requests
import yaml
from datasets import load_from_disk, Dataset, concatenate_datasets
from fastapi import FastAPI, Request, BackgroundTasks
from vllm.lora.request import LoRARequest
from vllm import LLM, SamplingParams
import uvicorn
from transformers import Trainer, AutoTokenizer
import time
import os
from pydantic import BaseModel
from huggingface_hub import snapshot_download
from pythonosc import udp_client

### Configure OSC Server
OSC_IP = "164.92.178.77"
OSC_PORT =21000
#client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
#client.send_message("/user_activity", data['status'])

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

general_config = config['general']
inference_config = config['inference']
max_tokens = inference_config['max_tokens']
temperature = inference_config['temperature']
num_answers = inference_config['num_answers']
model_path = "/workspace/models/mistral-7b-v0.3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
filepath = '/workspace/'
dataset_filepath = filepath + 'Datasets/'
#conversations_csv = open(dataset_filepath + "live/conversations.csv", "a")
app = FastAPI()
inference_port = general_config["inference_port"]
owner_port = general_config["owner_port"]
API_URL_INFERENCE = "http://localhost:"+str(inference_port)+"/"

ITERATION = 1

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 128
    temperature: float = 0.7
    top_p: float = 0.9

class ReloadRequest(BaseModel):
    path: str

# Load base model once into GPU
inference_llm_gui = LLM(model=model_path, dtype="bfloat16", gpu_memory_utilization=0.9, tokenizer_mode="mistral", enable_lora=True)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.pad_token if tokenizer.pad_token is not None else tokenizer.eos_token
adapter_path = filepath + "models/"


#First option for how to handle new prompts, probably slower than intermediate storing.
def append_new_prompts_immediately(req, outputs):
    prompt = [f"[INST] {req.prompt} [/INST]{outputs[0].outputs[0].text}{tokenizer.eos_token}"]
    tokenized_prompt = tokenizer(prompt, truncation=True, padding="max_length", max_length=512, )
    tokenized_prompt["labels"] = tokenized_prompt["input_ids"].copy()
    new_dataset = Dataset.from_dict(tokenized_prompt)
    if os.path.exists(dataset_filepath + "live/conversations_tokenized/"):
        tokenized_dataset = load_from_disk(dataset_filepath + "live/conversations_tokenized/")
        updated_dataset = concatenate_datasets([tokenized_dataset, new_dataset])
        updated_dataset.save_to_disk(dataset_filepath + "live/conversations_tokenized/")
    else:
        new_dataset.save_to_disk(dataset_filepath + "live/conversations_tokenized/")


def store_new_prompts(req, answer):
    with open(dataset_filepath + "live/conversations.jsonl", "a") as f:
        f.write(json.dumps({"prompt": req.prompt, "response": answer}) + "\n")



### This method is used to perform inference for the gui.
@app.post("/generate_gui")
def generate_inference(req: GenerateRequest):
    print("Owner received generate_inference")
    start_time = time.time()
    params = SamplingParams(max_tokens=max_tokens, temperature=req.temperature, top_p=req.top_p)
    if ITERATION > 1:
        outputs = inference_llm_gui.generate([req.prompt], params, lora_request=LoRARequest("gui_adapter", ITERATION, adapter_path))
    else:
        outputs = inference_llm_gui.generate([req.prompt], params)
    answer = outputs[0].outputs[0].text
    store_new_prompts(req, answer)
    payload = {"prompt": req.prompt, "iteration": ITERATION}
    requests.post(API_URL_INFERENCE + "generate_inference", json=payload)
    print("generate_inference took " + str(time.time()-start_time))
    return {"text": outputs[0].outputs[0].text, "iteration": ITERATION}


@app.post("/reload")
def reload_adapter(req: ReloadRequest):
    print("Owner received reload")
    """Reload latest adapter weights written by training process."""
    global adapter_path
    adapter_path = req.path
    global ITERATION
    ITERATION += 1 # keep track of the current iteration
    return {"status": f"reloaded adapter from {req.path}"}


@app.post("/typing_event")
async def typing_event(req: Request):
    data = await req.json()
    print(f"User is {data['status']}")
    client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
    client.send_message("/user_activity", data['status'])
    return {"ok": True}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=owner_port)
