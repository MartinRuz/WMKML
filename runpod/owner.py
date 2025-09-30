from fastapi import FastAPI
from vllm import LLM, SamplingParams
import uvicorn
from transformers import Trainer
import time
from huggingface_hub import login
import os

login(token=os.environ["HF_TOKEN"])
CHECKPOINT_PATH = "./checkpoints/latest_adapter.pt"
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

app = FastAPI()

# Load base model once into GPU
inference_llm_1 = LLM(model=BASE_MODEL, dtype="bfloat16", gpu_memory_utilization=0.45, hf_token="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", tokenizer_mode="mistral", device="cuda:1")
#training_llm = LLM(model=BASE_MODEL, dtype="bfloat16", gpu_memory_utilization=0.2, hf_token="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", tokenizer_mode="mistral")

@app.get("/generate")
def generate_inference(prompt: str, parameters: SamplingParams):
    outputs = inference_llm_1.generate([prompt], parameters)
    return {"text": outputs[0].outputs[0].text}

#@app.get("/generate_training")
#def generate_training(prompt: str, parameters: SamplingParams):
#    outputs = training_llm.generate([prompt], parameters)
#    return {"text": outputs[0].outputs[0].text}

@app.post("/reload")
def reload_adapter(path: str = CHECKPOINT_PATH):
    """Reload latest adapter weights written by training process."""
    inference_llm_1.load_lora(path) #TODO: time this
    return {"status": f"reloaded adapter from {path}"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
