import json
import os
import logging
import shutil
import torch
import yaml
from datasets import Dataset, load_from_disk
from pydantic import BaseModel
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import requests
import re

### Shared Constant Declarations

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

general_config = config['general']
available_gpus = general_config['available_gpus']
model_name = general_config['model_name']
model_path = "/workspace/models/mistral-7b-v0.3"
hf_token = general_config['hf_token']
inference_port = general_config["inference_port"]
owner_port = general_config["owner_port"]
OSC_IP = general_config["osc_ip"]
OSC_PORT = general_config["osc_port"]
available_datasets = general_config["available_datasets"]
logging_level = general_config['logging']
verification_mode = general_config['verification_mode']

inference_config = config['inference']
max_tokens = inference_config['max_tokens']
temperature = inference_config['temperature']
system_prompts = inference_config['system_prompts']
system_language = inference_config['system_language']
system_prompt = system_prompts[system_language]
num_generated_responses = inference_config['num_generated_responses']
top_p = inference_config['top_p']
frequency_penalty = inference_config["frequency_penalty"]
num_context_messages = inference_config["num_context_messages"]
context_length_switch = inference_config["context_length_switch"]
late_num_context_messages = inference_config["late_num_context_messages"]

finetuning_config = config['finetuning']
num_train_epochs = finetuning_config['num_train_epochs']
batch_size = finetuning_config['batch_size']
learning_rate = finetuning_config['learning_rate']
lr_precision = finetuning_config['lr_precision']
learning_rate_increase = finetuning_config['learning_rate_increase']
num_total_iters = finetuning_config['num_total_iters']
grad_acc_steps = finetuning_config['grad_acc_steps']
save_steps = int(200/batch_size)
eval_steps = save_steps*2
dataset_array = finetuning_config['dataset_array']
samples_by_iteration = finetuning_config['samples_by_iteration']
sample_increase_by_iteration = finetuning_config['sample_increase_by_iteration']
live_samples_by_iteration = finetuning_config['live_samples_by_iteration']
max_per_prompt = finetuning_config['max_per_prompt']
persistent_prompt = finetuning_config['persistent_prompt']
recreate_iteration_0 = finetuning_config['recreate_iteration_0']
minimum_iteration_time = finetuning_config['minimum_iteration_time']
dataloader_num_workers = finetuning_config['dataloader_num_workers']
delete_live_questions_at_start = finetuning_config['delete_live_questions_at_start']
lora_target_modules = finetuning_config['lora_target_modules']
duration_of_a_run = finetuning_config['duration_of_a_run']
last_iteration_time = finetuning_config['last_iteration_time']

filepath = '/workspace/'
dataset_filepath = filepath + 'Datasets/'
modelpath = filepath + 'models/'
initial_modelpath = "/workspace/models/mistral-7b-v0.3"

API_URL_INFERENCE = "http://localhost:"+str(inference_port)+"/"
API_URL_OWNER = "http://localhost:"+str(owner_port)+"/"
restart_flag = filepath + "restart_flag"

# Shared Class Definitions

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
    iteration: int = 1

class AnalyzeRequest(BaseModel):
    iteration: int = 1

class LanguageRequest(BaseModel):
    language: str

class KillRequest(BaseModel):
    model: str

### Local Constant Declarations

logger = logging.getLogger(__name__)
CHAT_ARTIFACTS = [
    "<s>", "</s>", "[INST]", "[/INST]", "<|im_start|>", "<|im_end|>",
    "### System:", "### User:", "### Assistant:"
]

### Shared Function Declarations

# General Utility

def cutoff_at_last_punctuation(text: str) -> str:
    """
    Cuts text at the last sentence-ending punctuation (. ! ? …)
    Keeps everything up to and including that punctuation.
    """
    if not text:
        return text

    # Check for newline in last 50 characters
    tail = text[-50:]
    last_newline = tail.rfind("\n")
    if last_newline != -1:
        # Cut at that newline (convert tail index to full-text index)
        cutoff = len(text) - 50 + last_newline
        return text[:cutoff].strip()

    matches = list(re.finditer(r"[.!?…]", text))
    if not matches:
        return text.strip()

    last = matches[-1].end()
    return text[:last].strip()


def fire_and_forget(url, json):
    """
    This function will send a request and immediately timeout, letting the main script continue.
    :param url: the address to send the request to
    :param json: the provided payload
    :return: nothing, throws an expected, ignorable Exception.
    """
    try:
        if not json:
            requests.post(url, timeout=0.0001)
        else:
            requests.post(url, json=json, timeout=0.0001)
    except requests.exceptions.ReadTimeout:
        # This is expected. The server will continue processing anyway.
        pass


def strip_lora_suffix(name):
    # remove full patterns:
    #   .lora_A.<adapter>.weight
    #   .lora_B.<adapter>.weight
    if ".lora_A" in name:
        return name.split(".lora_A")[0]
    if ".lora_B" in name:
        return name.split(".lora_B")[0]
    return name


def compute_lora_delta_norm_ratio(model):
    """
    Computes:
        ratio = ||ΔW|| / ||W||
    for each LoRA-modified linear layer.

    Returns:
        {
            "per_layer": {layer_name: ratio},
            "global_ratio": float   # weighted by base weight norms
        }

    Interpretation:
    < 0.01	Very small adapter, minimal behavioral shift
    0.01–0.05	Mild influence
    0.05–0.15	Strong influence
    > 0.15	Heavy rewiring; model behaves significantly differently

    Example Usage:
    base_model = AutoModelForCausalLM.from_pretrained(base_path, device_map="cpu")
    lora_model = PeftModel.from_pretrained(base_model, lora_adapter_path)

    result = compute_lora_delta_norm_ratio(lora_model)

    print("Global ΔW/W ratio:", result["global_ratio"])
    for layer, ratio in result["per_layer"].items():
        print(f"{layer}: {ratio}")
    """

    # Collect params
    lora_A = {}
    lora_B = {}
    base_weights = {}

    for name, p in model.named_parameters():
        if ".lora_A" in name:
            base = strip_lora_suffix(name)
            lora_A[base] = p.detach().cpu().clone()

        elif ".lora_B" in name:
            base = strip_lora_suffix(name)
            lora_B[base] = p.detach().cpu().clone()

        elif name.endswith(".weight"):
            base = name[:-7]  # remove trailing ".weight"
            base_weights[base] = p.detach().cpu().clone()

    # Match A and B pairs
    common_prefixes = set(lora_A.keys()) & set(lora_B.keys())
    if not common_prefixes:
        logger.warning("No LoRA A/B parameter pairs found by name pattern.")

    per_layer = {}
    total_delta_norm = 0.0
    total_weight_norm = 0.0

    for prefix in sorted(common_prefixes):
        A = lora_A[prefix]  # shape: (r, in_features) or similar
        B = lora_B[prefix]  # shape: (out_features, r)

        # ensure shapes are compatible for B @ A
        try:
            A_mat = A.view(A.size(0), -1)  # (r, in)
            B_mat = B.view(B.size(0), -1)  # (out, r)
            # delta = B @ A  -> (out, in)
            delta = torch.matmul(B_mat, A_mat)
        except Exception as e:
            logger.exception("Could not compute delta for prefix %s: %s", prefix, e)
            continue

        delta_norm = torch.norm(delta).item()

        # find corresponding base weight (prefix + ".weight")
        base_w = base_weights.get(prefix)
        if base_w is None:
            # fallback: try some common suffix removal (e.g., 'q_proj' vs 'q_proj')
            # try to find any base weight that startswith prefix
            alt = None
            for k in base_weights:
                if k.endswith(prefix) or prefix.endswith(k):
                    alt = base_weights[k]
                    break
            base_w = alt

        if base_w is None:
            # cannot compute ratio without a base weight; still include delta
            weight_norm = 0.0
            ratio = float("inf")
        else:
            weight_norm = torch.norm(base_w).item()
            ratio = delta_norm / (weight_norm + 1e-12)

        per_layer[prefix] = {
            "delta_norm": float(delta_norm),
            "weight_norm": float(weight_norm),
            "ratio": float(ratio)
        }

        total_delta_norm += delta_norm
        total_weight_norm += weight_norm

    global_ratio = total_delta_norm / (total_weight_norm + 1e-12) if total_weight_norm > 0 else float("inf")

    return {"per_layer": per_layer, "global_ratio": float(global_ratio)}



def check_lora_is_active(llm: LLM, lora: LoRARequest, lora_name) -> bool:
    """
    rudimentary verification that our llm applies the provided lora, by verifying that its output differs from the base model
    :param llm: the base model
    :param lora: the LoRARequest item that we wish to apply
    :param lora_name: the name of the lora
    :return: True if the outputs differ, False else
    """
    active = llm.llm_engine.lora_manager.list_loras()
    if not lora_name in active:
        logger.error("LoRA is not registered.")

    base_out = llm.chat(
        ["test"],
        SamplingParams(temperature=0.0),
    )

    lora_out = llm.chat(
        ["test"],
        SamplingParams(temperature=0.0),
        lora_request=lora,
    )

    if base_out[0].outputs[0].text == lora_out[0].outputs[0].text:
        logger.error("LoRA seems not active.")
        return False
    else:
        return True


def delete_subfolders(path: str, keep_folders=None):
    """
    deletes every folder at the specified address. e.g., when given workspace/Datasets/AntonsFragen/iteration/ it will delete every folder in there, but not the iteration folder
    :param path: the path at which to delete all subfolders
    :param keep_folders: exceptions, i.e. a folder that is not to be deleted (typically 0, since the base model does not change.
    :return: No return value, this function completes silently.
    """
    for folder in os.listdir(path):
        if folder == keep_folders:
            logger.debug(f"not deleting folder {folder} ")
        else:
            logger.debug(f"deleting {path + folder}")
            shutil.rmtree(path + folder)

# For dataset_plain

def validate_prompt_is_not_already_templated(prompt: str) -> bool:
    """
    Detects whether a prompt already contains chat template tokens or structures.
    """
    for artifact in CHAT_ARTIFACTS:
        if artifact in prompt:
            return False
    return True


def validate_plain_dataset_structure(path: str) -> bool:
    """
    Verify that the dataset at `path` is loadable and returns a Dataset
    containing a 'prompt' field with strings.

    Suggested Usage:
    base_plain = os.path.join(dataset_filepath, dataset, f"{dataset}_plain")

    if not validate_plain_dataset_structure(base_plain):
        logger.error("Structural validation failed for dataset %s", base_plain)
        continue
    """
    if not os.path.exists(path):
        logger.error("Path does not exist: %s", path)
        return False

    try:
        ds = load_from_disk(path)

    except Exception as e:
        logger.exception("Failed to load dataset from %s: %s", path, e)
        return False

    if not isinstance(ds, Dataset):
        logger.error("Object loaded from %s is not a Dataset.", path)
        return False

    if "prompt" not in ds.column_names:
        logger.error("Dataset at %s does not contain required 'prompt' column.", path)
        return False

    # Type check each prompt
    for i, row in enumerate(ds):
        if "prompt" not in row:
            logger.error("Row %d has no 'prompt' field.", i)
            return False

        if not isinstance(row["prompt"], str):
            logger.error("Row %d 'prompt' is not a string: %r", i, row["prompt"])
            return False

        if not validate_prompt_is_not_already_templated(row["prompt"]):
            logger.error(f"Row {i} is already templated. Content: {row['prompt']}")

    return True


#  For Iteration datasets

# Local utility function

def _load_dataset_auto(path: str):
    """
    Try to load either a HF dataset directory or a JSONL file.
    Returns a Dataset or None.
    """
    if os.path.isdir(path):
        try:
            return load_from_disk(path)
        except Exception as e:
            logger.error("Failed to load HF dataset at %s: %s", path, e)
            return None

    if os.path.isfile(path) and path.endswith(".jsonl"):
        rows = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    rows.append(json.loads(line))
        except Exception as e:
            logger.error("Failed to load JSONL dataset at %s: %s", path, e)
            return None
        return Dataset.from_list(rows)

    logger.error("Path %s is neither a directory nor a .jsonl file.", path)
    return None


def validate_answered_dataset(path: str) -> bool:
    """
    Validate a dataset containing answered iterations.

    Expected format:
    {
        "prompt": str (no chat template)
        "response": str
        "iteration": int >= 0
    }

    Suggested Usage:
    from utility import validate_answered_dataset

    path = "/workspace/Datasets/.../iteration/i"
    ok = validate_answered_dataset(path)

    if not ok:
        print("Dataset invalid!")
    """
    ds = _load_dataset_auto(path)
    if ds is None:
        logger.error("Dataset could not be loaded: %s", path)
        return False

    required_fields = {"prompt", "response", "iteration"}

    for idx, row in enumerate(ds):
        # Field existence
        if not required_fields.issubset(row.keys()):
            logger.error("Missing required fields in row %d: %s", idx, row)
            return False

        prompt = row["prompt"]
        response = row["response"]
        iteration = row["iteration"]

        # Type checks
        if not isinstance(prompt, str):
            logger.error("Invalid prompt type in row %d: %s", idx, type(prompt))
            return False

        if not isinstance(response, str):
            logger.error("Invalid response type in row %d: %s", idx, type(response))
            return False

        if not (isinstance(iteration, int) and iteration >= 0):
            logger.error("Invalid iteration value in row %d: %s", idx, iteration)
            return False

        # No templated chat markup in prompt
        if not validate_prompt_is_not_already_templated(prompt):
            logger.error("Prompt already contains chat-template markup in row %d: %s", idx, prompt)
            return False

    return True


### For Tokenized Dataset
def validate_tokenized_dataset(
    ds: Dataset,
    tokenizer,
) -> bool:
    """
    Ensures HF 'tokenized_dataset' has the correct structure for LM training.
    Expected Format:
        A HF-Dataset with columns input_ids, attention_mask, labels, prompt, response, iteration
    Suggested Usage:
    from utility import validate_tokenized_dataset

    ok = validate_tokenized_dataset(
        tokenized_dataset,
        tokenizer=tokenizer
    )

    if not ok:
        logger.error("INVALID TOKENIZED DATASET")
    """

    required = ["input_ids", "attention_mask", "labels",
                "prompt", "response", "iteration"]

    # Check columns
    for key in required:
        if key not in ds.column_names:
            logger.error("Dataset missing required column '%s'", key)
            return False

    # Row checks
    for idx, row in enumerate(ds):
        # ---- type checks ----
        if not isinstance(row["prompt"], str):
            logger.error("Invalid prompt type in row %d.", idx)
            return False
        if not isinstance(row["response"], str):
            logger.error("Invalid response type in row %d.", idx)
            return False
        if not (isinstance(row["iteration"], int) and row["iteration"] >= 0):
            logger.error("Invalid iteration value in row %d.", idx)
            return False

        # ---- prompt must NOT be templated ----
        if not validate_prompt_is_not_already_templated(row["prompt"]):
            logger.error("Prompt contains chat template markers in row %d: %s",
                         idx, row["prompt"])
            return False
        if not validate_prompt_is_not_already_templated(row["response"]):
            logger.error("Response contains chat template markers in row %d.", idx)
            logger.error(f"Corresponding row: {row}")
            logger.error(f"Corresponding dataset: {ds}")
            return False

        # ---- structure of tokenized ----
        ids = row["input_ids"]
        mask = row["attention_mask"]
        labels = row["labels"]

        if not (isinstance(ids, list) and isinstance(mask, list) and isinstance(labels, list)):
            logger.error("Tokenized fields must be lists in row %d.", idx)
            return False

        if not (len(ids) == len(mask) == len(labels)):
            logger.error("Mismatched token lengths in row %d.", idx)
            return False

        # ---- decode a preview and check chat template presence ----
        try:
            decoded = tokenizer.decode(ids, skip_special_tokens=False)
        except Exception as e:
            logger.error("Failed to decode input_ids at row %d: %s", idx, e)
            return False

        # Must contain a template (e.g. <s>[INST] ... [/INST])
        if "[INST]" not in decoded or "[/INST]" not in decoded:
            logger.error("Decoded text missing [INST] template in row %d: %s",
                         idx, decoded[:250])
            return False

        # Should contain system + user + assistant in one sequence
        # System included at beginning
        if decoded.count("[INST]") != 1 or decoded.count("[/INST]") != 1:
            logger.error("Unexpected number of INST markers in row %d: %s",
                         idx, decoded[:250])
            return False

    return True