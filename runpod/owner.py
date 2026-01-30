import datetime

from utility import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print("using gpu 0")

import torch
from fastapi import FastAPI, Request, BackgroundTasks
from vllm.lora.request import LoRARequest
from vllm import LLM, SamplingParams
import uvicorn
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import json
from pythonosc import udp_client
import multiprocessing as mp
import threading
import logging
from peft import PeftModel


_save_lock = threading.Lock()
_reload_lock = threading.Lock()
_adapter_lock = threading.Lock()
_swap_lock = threading.Lock()
_inference_lock = threading.Lock()
_adapter_ready_events = {}
adapter_path = ""
ITERATION = -1
logging.basicConfig(level=logging_level, format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s')
logger = logging.getLogger("owner")
client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
message_history = [{"role": "system", "content": system_prompt}]
initial_num_context_messages = num_context_messages
app = FastAPI()



logger.debug(f"visible devices: {torch.cuda.device_count()}")
logger.debug(f"current device: {torch.cuda.current_device()}")
logger.debug(f"gpu name: {torch.cuda.get_device_name()}")


"""
This is a helper function to keep a message-history, in order to supply context to the llm.
message_history is the message-history that was created so far, a list of dicts with "role" and "content" fields
messages is the message that should be added, it should follow the dict-format with "role" and "content" fields
system_prompt is the current system prompt
iteration is the current iteration
"""
def extend_message_history(message_history, messages, system_prompt, iteration):
    global num_context_messages
    message_history[0] = {"role": "system", "content": system_prompt}
    message_history.append(messages) # append the message (which is prompt or answer) to the end
    while len(message_history) > 2 * num_context_messages + 1: # we keep *num_context_messages* times two messages (prompt + answer) and the system prompt
        message_history.pop(1) # if we exceed the threshold, we remove one set of prompt + answer from the beginning, aka the oldest
        message_history.pop(1)
    logger.debug(f"In iteration {iteration} we have a context length of {num_context_messages} and after popping, we have {message_history}")
    return message_history


"""
Store all live-questions in a jsonl file, to be used by the trainer for finetuning.
req is the object containing the prompt
answer is the generated answer
iteration is the current iteration
"""
def store_new_prompts(req, answer, iteration):
    out_dir = os.path.join(dataset_filepath, "live", f"{max_tokens}_tokens")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "conversations.jsonl")
    if not os.path.exists(path):
        open(path, 'w').close()
    with _save_lock:
        with open(path, "a") as f:
            f.write(json.dumps({"prompt": req.prompt, "response": answer, "iteration": iteration}) + "\n")
            f.flush()
            os.fsync(f.fileno())


"""
Loading newly trained adapters into the VRAM takes time (approx. 1 min). We offload this time to a secondary llm, since vllm does not support multitasking.
Speed is a primary concern for the main llm, users should never have to wait 1 minute for a response.
path is the path of the new adapter.
iteration is the incoming iteration.
"""
def generate_once(path, iteration):
    logger.debug(f"loading the following path: {path}")
    global swap_llm, inference_llm_gui
    global ITERATION, adapter_path
    prompt = persistent_prompt
    start_wait = time.time()
    last_size = -1
    critical_file = "adapter_model.safetensors"
    while critical_file not in os.listdir(path):
        time.sleep(0.2)
        logger.debug(f"waiting for the file to exist {time.time()-start_wait}")

    while True:
        size = (os.path.getsize(path+"/"+critical_file))
        if size != last_size: # wait for the writing operation that was started by the trainer, to finish
            time.sleep(3)
            logger.debug(f"waiting for the file size not to change {time.time()-start_wait}")
            last_size = size
        else:
            break
    logger.info(f"Owner received generate_once with prompt {prompt}")
    params = SamplingParams(max_tokens=1, temperature=0.0)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    maybe_lora = LoRARequest(f"adapter_iter_{iteration}", 100 + iteration, path)
    logger.debug("maybe_lora has been created")
    _ = swap_llm.chat(messages=messages, sampling_params=params, lora_request=maybe_lora) # generate one dummy output.
    logger.info(f"generate_once generated a warmup answer")
    with _swap_lock:
        ITERATION = iteration
        adapter_path = path
    logger.info(f"successfully swapped to iteration {ITERATION}")



"""
This method is used to perform inference for the gui.
Also, it informs the osc-process about prompt, response and iteration.
Also, it tells the inference-script to perform inference in the background.
req is the json-object that contains the prompt
"""
@app.post("/generate_gui")
async def generate_inference(req: GenerateRequest, background_tasks: BackgroundTasks):
    global message_history, inference_llm_gui
    logger.info(f"Owner received generate_inference with prompt {req.prompt}")
    start_time = time.time()
    messages = {"role": "user", "content": req.prompt}
    with _inference_lock:
        engine = inference_llm_gui
        iteration = ITERATION
        adapter_path_local = adapter_path
        message_history = extend_message_history(message_history, messages, system_prompt, iteration) # create the prompt object with the respective context
        history_for_model = [dict(m) for m in message_history]
    client.send_message("/prompt", req.prompt) # send the prompt to the osc-port for further use
    params = SamplingParams(max_tokens=max_tokens, temperature=req.temperature, top_p=req.top_p, frequency_penalty=frequency_penalty)

    logger.debug(f"message_history: {history_for_model}")

    maybe_lora = None
    if adapter_path_local != "": # if we have already trained an adapter, we use the LoRARequest object (aka the adapter) with our prompt
        maybe_lora=LoRARequest(f"adapter_iter_{iteration}", 100 + iteration, adapter_path_local)

    try:
        multi_inference = time.time()
        outputs = engine.chat(messages=history_for_model, sampling_params=params, lora_request=maybe_lora) # prompt the model
        logger.debug(f"single inference took {time.time()-multi_inference}")
    except Exception as e:
        logger.exception("vLLM generate failed: %s", e)
        return {"error": "internal model error"}

    answer = outputs[0].outputs[0].text # extract the text-answer
    background_tasks.add_task(store_new_prompts, req, answer, iteration) # save the prompt-answer pair
    processed_answer = cutoff_at_last_punctuation(answer) # for nicer outputs, we perform a basic postprocessing, to decrease the number of cut-off answers
    client.send_message("/response", processed_answer) # send the response to the osc-port
    client.send_message("/iteration", iteration+1) # also send the current iteration (+1 since we start at -1 here)
    fire_and_forget(API_URL_INFERENCE + "generate_inference", json={"prompt": req.prompt, "iteration": iteration}) # notify the inference-script to generate more answers for the dataset
    messages = {"role": "assistant", "content": processed_answer}
    message_history = extend_message_history(message_history, messages, system_prompt, iteration) # append the answer to the history
    logger.info("generate_inference took " + str(time.time()-start_time))
    return {"text": processed_answer, "iteration": iteration}


"""
For compatibility reasons, this function is named change_language, however besides changing the language, this also resets the system!
The language (aka the system_prompt which defines the language for the llm) will be set according to the req-json
There are a few global parameters that are changed during a run: adapter_path, ITERATION, message_history, num_context_messages.
All of them are reset to their respective initial values. If you should ever add more changing parameters, reset them here.
Warning:
This function must be called *safely*.
The adapter_path is set to "", which means that you must ensure that there are no concurrent inference-calls, that expect adapter_path to contain the path to an adapter.
req contains the selected language
"""
@app.post("/language")
def change_language(req: LanguageRequest):
    global system_language, system_prompt
    if req.language == "de":
        logger.debug("switching language to de")
        system_language = req.language
        system_prompt = system_prompts[system_language] # select the correct system-prompt.
        logger.debug(f"new system prompt is {system_prompt}")
    elif req.language == "en":
        logger.debug("switching language to en")
        system_language = req.language
        system_prompt = system_prompts[system_language]
        logger.debug(f"new system prompt is {system_prompt}")
    else:
        logger.exception("unexpected language")

    global ITERATION, adapter_path, message_history, num_context_messages # reset all of these parameters to their initial value.
    adapter_path = ""
    ITERATION = -1
    message_history = [{"role": "system", "content": system_prompt}]
    num_context_messages = initial_num_context_messages
    logger.debug(f"setting context length to {num_context_messages}")
    logger.debug("Owner has set language and restarted the system.")
    return {"status": "success"}


"""
This method is not maintained anymore.
It triggers a basic analysis of the trained lora-adapters, measuring the change that the new adapter introduces.
"""
@app.post("/analyze_lora")
def analyze_lora(req: AnalyzeRequest):
    iter = req.iteration
    base_model = AutoModelForCausalLM.from_pretrained(initial_modelpath, device_map="cpu")
    lora_model = PeftModel.from_pretrained(base_model, adapter_path)

    result = compute_lora_delta_norm_ratio(lora_model)

    logger.debug(f"Global ΔW/W ratio in iteration {iter}: {result['global_ratio']}")
    for layer, ratio in result["per_layer"].items():
        logger.debug(f"{layer}: {ratio}")


"""
This method loads a new adapter.
For this, we mostly need to change the adapter_path to the new path, and increase the iteration.
However, due to the delay that this introduces (as described earlier), we first perform a warm-up iteration with the dummy swap_llm.
req contains the new path and iteration.
"""
@app.post("/reload")
def reload_adapter(req: ReloadRequest):
    logger.info(f"Owner received reload, with incoming iteration {req.iteration} at {datetime.datetime.now()}")
    """Reload latest adapter weights written by training process."""
    global adapter_path, ITERATION, num_context_messages
    threading.Thread( # call the warmup function
        target=generate_once,
        args=(req.path, req.iteration),
        daemon=True,
    ).start()
    logger.debug(f"Owner completed reload")
    if req.iteration == context_length_switch:
        logger.debug(f"reducing context length from {num_context_messages} to {late_num_context_messages}")
        num_context_messages = late_num_context_messages
    return {"status": f"reloaded adapter from {req.path}"}


"""
This method notifies osc, when the user is typing in the gui.
For the actual product, we use another gui that does not require this functionality, it is merely for testing purposes.
req contains the information, whether the user started typing or stopped typing.
"""
@app.post("/typing_event")
async def typing_event(req: Request):
    data = await req.json()
    logger.debug(f"User is {data['status']}")
    client.send_message("/user_activity", data['status']) # the osc-port must know when a user is typing, or no longer typing, in order to adjust the music.
    return {"ok": True}


"""
Initial method, that loades the two llms. This method takes around 10 minutes on an A100 GPU.
"""
def main():
    global inference_llm_gui, swap_llm, tokenizer
    # Load base model once into GPU
    if torch.cuda.is_bf16_supported():
        data_type = "bfloat16"
    else:
        data_type = "float16"
    try: # load the gpus.
        #inference_llm_gui is the main llm, that answers incoming prompts.
        inference_llm_gui = LLM(model=model_path, dtype=data_type, tokenizer_mode="mistral", max_model_len=16384, max_num_batched_tokens=16384,
                        enable_lora=True, max_lora_rank=16, enforce_eager=False, gpu_memory_utilization=0.6)
        logger.debug("loaded inference_llm_gui")
        logger.debug(
            "GPU memory free: %.2f GB",
            torch.cuda.mem_get_info()[0] / 1e9,
        )
        #swap_llm is only used for the warmup-call, so it has less vram etc.
        swap_llm = LLM(model=model_path, dtype=data_type, tokenizer_mode="mistral", max_model_len=64, max_num_batched_tokens=64,
                        enable_lora=True, max_lora_rank=16, enforce_eager=False, gpu_memory_utilization=0.3)
        logger.debug("loaded swap_llm")
        logger.debug(
            "GPU memory free: %.2f GB",
            torch.cuda.mem_get_info()[0] / 1e9,
        )
    except Exception as e:
        logger.exception("Failed to init vLLM: %s", e)
        raise
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.pad_token if tokenizer.pad_token is not None else tokenizer.eos_token
    uvicorn.run(app, host="0.0.0.0", port=owner_port)


if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("spawn", force=True)
    main()
