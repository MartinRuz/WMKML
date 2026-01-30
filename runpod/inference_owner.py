from utility import *

if available_gpus > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print("using gpu 1")
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("using gpu 0")

import torch
import json
import time
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from fastapi import FastAPI, Request, BackgroundTasks
from pathlib import Path
import uvicorn
import multiprocessing as mp
from starlette.concurrency import run_in_threadpool
import logging

print("visible devices:", torch.cuda.device_count())
print("current device:", torch.cuda.current_device())
print("gpu name:", torch.cuda.get_device_name())

adapter_path = ""
logging.basicConfig(level=logging_level, format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s')
logger = logging.getLogger("inference_owner")

inference_llm = None
tokenizer = None

app = FastAPI()


"""
Utility function to ensure that a file exists, to be used before a writing operation.
"""
def ensure_dir_for_file(path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)


"""
Utility function to safely append a dict to a jsonl-file.
path is the path of the file where you want to write.
obj contains the dict of data that will be written to the file.
"""
def append_jsonl(path: str, obj: dict):
    ensure_dir_for_file(path)
    # append + flush
    if not os.path.exists(path):
        open(path, 'w').close()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


"""
For compatibility reasons, this function is named change_language, however besides changing the language, this also resets the system!
The language (aka the system_prompt which defines the language for the llm) will be set according to the req-json
This resets the global parameters that is changed during a run, adapter_path, to its initial value. 
If you should ever add more changing parameters, reset them here.
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

    global adapter_path # reset the path to its initial value
    adapter_path = ""
    open(restart_flag, "w").close() # this flag notifies the trainer to start training.
    logger.debug("Owner has set language and restarted the system.")
    return {"status": "success"}


"""
This function is assumed to be called during the first iteration. Its purpose is to create one datapoint in the live-dataset.
req contains the prompt
"""
@app.post("/generate_initial")
async def initial_inference_and_store(req: GenerateRequest, background_tasks: BackgroundTasks):
    logger.info(f"initial inference received {req.prompt}")
    params = SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p, n=num_generated_responses, frequency_penalty=frequency_penalty)
    messages = [{"role": "system", "content": system_prompt}, # create the object that gets passed to the llm.
        {"role": "user", "content": req.prompt},]

    async def _bg_generate_and_save(messages: str, params, req_obj: GenerateRequest): # generate the response in a background task.
        start_time = time.time()
        try:
            outputs = inference_llm.chat(messages=[messages], sampling_params=params)
        except Exception as e:
            logger.exception("vLLM generate failed: %s", e)
            return

        responses = [out.text for out in outputs[0].outputs] # extract the textual response
        out_path = os.path.join(dataset_filepath, "live", str(max_tokens) + "_tokens", "conversations.jsonl")
        for resp in responses:
            append_jsonl(out_path, {"prompt": req_obj.prompt, "response": resp, "iteration": req_obj.iteration}) # save the response
        logger.info("initial_inference took " + str(time.time() - start_time))

    background_tasks.add_task(_bg_generate_and_save, messages, params, req)

    return {"status": "received and scheduled"}


"""
Create multiple response to the live-question, so we have more datapoints for our training loop.
req contains the prompt
"""
@app.post("/generate_inference")
async def multi_inference_and_store(req: GenerateRequest, background_tasks: BackgroundTasks):
    logger.info(f"inference received multi_inference with {req.prompt}")
    params = SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p, n=num_generated_responses, frequency_penalty=frequency_penalty)
    messages = [{"role": "system", "content": system_prompt}, # create the object that gets passed to the llm.
        {"role": "user", "content": req.prompt},]

    maybe_lora = None
    if adapter_path != "": # if we have already trained an adapter, we use the LoRARequest object (aka the adapter) with our prompt
        adapter_name = f"adapter_iter_{req.iteration}"
        maybe_lora=LoRARequest(adapter_name, 100 + req.iteration, adapter_path)

    async def _bg_generate_and_save(messages: str, params, req_obj: GenerateRequest):
        start_time = time.time()
        try:
            outputs = inference_llm.chat(messages=[messages], sampling_params=params, lora_request=maybe_lora)
        except Exception as e:
            logger.exception("vLLM generate failed: %s", e)
            return

        responses = [out.text for out in outputs[0].outputs] # extract the text-answer
        out_path = os.path.join(dataset_filepath, "live", str(max_tokens) + "_tokens", "conversations.jsonl")
        for resp in responses:
            append_jsonl(out_path, {"prompt": req_obj.prompt, "response": resp, "iteration": req_obj.iteration}) # save the response
        logger.info("multi_inference took " + str(time.time() - start_time))

    background_tasks.add_task(_bg_generate_and_save, messages, params, req)

    return {"status": "received and scheduled"}


"""
This function creates the training dataset. We need to answer the prompts of the backup-datasets that will be used in the next iterations with the decaying model.
req contains the current iteration and the datasets that should be answered.
"""
@app.post("/infer_next_iters")
async def perform_next_iterations(req: InferenceRequest, background_tasks: BackgroundTasks):
    logger.info("perform_next_iteration started")
    data = req.data
    iter = req.iteration
    params = SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p, n=1, frequency_penalty=frequency_penalty)
    async def _bg_perform_next():
        start_time = time.time()
        set_of_data = {item for sublist in data for item in sublist} # the set of datasets that we should answer
        maybe_lora = None
        if adapter_path != "":
            adapter_name = f"adapter_iter_{iter}"
            maybe_lora=LoRARequest(adapter_name, 100 + req.iteration, adapter_path)
        logger.debug("perform next iteration is answering: %s", set_of_data)
        for dataset in set_of_data:
            results = []
            base_plain = os.path.join(dataset_filepath, dataset, f"{dataset}_plain")
            if verification_mode and not validate_plain_dataset_structure(base_plain): # raise an error if there is a problem with the data-structure
                logger.error("Structural validation failed for dataset %s", base_plain)
                continue
            try:
                question_dataset = load_from_disk(base_plain) # load the plain dataset, aka questions only
            except Exception as e:
                logger.exception("Could not load dataset %s: %s", base_plain, e)
                continue
            questions = [x["prompt"] for x in question_dataset]
            messages = [[ # create the object that gets passed to the llm
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ] for prompt in questions]

            try:
                outputs = inference_llm.chat(messages=messages, sampling_params=params, lora_request=maybe_lora)
            except Exception as e:
                logger.exception("Batch generate failed for dataset %s: %s", dataset, e)
                continue
            for q, out in zip(questions, outputs): # create a dict with the proper fields so we can process it in the training script
                for generated in out.outputs:
                    results.append({
                        "prompt": q,
                        "response": generated.text,
                        "iteration": iter,
                    })
            out_dir = os.path.join(dataset_filepath, dataset, str(max_tokens) + "_tokens", "iteration", str(iter))
            os.makedirs(out_dir, exist_ok=True)
            answered_dataset = Dataset.from_list(results)
            answered_dataset.save_to_disk(out_dir) # save the answered dataset to the disk.
            logger.debug("saved dataset %s to location %s", dataset, out_dir)

        # Live dataset requires separate handling because of other structure due to live evolving dataset
        live_path = os.path.join(dataset_filepath, "live", "conversations.jsonl")
        live_questions = []
        live_results = []
        if os.path.exists(live_path):
            with open(live_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        # ensure expected keys
                        if "prompt" in entry:
                            live_questions.append({"prompt": entry["prompt"], "iteration": iter})
                    except Exception as e:
                        logger.exception("Skipping bad line in live file: %s", e)

        if live_questions:
            messages = [[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
                ] for item in live_questions for prompt in [item["prompt"]]]
            try:
                outputs = inference_llm.chat(messages=messages, sampling_params=params, lora_request=maybe_lora)
            except Exception as e:
                logger.exception("Batch generate failed for live questions: %s", e)
                outputs = []

            for q, out in zip(live_questions, outputs):
                for generated in out.outputs:
                    live_results.append({
                        "prompt": q["prompt"],
                        "response": generated.text,
                        "iteration": q["iteration"]
                    })
            out_dir = os.path.join(dataset_filepath, "live", str(max_tokens) + "_tokens", "iteration", str(iter))
            os.makedirs(out_dir, exist_ok=True)
            answered_dataset = Dataset.from_list(live_results)
            answered_dataset.save_to_disk(out_dir)
            logger.debug("Saved tokenized live dataset to %s", out_dir)

        logger.info(f"perform_next_iteration ended after {time.time()-start_time} seconds")
    background_tasks.add_task(_bg_perform_next)
    return {"status": "received and scheduled"}


"""
This function loads an incoming adapter. Since this script only performs background tasks that are not seen by the user, we do not need to ensure speed, and thus do not need to introduce a swap_llm.
req contains the new path.
"""
@app.post("/reload")
def reload_adapter(req: ReloadRequest):
    logger.info("Inference received reload")
    global adapter_path
    adapter_path = req.path
    return {"status": f"reloaded adapter from {req.path}"}


"""
This method generates a response to the persistent prompt, for debugging purposes.
req contains the prompt."""
@app.post("/generate_persistent")
async def generate_persistent(req:GenerateRequest, background_tasks: BackgroundTasks):
    logger.info("Inference received persistent prompt")
    outfile = "logs/persistent_answers.txt"
    params = SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p, n=1)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": req.prompt},
    ]
    logger.debug(f"generate persistent prompt without chat template in perform_next_iteration: {messages}")
    async def _bg_persistent(): # generate the response in a background task.
        try:
            if req.iteration > 0:
                adapter_name = f"adapter_iter_{req.iteration}"
                lora_request = LoRARequest(adapter_name, 100 + req.iteration, adapter_path)
                logger.debug(f"Calling generate in generate_persistent if with {messages}")
                outputs = inference_llm.chat(messages=[messages], sampling_params=params, lora_request=lora_request)
                output = outputs[0].outputs[0].text
            else:
                logger.debug(f"Calling generate in generate_persistent else with {messages}")
                outputs = inference_llm.chat(messages=[messages], sampling_params=params)
                output = outputs[0].outputs[0].text
            print(f"In iteration {req.iteration} our model answered the persistent prompt {req.prompt} with the following answer:\n{output}")
            with open(outfile, 'a') as f:
                f.write(f"In iteration {req.iteration} our model answered the persistent prompt {req.prompt} with the following answer:\n{output}\n\n")
                f.close()
        except Exception as e:
            logger.exception("generate_persistent failed: %s", e)

    background_tasks.add_task(_bg_persistent)
    return {"status": "received and scheduled"}


"""
Here we perform some initial loading. This is much faster (ca 5 minutes) than the owner's main, since it only loads one llm.
"""
def main():
    global inference_llm, tokenizer
    if torch.cuda.is_bf16_supported():
        data_type = "bfloat16"
    else:
        data_type = "float16"
    try: # load the llm that will answer prompts in the background.
        inference_llm = LLM(model=model_path, dtype=data_type, tokenizer_mode="mistral", max_model_len=16384, max_num_batched_tokens=16384,
                        enable_lora=True, max_lora_rank=16, enforce_eager=True)
    except Exception as e:
        print("Failed to init vLLM:", e)
        raise
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.pad_token if tokenizer.pad_token is not None else tokenizer.eos_token
    uvicorn.run(app, host="0.0.0.0", port=inference_port)



if __name__ == "__main__":
    mp.freeze_support()
    mp.set_start_method("spawn", force=True)
    main()
