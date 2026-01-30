# Watch me kill my language
## Introduction
Watch me kill my language is an art project that makes the phenomenon of "model collapse"[1] experienceable in real-time.

For this exposition, users can interact with an LLM for 20-30 minutes, accompanied by music and visuals. The prompts that are sent to the model will be used for a finetuning loop, that leads the model to its collapse.
The model is finetuned over and over again, with a decreasing learning-rate, on the prompts and answers that were previously generated. This iterative loop must take place quickly (we aim to perform more than 5 iterations in 20 minutes), therefore we use LoRA.
This repository contains the code for the feedback finetuning loop. 
## Installation
In requirements.txt, we have defined minimum compatible versions of the required libraries.
Create a virtual environment with [venv](https://docs.python.org/3/library/venv.html) and activate it.
```py
python -m venv env
source env/bin/activate
```
Clone this repository and install dependencies.
```shell
git clone https://github.com/MartinRuz/WMKML.git
cd WMKML
pip install -r requirements.txt
```
## Running the code 
For the exposition, we used 3x A100 GPUs that were rented from [runpod](https://www.runpod.io/). Especially one script, owner.py, requires at least 32GB of VRAM. If you have less available compute, you can try selecting a different datatype such as int8/int4.

There is a lot of communication between different ports in this project. The owner script and the inference_owner script both expose api-services, at port 8000 and 8011 respectively, as specified in config.yaml Furthermore, there is communication with an osc-server, for which you can specify a port and an ip-address. If you do not have this, you can use the provided GUI, however please note that it can lead to unintended behavior, as described in the file gui.py

Before running the code for the first time, it is recommended to download the model. We used [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
```py
python download_model.py
```
If you use a different model, specify its path in config.yaml
Furthermore, it is suggested to use some additional datasets. They should be created in Datasets/ and contain a csv-file of prompts. Then, specify your datasets in tokenize_datasets.py in line 9, files = [...] and tokenize the datasets.
```py
python Datasets/tokenize_datasets.py
```
Before running the code, it is recommended to check config.yaml and verify that all fields have their intended value. Especially hf_token, available_datasets, osc_port, osc_ip, dataset_array do not currently have a value, so you should provide them.
To run the code, we provide a master script, that can be executed via 
```shell
# Make the script executable, this is needed only once.
chmod +x master_script.sh
bash master_script.sh
```
Alternatively, you can run each script individually, by following the workflow in the master script.
## Used technology
This project uses many awesome python-libraries, big thanks to

* PyTorch

* Transformers

* VLLM

* pythonosc
and many others!
## Disclaimer
This project is partly documented in German (since it's sponsored by a German ministry [Neue Künste Ruhr](https://neuekuensteruhr.de/events/watch-me-kill-my-language-7).


The finetuning of parameters was done by Anton Donle.

The visual design and project management was done by Heinrich Lenz.

The visual implementation was done by Hamidreza Ghasemi.

The code in this repo was written by [Martin Ruzicka](https://github.com/MartinRuz/).


While some effort was taken to make this repo self-contained and accessible, we are aware that questions and issues may arise. If you are interested in this project, feel free to [contact me](mailto:martin.ruzicka@gmx.de) 
## References
[1] 
@article{shumailov2023curse,
  title={The curse of recursion: Training on generated data makes models forget},
  author={Shumailov, Ilia and Shumaylov, Zakhar and Zhao, Yiren and Gal, Yarin and Papernot, Nicolas and Anderson, Ross},
  journal={arXiv preprint arXiv:2305.17493}, 
  year={2023}
}
