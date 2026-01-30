# Watch me kill my language
## Introduction
Watch me kill my language is an art project that makes the phenomenon of "model collapse"[1] experienceable in real-time.
For this exposition, users can interact with an LLM for 20-30 minutes, accompanied by music and visuals. The prompts that are sent to the model will be used for a finetuning loop, that leads the model to its collapse.
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
For the exposition, we used 3x A100 GPUs that were rented from [runpod](https://www.runpod.io/). Especially one script, owner.py, requires about 32GB of VRAM. If you have less available compute, you can try selecting a different datatype such as int8/int4.

Before running the code for the first time, it is recommended to download the model. We used [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
```py
python download_model.py
```
If you use a different model, specify its path in config.yaml
Furthermore, it is suggested to use some additional datasets. They should be created in Datasets/ and contain a csv-file of prompts. Then, specify your datasets in tokenize_datasets.py in line 9, files = [...] and tokenize the datasets.
```py
python tokenize_datasets.py
```
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
While some effort was taken to make this repo self-contained and accessible, we are aware that there are always some issues. Feel free to [contact me](mailto:martin.ruzicka@gmx.de) 
## References
[1] 
@article{shumailov2023curse,
  title={The curse of recursion: Training on generated data makes models forget},
  author={Shumailov, Ilia and Shumaylov, Zakhar and Zhao, Yiren and Gal, Yarin and Papernot, Nicolas and Anderson, Ross},
  journal={arXiv preprint arXiv:2305.17493}, 
  year={2023}
}
