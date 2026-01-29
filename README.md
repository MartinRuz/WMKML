## Introduction
Watch me kill my language is an art project that makes the phenomenon of "model collapse" \footnote{@article{shumailov2023curse,
  title={The curse of recursion: Training on generated data makes models forget},
  author={Shumailov, Ilia and Shumaylov, Zakhar and Zhao, Yiren and Gal, Yarin and Papernot, Nicolas and Anderson, Ross},
  journal={arXiv preprint arXiv:2305.17493},
  year={2023}
}} experienceable in real-time.
For this exposition, users can interact with an LLM for 20-30 minutes, accompanied by music and visuals. The prompts that are sent to the model will be used for a finetuning loop, that leads the model into insanity, aka the model collapse.
This repo contains the code for this feedback finetuning loop. 
## Prerequisites and Running the code
This project was run on 3x A100 GPUs. 
Create a virtual environment, and install the required libraries via
pip install -r requirements.txt
Then, either make the shell-script executable via 
chmod +x master_script.sh
and then run
bash master_script.sh
Or run every script individually, via python owner.py etc, after activating the environment.
## Used technology
This project uses many awesome python-libraries, big thanks to
PyTorch
Transformers
VLLM
pythonosc
and many others!
For this project, we rented GPU-time on runpod.io
