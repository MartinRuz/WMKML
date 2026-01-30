import csv
from transformers import AutoTokenizer
from datasets import Dataset

model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
files = ['']

def tokenize_dataset(questions):
    texts = []

    for q,r in zip(questions["prompt"], questions["response"]):
        messages = [
            {"role": "user", "content": q},
            {"role": "assistant", "content": r}
        ]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        texts.append(formatted)

    tokenized = tokenizer(
        texts,
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

for file in files:
    questions = []
    with open(file.removesuffix(".csv") + "/" + file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            for cell in row:
                questions.append({"prompt": cell}) #"text":
    dataset = Dataset.from_list(questions)

    out_dir = file.removesuffix(".csv") + "/" + file.removesuffix(".csv") + "_plain"
    dataset.save_to_disk(out_dir)
