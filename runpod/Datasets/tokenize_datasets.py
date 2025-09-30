import csv
from transformers import AutoTokenizer
from datasets import Dataset

model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
files = ['AntonsFragen.csv', 'deutschland.csv', 'fucking_racist.csv', 'hack_it.csv', 'hdr.csv', 'hp.csv',
         'integration.csv', 'krieg.csv', 'männer.csv', 'politik.csv', 'wfragen.csv']

def tokenize_dataset(questions):
    prompts = [f"<s>[INST] {q} [/INST]" for q in questions["text"]]
    tokenized = tokenizer(
        prompts,
        truncation=True,
        padding="longest"
    )
    # Add labels for language modeling
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

for file in files:
    questions = []
    with open(file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            for cell in row:
                questions.append({"text": cell})
    dataset = Dataset.from_list(questions)

    tokenized_dataset = dataset.map(tokenize_dataset, batched=True, remove_columns=["text"])
    out_dir = file.removesuffix(".csv") + "_tokenized"
    tokenized_dataset.save_to_disk(out_dir)
