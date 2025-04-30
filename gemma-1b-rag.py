import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, Gemma3ForCausalLM
import torch


def ask_question(question, top_k=3, max_new_tokens=100):
    q_embedding = embedder.encode([question], convert_to_numpy=True)
    _, indices = index.search(q_embedding, top_k)
    retrieved = "\n".join([docs[i] for i in indices[0]])
    prompt = f"Kontext:\n{retrieved}\n\nFrage: {question}\nAntwort:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
    output = model.generate(**inputs, max_new_tokens=max_new_tokens)
    return tokenizer.decode(output[0], skip_special_tokens=True)



if __name__ == '__main__':
    docs = []
    with open("2024_rag_docs.jsonl", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            docs.append(entry["text"])

    embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    embeddings = embedder.encode(docs, convert_to_numpy=True, show_progress_bar=True)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    model_id = "google/gemma-3-1b-it"

    model = Gemma3ForCausalLM.from_pretrained(model_id, torch_dtype="auto", attn_implementation="eager").to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    response = ask_question("Was geschah am 1.Januar?")
    print(response)
    response = ask_question("Wer ist Friedrich Merz?")
    print(response)