from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

model_name = "mistralai/Mistral-7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
response = generator("What is AI?", max_length=100)
print(response)