from transformers import pipeline
print(pipeline('sentiment-analysis')("huggingface is the best"))