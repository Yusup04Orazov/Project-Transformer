from transformers import pipeline

classifier = pipeline("sentiment-analysis")

res = classifier("I want to go outside")

print(res)