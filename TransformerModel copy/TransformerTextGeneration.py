# from transformers import pipeline

# generator = pipeline("text-generation", model = "distilgpt2")

# res = generator(
#     "in this course, we will teach you how to", 
#     max_length=30,
#     num_return_sequences=2,
# )
# print(res)

from transformers import pipeline
import textwrap

generator = pipeline("text-generation", model="distilgpt2")

res = generator(
    "I want to kill myself  ",
    max_length=30,
    num_return_sequences=2,
)

# Extract the generated text
generated_text = [output['generated_text'] for output in res]

# Format the generated text in a big format
big_format_text = textwrap.fill(' '.join(generated_text), width=100)

# Print the big format text
print(big_format_text)
