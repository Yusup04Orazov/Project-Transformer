from transformers import pipeline

# Load the text classification pipeline
classifier = pipeline("text-classification", model="facebook/bart-large-mnli")

# Prompt user for patient information
patient_text = input("Enter patient information: ")

# Perform text classification
results = classifier(patient_text)

# Get the predicted label and score
predicted_label = results[0]["label"]
score = results[0]["score"]

# Print the prediction
print("Prediction:", predicted_label)
print("Confidence score:", score)
