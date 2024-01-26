# import torch
# import torch.optim
# from transformers import AutoTokenizer, AutoModelForQuestionAnswering
# import pickle
# import os

# # Define the filename for storing the model_memory data
# model_memory_file = "model_memory.pkl"


# # Load the pre-trained model and tokenizer
# model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# # Check if the model_memory file exists, otherwise create an empty model_memory
# if not os.path.exists(model_memory_file):
#     with open(model_memory_file, "wb") as file:
#         model_memory = {}
#         pickle.dump(model_memory, file)

# # Load the model_memory data from file
# with open(model_memory_file, "rb") as file:
#     model_memory = pickle.load(file)

# # Function to perform medical diagnosis
# def medical_diagnosis(question, context):
#     # Tokenize the question and context
#     inputs = tokenizer.encode_plus(question, context, return_tensors="pt", padding="max_length", truncation=True)

#     # Perform question answering
#     with torch.no_grad():
#         start_scores, end_scores = model(**inputs).values()

#     # Find the start and end positions of the answer
#     start_index = torch.argmax(start_scores)
#     end_index = torch.argmax(end_scores) + 1

#     # Get the answer from the context
#     answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index:end_index]))

#     return answer

# # Run the program until "STOP" is entered
# while True:
#     # Prompt user for patient data
#     patient_symptoms = input("Enter patient symptoms (or 'STOP' to quit): ")

#     # Check if user wants to stop the program
#     if patient_symptoms.upper() == "STOP":
#         print("Program stopped.")
#         break

#     patient_history = input("Enter patient medical history: ")

#     # Construct the context with patient information
#     context = f"The patient presents with {patient_symptoms}. {patient_history}."

#     # Example usage
#     question = "What disease does the patient have?"

#     # Get the initial diagnosis
#     if (question, context) in model_memory:
#         # Use stored diagnosis
#         initial_diagnosis = model_memory[(question, context)]
#     else:
#         # Use BERT model to generate diagnosis
#         initial_diagnosis = medical_diagnosis(question, context)
#         model_memory[(question, context)] = (patient_symptoms, patient_history, initial_diagnosis)

#     print("Initial Diagnosis:", initial_diagnosis)

#     # Prompt user for feedback
#     correct_diagnosis = input("Did the model provide the correct diagnosis? (yes/no): ")

#     if correct_diagnosis.lower() == "no":
#         # Prompt user for the correct diagnosis
#         corrected_diagnosis = input("Enter the correct diagnosis: ")

#         # Store corrected diagnosis in model memory
#         model_memory[(question, context)] = (patient_symptoms, patient_history, corrected_diagnosis)

#     # Re-run the diagnosis with the updated model (if available)
#     if (question, context) in model_memory:
#         # Use stored diagnosis
#         updated_diagnosis = model_memory[(question, context)][2]
#     else:
#         # Use BERT model to generate diagnosis
#         updated_diagnosis = medical_diagnosis(question, context)
#         model_memory[(question, context)] = (patient_symptoms, patient_history, updated_diagnosis)

#     print("Updated Diagnosis:", updated_diagnosis)

# # Save the model_memory data to file
# with open(model_memory_file, "wb") as file:
#     pickle.dump(model_memory, file)




# import os

# # Define the filename for storing the model memory data
# model_memory_file = "model_memory.pkl"

# # Check if the model memory file exists
# if os.path.exists(model_memory_file):
#     # Remove the model memory file
#     os.remove(model_memory_file)
#     print("Model memory file cleared.")
# else:
#     print("Model memory file does not exist.")


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
