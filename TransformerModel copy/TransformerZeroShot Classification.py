from transformers import pipeline
classifier = pipeline("zero-shot-classification" )
res = classifier(
    "Everything is impossible until it is done by somebody who had exact potential as you did.", 
    candidate_labels= ["Opportunity", "Taking Action", "Give Up"],)  
print(res)