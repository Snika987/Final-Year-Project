import uvicorn
from fastapi import FastAPI, UploadFile, File
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
import torch
import io

app = FastAPI()

# Load model and feature extractor
model_name = "mmuratarat/kvasir-v2-classifier"
model = AutoModelForImageClassification.from_pretrained(model_name)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

# Class ID to Label mapping
id2label = {
    '0': 'dyed-lifted-polyps',
    '1': 'dyed-resection-margins',
    '2': 'esophagitis',
    '3': 'normal-cecum',
    '4': 'normal-pylorus',
    '5': 'normal-z-line',
    '6': 'polyps',
    '7': 'ulcerative-colitis'
}

# Mapping for polyp presence
polyp_mapping = {
    'dyed-lifted-polyps': "Polyp Present",
    'dyed-resection-margins': "Polyp Present",
    'polyps': "Polyp Present",
    'ulcerative-colitis': "Polyp Absent",  
    'esophagitis': "Polyp Absent",
    'normal-cecum': "Polyp Absent",
    'normal-pylorus': "Polyp Absent",
    'normal-z-line': "Polyp Absent"
}

@app.get("/")
def home():
    return {"message": "Welcome to the Kvasir Image Classification API! Use /predict/ to classify an image."}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read image file
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Process the image
    inputs = feature_extractor(image, return_tensors="pt")
    logits = model(**inputs).logits

    # Get predicted class
    predicted_label = logits.argmax(-1).item()
    predicted_class = id2label[str(predicted_label)]
    
    # Determine polyp presence
    polyp_status = polyp_mapping[predicted_class]

    return {
        "predicted_class": predicted_class,
        "polyp_status": polyp_status
    }

import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)

