from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt
import uvicorn
import os
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

MODEL_PATH = r"C:\Users\San_D\Desktop\Final Year Project\Project API\KVASIR & RAG STREAMLIT\resent50_base_model_wt.h5"


# Load the model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

# Class mapping for binary classification
CLASS_MAPPING = {
    'ADI': 'Cancer Absent',
    'BACK': 'Cancer Absent',
    'DEB': 'Cancer Present',
    'LYM': 'Cancer Present',
    'MUC': 'Cancer Present',
    'MUS': 'Cancer Present',
    'NORM': 'Cancer Absent',
    'STR': 'Cancer Present',
    'TUM': 'Cancer Present'
}
CLASS_INDICES = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']

# Function to preprocess image
def preprocess_image(image: Image.Image) -> tuple:
    try:
        start_time = time.time()
        # Handle different image modes
        if image.mode != 'RGB':
            if image.mode in ('L', 'LA', 'P'):
                image = image.convert('RGB')
            elif image.mode == 'RGBA':
                image = image.convert('RGB')
            else:
                raise ValueError(f"Unsupported image mode: {image.mode}")
        
        # Resize to 224x224
        image = image.resize((224, 224))
        # Convert to array and normalize
        image_array = np.array(image) / 255.0
        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)
        logger.info(f"Image preprocessing took {time.time() - start_time:.2f} seconds")
        return image_array, np.array(image)  # Return both for prediction and LIME
    except Exception as e:
        raise ValueError(f"Image preprocessing failed: {str(e)}")

# Function to map 9-class probabilities to binary classes
def map_to_binary(prediction: np.ndarray) -> tuple:
    if model is None:
        return None, "Model not loaded"
    
    probs = prediction[0]
    cancer_present_prob = 0.0
    cancer_absent_prob = 0.0
    
    for idx, class_name in enumerate(CLASS_INDICES):
        if CLASS_MAPPING[class_name] == 'Cancer Present':
            cancer_present_prob += probs[idx]
        else:
            cancer_absent_prob += probs[idx]
    
    if cancer_present_prob > cancer_absent_prob:
        return "Cancer Present", float(cancer_present_prob)
    else:
        return "Cancer Absent", float(cancer_absent_prob)

# LIME explanation function
def get_lime_explanation(image: np.ndarray) -> str:
    start_time = time.time()
    def predict_fn(images):
        images = images / 255.0
        return model.predict(images, verbose=0)  # Suppress TensorFlow logs
    
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image.astype(np.uint8),
        predict_fn,
        top_labels=9,
        hide_color=0,
        num_samples=300,  # Reduced for faster processing
        num_features=3   # Fewer features for simpler explanation
    )
    
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=3,
        hide_rest=False
    )
    
    explained_image = mark_boundaries(temp / 255.0, mask)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(explained_image)
    plt.axis('off')
    
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    plt.close()
    buffer.seek(0)
    
    encoded_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    logger.info(f"LIME explanation took {time.time() - start_time:.2f} seconds")
    return encoded_image

@app.post("/predict_crc")
async def predict_crc(file: UploadFile = File(...)):
    try:
        start_time = time.time()
        # Validate file extension
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            return JSONResponse(status_code=400, content={"error": f"Unsupported file format: {file_ext}. Use JPG, PNG, or TIFF."})
        
        # Check file size (10MB limit)
        contents = await file.read()
        if len(contents) > 10 * 1024 * 1024:
            return JSONResponse(status_code=400, content={"error": "File too large. Maximum size is 10MB."})
        
        # Read and process the image
        logger.info("Loading image...")
        try:
            image = Image.open(io.BytesIO(contents))
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": f"Failed to open image: {str(e)}"})
        
        # Preprocess image
        logger.info("Preprocessing image...")
        try:
            image_array, image_for_lime = preprocess_image(image)
        except Exception as e:
            return JSONResponse(status_code=400, content={"error": str(e)})
        
        # Get model prediction
        logger.info("Running model prediction...")
        if model is None:
            return JSONResponse(status_code=500, content={"error": "Model not loaded"})
        prediction = model.predict(image_array, verbose=0)
        
        # Map to binary classification
        binary_class, confidence = map_to_binary(prediction)
        if binary_class is None:
            return JSONResponse(status_code=500, content={"error": confidence})
        
        # Get LIME explanation
        logger.info("Generating LIME explanation...")
        lime_image = get_lime_explanation(image_for_lime)
        
        logger.info(f"Total processing took {time.time() - start_time:.2f} seconds")
        # Return response
        return {
            "prediction": binary_class,
            "confidence": confidence,
            "explanation_image": f"data:image/png;base64,{lime_image}"
        }
    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        return JSONResponse(status_code=500, content={"error": f"Server error: {str(e)}"})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7000)