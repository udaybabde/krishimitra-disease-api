from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import io

app = FastAPI()

model = tf.keras.models.load_model("disease_model.keras")
with open("class_names.json") as f:
    class_names = json.load(f)

fertilizer_data = {
    "Tomato_Bacterial_spot": {"treatment": "Copper Oxychloride 50% WP", "dosage": "3g/liter water, spray every 10 days"},
    "Tomato_Early_blight": {"treatment": "Mancozeb 75% WP", "dosage": "2.5g/liter water, every 10-15 days"},
    "Tomato_Late_blight": {"treatment": "Metalaxyl + Mancozeb (Ridomil Gold)", "dosage": "2g/liter water, every 7-10 days"},
    "Tomato_Leaf_Mold": {"treatment": "Chlorothalonil 75% WP", "dosage": "2g/liter water + improve ventilation"},
    "Tomato_Septoria_leaf_spot": {"treatment": "Mancozeb 75% WP", "dosage": "2.5g/liter water"},
    "Tomato_Spider_mites_Two_spotted_spider_mite": {"treatment": "Spiromesifen 22.9% SC", "dosage": "1ml/liter water"},
    "Tomato__Target_Spot": {"treatment": "Azoxystrobin 23% SC", "dosage": "1ml/liter water"},
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {"treatment": "Imidacloprid 17.8% SL (control whitefly vector)", "dosage": "0.3ml/liter water + remove infected plants"},
    "Tomato__Tomato_mosaic_virus": {"treatment": "No chemical cure available", "dosage": "Remove infected plants, disinfect tools, control aphids"},
    "Potato___Early_blight": {"treatment": "Mancozeb 75% WP", "dosage": "2.5g/liter water"},
    "Potato___Late_blight": {"treatment": "Metalaxyl + Mancozeb", "dosage": "2g/liter water, every 7 days in humid weather"},
    "Potato___healthy": {"treatment": "No treatment needed", "dosage": "Plant is healthy"},
    "Tomato_healthy": {"treatment": "No treatment needed", "dosage": "Plant is healthy"},
}

@app.get("/")
def home():
    return {"message": "KrishiMitra Disease Detection API is running"}

@app.post("/detect-disease")
async def detect_disease(file: UploadFile = File(...)):
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    img = img.resize((160, 160))
    img_array = np.expand_dims(np.array(img), axis=0)

    pred = model.predict(img_array)
    predicted_class = class_names[np.argmax(pred)]
    confidence = float(np.max(pred) * 100)

    remedy = fertilizer_data.get(predicted_class, {})

    return {
        "disease": predicted_class,
        "confidence": round(confidence, 2),
        "treatment": remedy.get("treatment", "Data not available"),
        "dosage": remedy.get("dosage", "Data not available")
    }
