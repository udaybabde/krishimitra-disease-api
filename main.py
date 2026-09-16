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

    return {
        "disease": predicted_class,
        "confidence": round(confidence, 2)
    }
