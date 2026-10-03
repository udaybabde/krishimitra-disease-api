from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import json
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = tf.keras.models.load_model("disease_model.keras")
with open("class_names.json") as f:
    class_names = json.load(f)

fertilizer_data = {
    # ---- Apple ----
    "Apple___Apple_scab": {"treatment": "Captan 50% WP", "dosage": "2g/liter water, every 10-14 days"},
    "Apple___Black_rot": {"treatment": "Mancozeb 75% WP", "dosage": "2.5g/liter water, every 10 days"},
    "Apple___Cedar_apple_rust": {"treatment": "Myclobutanil 10% WP", "dosage": "1g/liter water, every 14 days"},
    "Apple___healthy": {"treatment": "No treatment needed", "dosage": "Plant is healthy"},

    # ---- Blueberry ----
    "Blueberry___healthy": {"treatment": "No treatment needed", "dosage": "Plant is healthy"},

    # ---- Cherry ----
    "Cherry_(including_sour)___Powdery_mildew": {"treatment": "Sulfur 80% WP", "dosage": "2.5g/liter water, every 7-10 days"},
    "Cherry_(including_sour)___healthy": {"treatment": "No treatment needed", "dosage": "Plant is healthy"},

    # ---- Corn / Maize ----
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {"treatment": "Azoxystrobin 23% SC", "dosage": "1ml/liter water"},
    "Corn_(maize)___Common_rust_": {"treatment": "Mancozeb 75% WP", "dosage": "2.5g/liter water, every 10 days"},
    "Corn_(maize)___Northern_Leaf_Blight": {"treatment": "Propiconazole 25% EC", "dosage": "1ml/liter water"},
    "Corn_(maize)___healthy": {"treatment": "No treatment needed", "dosage": "Plant is healthy"},

    # ---- Cotton ----
    "Cotton___diseased_cotton_leaf": {"treatment": "Copper Oxychloride 50% WP", "dosage": "3g/liter water, every 10 days"},
    "Cotton___diseased_cotton_plant": {"treatment": "Imidacloprid 17.8% SL (for sucking pests) + Copper Oxychloride", "dosage": "0.3ml/liter + 3g/liter water"},
    "Cotton___fresh_cotton_leaf": {"treatment": "No treatment needed", "dosage": "Plant is healthy"},
    "Cotton___fresh_cotton_plant": {"treatment": "No treatment needed", "dosage": "Plant is healthy"},

    # ---- Grape ----
    "Grape___Black_rot": {"treatment": "Mancozeb 75% WP", "dosage": "2.5g/liter water, every 10 days"},
    "Grape___Esca_(Black_Measles)": {"treatment": "No direct chemical cure", "dosage": "Remove and destroy infected vines"},
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {"treatment": "Copper Oxychloride 50% WP", "dosage": "3g/liter water"},
    "Grape___healthy": {"treatment": "No treatment needed", "dosage": "Plant is healthy"},

    # ---- Mango ----
    "Mango___Anthracnose": {"treatment": "Carbendazim 50% WP", "dosage": "1g/liter water, every 10-15 days"},
    "Mango___Bacterial_Canker": {"treatment": "Copper Oxychloride 50% WP", "dosage": "3g/liter water, every 15 days"},
    "Mango___Cutting_Weevil": {"treatment": "Chlorpyrifos 20% EC", "dosage": "2ml/liter water"},
    "Mango___Die_Back": {"treatment": "Carbendazim 50% WP + pruning of affected twigs", "dosage": "1g/liter water"},
    "Mango___Gall_Midge": {"treatment": "Dimethoate 30% EC", "dosage": "2ml/liter water"},
    "Mango___Healthy": {"treatment": "No treatment needed", "dosage": "Plant is healthy"},
    "Mango___Powdery_Mildew": {"treatment": "Sulfur 80% WP", "dosage": "2.5g/liter water, every 7-10 days"},
    "Mango___Sooty_Mould": {"treatment": "Control sap-sucking insects with Imidacloprid 17.8% SL", "dosage": "0.3ml/liter water"},

    # ---- Orange ----
    "Orange___Haunglongbing_(Citrus_greening)": {"treatment": "No chemical cure; control psyllid vector with Imidacloprid", "dosage": "0.3ml/liter water; remove infected trees"},

    # ---- Peach ----
    "Peach___Bacterial_spot": {"treatment": "Copper Oxychloride 50% WP", "dosage": "3g/liter water, every 10 days"},
    "Peach___healthy": {"treatment": "No treatment needed", "dosage": "Plant is healthy"},

    # ---- Pepper (Bell) ----
    "Pepper,_bell___Bacterial_spot": {"treatment": "Copper Oxychloride 50% WP", "dosage": "3g/liter water, every 10 days"},
    "Pepper,_bell___healthy": {"treatment": "No treatment needed", "dosage": "Plant is healthy"},

    # ---- Potato ----
    "Potato___Early_blight": {"treatment": "Mancozeb 75% WP", "dosage": "2.5g/liter water"},
    "Potato___Late_blight": {"treatment": "Metalaxyl + Mancozeb", "dosage": "2g/liter water, every 7 days in humid weather"},
    "Potato___healthy": {"treatment": "No treatment needed", "dosage": "Plant is healthy"},

    # ---- Raspberry ----
    "Raspberry___healthy": {"treatment": "No treatment needed", "dosage": "Plant is healthy"},

    # ---- Soybean ----
    "Soybean___healthy": {"treatment": "No treatment needed", "dosage": "Plant is healthy"},

    # ---- Squash ----
    "Squash___Powdery_mildew": {"treatment": "Sulfur 80% WP", "dosage": "2.5g/liter water, every 7-10 days"},

    # ---- Strawberry ----
    "Strawberry___Leaf_scorch": {"treatment": "Captan 50% WP", "dosage": "2g/liter water, every 10 days"},
    "Strawberry___healthy": {"treatment": "No treatment needed", "dosage": "Plant is healthy"},

    # ---- Sugarcane ----
    "Sugarcane___Healthy": {"treatment": "No treatment needed", "dosage": "Plant is healthy"},
    "Sugarcane___Mosaic": {"treatment": "No chemical cure; control aphid vector with Imidacloprid", "dosage": "0.3ml/liter water; remove infected plants"},
    "Sugarcane___RedRot": {"treatment": "Carbendazim 50% WP (sett treatment)", "dosage": "1g/liter water before planting"},
    "Sugarcane___Rust": {"treatment": "Propiconazole 25% EC", "dosage": "1ml/liter water"},
    "Sugarcane___Yellow": {"treatment": "Balanced NPK + Imidacloprid for vector control", "dosage": "0.3ml/liter water"},

    # ---- Tomato ----
    "Tomato___Bacterial_spot": {"treatment": "Copper Oxychloride 50% WP", "dosage": "3g/liter water, every 10 days"},
    "Tomato___Early_blight": {"treatment": "Mancozeb 75% WP", "dosage": "2.5g/liter water, every 10-15 days"},
    "Tomato___Late_blight": {"treatment": "Metalaxyl + Mancozeb (Ridomil Gold)", "dosage": "2g/liter water, every 7-10 days"},
    "Tomato___Leaf_Mold": {"treatment": "Chlorothalonil 75% WP", "dosage": "2g/liter water + improve ventilation"},
    "Tomato___Septoria_leaf_spot": {"treatment": "Mancozeb 75% WP", "dosage": "2.5g/liter water"},
    "Tomato___Spider_mites Two-spotted_spider_mite": {"treatment": "Spiromesifen 22.9% SC", "dosage": "1ml/liter water"},
    "Tomato___Target_Spot": {"treatment": "Azoxystrobin 23% SC", "dosage": "1ml/liter water"},
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {"treatment": "Imidacloprid 17.8% SL (control whitefly vector)", "dosage": "0.3ml/liter water + remove infected plants"},
    "Tomato___Tomato_mosaic_virus": {"treatment": "No chemical cure available", "dosage": "Remove infected plants, disinfect tools"},
    "Tomato___healthy": {"treatment": "No treatment needed", "dosage": "Plant is healthy"},
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
    "treatment": remedy.get("treatment", "Data not available") if confidence >= 60 else "Low confidence - please upload a clearer photo",
    "dosage": remedy.get("dosage", "Data not available") if confidence >= 60 else "N/A"
}
