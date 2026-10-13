"""
FastAPI Backend for Dermatology Disease Classifier
Serves the web UI and handles prediction requests.
"""

import os
import torch
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List

from model import DermatologyClassificationModel, COLUMNS, CLASS_NAMES

# ──────────────────────── App Setup ────────────────────────
app = FastAPI(
    title="Dermatology Disease Classifier",
    description="Deep learning-based erythemato-squamous disease classification",
    version="1.0.0",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# ──────────────────────── Load Model ────────────────────────
model = DermatologyClassificationModel()
model_path = os.path.join(BASE_DIR, "model_state_dict.pth")

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    print("[OK] Model loaded successfully.")
else:
    print("[WARN] Model weights not found. Run `python model.py` first to train the model.")

model.eval()


# ──────────────────────── Schemas ────────────────────────
FEATURE_NAMES = COLUMNS[:-1]  # 34 features, without 'class'

class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    predicted_class: int
    predicted_label: str
    probabilities: dict


# ──────────────────────── Feature Metadata ────────────────────────
# Grouped feature info for the frontend form
FEATURE_GROUPS = {
    "clinical": {
        "title": "Klinik Özellikler",
        "subtitle": "Fiziki muayene bulguları",
        "icon": "stethoscope",
        "features": [
            {"name": "erythema", "label": "Erythema (Kızarıklık)", "type": "scale", "min": 0, "max": 3},
            {"name": "scaling", "label": "Scaling (Pullanma)", "type": "scale", "min": 0, "max": 3},
            {"name": "definite_borders", "label": "Definite Borders (Belirgin Sınırlar)", "type": "scale", "min": 0, "max": 3},
            {"name": "itching", "label": "Itching (Kaşıntı)", "type": "scale", "min": 0, "max": 3},
            {"name": "koebner_phenomenon", "label": "Koebner Phenomenon", "type": "scale", "min": 0, "max": 3},
            {"name": "polygonal_papules", "label": "Polygonal Papules", "type": "scale", "min": 0, "max": 3},
            {"name": "follicular_papules", "label": "Follicular Papules", "type": "scale", "min": 0, "max": 3},
            {"name": "oral_mucosal_involvement", "label": "Oral Mucosal Involvement", "type": "scale", "min": 0, "max": 3},
            {"name": "knee_and_elbow_involvement", "label": "Knee & Elbow Involvement", "type": "scale", "min": 0, "max": 3},
            {"name": "scalp_involvement", "label": "Scalp Involvement", "type": "scale", "min": 0, "max": 3},
            {"name": "family_history", "label": "Family History (Aile Öyküsü)", "type": "binary", "min": 0, "max": 1},
        ]
    },
    "histopathological": {
        "title": "Histopatolojik Özellikler",
        "subtitle": "Mikroskobik biyopsi bulguları",
        "icon": "microscope",
        "features": [
            {"name": "melanin_incontinence", "label": "Melanin Incontinence", "type": "scale", "min": 0, "max": 3},
            {"name": "eosinophils_infiltrate", "label": "Eosinophils in Infiltrate", "type": "scale", "min": 0, "max": 3},
            {"name": "PNL_infiltrate", "label": "PNL Infiltrate", "type": "scale", "min": 0, "max": 3},
            {"name": "fibrosis_papillary_dermis", "label": "Fibrosis of Papillary Dermis", "type": "scale", "min": 0, "max": 3},
            {"name": "exocytosis", "label": "Exocytosis", "type": "scale", "min": 0, "max": 3},
            {"name": "acanthosis", "label": "Acanthosis", "type": "scale", "min": 0, "max": 3},
            {"name": "hyperkeratosis", "label": "Hyperkeratosis", "type": "scale", "min": 0, "max": 3},
            {"name": "parakeratosis", "label": "Parakeratosis", "type": "scale", "min": 0, "max": 3},
            {"name": "clubbing_rete_ridges", "label": "Clubbing of Rete Ridges", "type": "scale", "min": 0, "max": 3},
            {"name": "elongation_rete_ridges", "label": "Elongation of Rete Ridges", "type": "scale", "min": 0, "max": 3},
            {"name": "thinning_suprapapillary_epidermis", "label": "Thinning of Suprapapillary Epidermis", "type": "scale", "min": 0, "max": 3},
            {"name": "spongiform_pustule", "label": "Spongiform Pustule", "type": "scale", "min": 0, "max": 3},
            {"name": "munro_microabcess", "label": "Munro Microabcess", "type": "scale", "min": 0, "max": 3},
            {"name": "focal_hypergranulosis", "label": "Focal Hypergranulosis", "type": "scale", "min": 0, "max": 3},
            {"name": "disappearance_granular_layer", "label": "Disappearance of Granular Layer", "type": "scale", "min": 0, "max": 3},
            {"name": "vacuolisation_damage_basal_layer", "label": "Vacuolisation & Damage of Basal Layer", "type": "scale", "min": 0, "max": 3},
            {"name": "spongiosis", "label": "Spongiosis", "type": "scale", "min": 0, "max": 3},
            {"name": "saw_tooth_appearance_retes", "label": "Saw-tooth Appearance of Retes", "type": "scale", "min": 0, "max": 3},
            {"name": "follicular_horn_plug", "label": "Follicular Horn Plug", "type": "scale", "min": 0, "max": 3},
            {"name": "perifollicular_parakeratosis", "label": "Perifollicular Parakeratosis", "type": "scale", "min": 0, "max": 3},
            {"name": "inflammatory_mononuclear_infiltrate", "label": "Inflammatory Mononuclear Infiltrate", "type": "scale", "min": 0, "max": 3},
            {"name": "band_like_infiltrate", "label": "Band-like Infiltrate", "type": "scale", "min": 0, "max": 3},
        ]
    },
    "demographic": {
        "title": "Demografik Bilgiler",
        "subtitle": "Hasta bilgileri",
        "icon": "user",
        "features": [
            {"name": "age", "label": "Yaş (Age)", "type": "number", "min": 0, "max": 120},
        ]
    }
}


# ──────────────────────── Routes ────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main prediction form page."""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "feature_groups": FEATURE_GROUPS,
        "class_names": CLASS_NAMES,
    })


@app.post("/predict", response_class=JSONResponse)
async def predict(payload: PredictionRequest):
    """Run model inference and return predictions."""
    if len(payload.features) != 34:
        return JSONResponse(
            status_code=400,
            content={"error": f"Expected 34 features, got {len(payload.features)}."}
        )

    with torch.inference_mode():
        input_tensor = torch.tensor([payload.features], dtype=torch.float32)
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze()
        predicted_idx = probs.argmax().item()

    probabilities = {
        CLASS_NAMES[i]: round(probs[i].item() * 100, 2) for i in range(6)
    }

    return PredictionResponse(
        predicted_class=predicted_idx + 1,
        predicted_label=CLASS_NAMES[predicted_idx],
        probabilities=probabilities,
    )


@app.get("/api/features", response_class=JSONResponse)
async def get_features():
    """Return feature metadata for dynamic form building."""
    return {"feature_groups": FEATURE_GROUPS, "class_names": CLASS_NAMES}
