import os
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import torch
import joblib
from models.tft_model import TemporalFusionTransformer
from models.tft_data_module import CMAPSSDataModule

# --- CONFIG ---
MODEL_CKPT_DIR = 'checkpoints'
SCALER_PATH = 'transformer_data/scaler.pkl'  # If you save scaler, else None
METADATA_PATH = 'transformer_data/metadata.json'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- LOAD METADATA ---
import json
with open(METADATA_PATH, 'r') as f:
    metadata = json.load(f)
FEATURE_COLS = metadata['feature_columns'] if 'feature_columns' in metadata else [f'feature_{i}' for i in range(metadata['num_features'])]
WINDOW_SIZE = metadata['window_size']

# --- LOAD SCALER (OPTIONAL) ---
if os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)
else:
    scaler = None

# --- LOAD MODEL ---
def get_latest_checkpoint(ckpt_dir):
    ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
    if not ckpts:
        raise FileNotFoundError('No checkpoint found in directory.')
    ckpts = sorted(ckpts, key=lambda x: os.path.getmtime(os.path.join(ckpt_dir, x)), reverse=True)
    return os.path.join(ckpt_dir, ckpts[0])

ckpt_path = get_latest_checkpoint(MODEL_CKPT_DIR)
model = TemporalFusionTransformer(
    num_time_varying_real_vars=metadata['num_features'],
    hidden_size=64  # Should match training
)
model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE)['state_dict'])
model.to(DEVICE)
model.eval()

# --- FASTAPI APP ---
app = FastAPI(title="Predictive Maintenance RUL API")

class SensorInput(BaseModel):
    data: List[List[float]]  # 2D: [window_size, num_features]
    feature_names: Optional[List[str]] = None

class PredictionResponse(BaseModel):
    rul: float
    details: Optional[dict] = None

@app.post('/predict', response_model=PredictionResponse)
def predict_rul(input: SensorInput):
    # Validate input
    if len(input.data) != WINDOW_SIZE:
        raise HTTPException(status_code=400, detail=f"Input must have window_size={WINDOW_SIZE} rows.")
    if input.feature_names:
        if set(input.feature_names) != set(FEATURE_COLS):
            raise HTTPException(status_code=400, detail="Feature names do not match model features.")
        df = pd.DataFrame(input.data, columns=input.feature_names)
        df = df[FEATURE_COLS]
    else:
        df = pd.DataFrame(input.data, columns=FEATURE_COLS)
    # Scale if scaler is available
    if scaler:
        arr = scaler.transform(df.values)
    else:
        arr = np.array(df.values, dtype=np.float32)
    # Convert to tensor
    x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # [1, window, features]
    with torch.no_grad():
        pred = model(x)
        if isinstance(pred, tuple):
            pred = pred[0]
        rul = float(pred.cpu().numpy().flatten()[0])
    return PredictionResponse(rul=rul)

@app.get('/')
def root():
    return {"message": "Predictive Maintenance RUL API. Use /predict POST endpoint."} 