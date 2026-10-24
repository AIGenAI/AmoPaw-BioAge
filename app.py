from pathlib import Path
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Literal
import joblib

BASE = Path(__file__).resolve().parent
app = FastAPI(title="AmoPaw BioAge API", version="0.1.0",
              description="Exploratory prototype only.")
metadata = joblib.load(BASE / "model_metadata.joblib")
kind = metadata["kind"]
if kind == "CatBoostRegressor":
    from catboost import CatBoostRegressor
    model = CatBoostRegressor()
    model.load_model(str(BASE / metadata["model_file"]))
else:
    model = metadata["model"]

FEATURES = metadata["feature_cols"]
CAT = metadata["cat_features"]
NUM = metadata["num_features"]
MEDIANS = metadata.get("imputation_medians", {})

class BioAgeRequest(BaseModel):
    pet_id: str = Field(..., min_length=1, max_length=100)
    species: Literal["犬", "猫", "dog", "cat"]
    chronological_age_years: Optional[float] = Field(None, ge=0, le=40)
    sex: Optional[str] = "未知"
    neutered: Optional[str] = "未知"
    breed: Optional[str] = "未知"
    weight_kg: Optional[float] = Field(None, ge=0, le=250)
    respiratory_rate_median: Optional[float] = Field(None, ge=0, le=300)
    respiratory_rate_std: Optional[float] = Field(None, ge=0, le=300)
    respiratory_valid_rate: Optional[float] = Field(None, ge=0, le=1)
    heart_rate_median: Optional[float] = Field(None, ge=0, le=500)
    heart_rate_std: Optional[float] = Field(None, ge=0, le=500)
    heart_rate_valid_rate: Optional[float] = Field(None, ge=0, le=1)
    hrv_rmssd_median: Optional[float] = Field(None, ge=0, le=5000)
    hrv_rmssd_std: Optional[float] = Field(None, ge=0, le=5000)
    hrv_valid_rate: Optional[float] = Field(None, ge=0, le=1)
    high_signal_quality_ratio: Optional[float] = Field(None, ge=0, le=1)
    valid_measurement_ratio: Optional[float] = Field(None, ge=0, le=1)
    valid_record_count: Optional[int] = Field(0, ge=0, le=100000)

def make_row(req):
    species = {"dog":"犬","cat":"猫"}.get(req.species, req.species)
    d = {
        "物种": species, "性别": req.sex or "未知", "绝育状态": req.neutered or "未知",
        "品种": req.breed or "未知", "体重_kg": req.weight_kg,
        "呼吸率中位数": req.respiratory_rate_median,
        "呼吸率中位数_波动": req.respiratory_rate_std,
        "呼吸率中位数_有效率": req.respiratory_valid_rate,
        "心率中位数": req.heart_rate_median,
        "心率中位数_波动": req.heart_rate_std,
        "心率中位数_有效率": req.heart_rate_valid_rate,
        "HRV_RMSSD中位数": req.hrv_rmssd_median,
        "HRV_RMSSD中位数_波动": req.hrv_rmssd_std,
        "HRV_RMSSD中位数_有效率": req.hrv_valid_rate,
        "高信号质量占比": req.high_signal_quality_ratio,
        "有效测量占比": req.valid_measurement_ratio,
        "有效记录数": req.valid_record_count or 0,
    }
    return pd.DataFrame([[d.get(c) for c in FEATURES]], columns=FEATURES)

@app.get("/health")
def health():
    return {"status":"ok","model_version":"bioage-synthetic-v0.1","model_type":kind,
            "warning":"prototype, not validated for real pet age assessment."}

@app.post("/api/v1/bioage/predict")
def predict(req: BioAgeRequest):
    # Hard gate: do not produce a pseudo-estimate without sufficient reliable inputs.
    vital_inputs = [req.respiratory_rate_median, req.heart_rate_median]
    if req.valid_record_count is None or req.valid_record_count < 2:
        return {"pet_id":req.pet_id,"status":"insufficient_data","predicted_age_years":None,
                "interval_low":None,"interval_high":None,"reason":"need_at_least_2_valid_records",
                "model_version":"bioage-synthetic-v0.1"}
    if sum(v is not None for v in vital_inputs) < 1:
        return {"pet_id":req.pet_id,"status":"insufficient_data","predicted_age_years":None,
                "interval_low":None,"interval_high":None,"reason":"no_valid_vital_sign_features",
                "model_version":"bioage-synthetic-v0.1"}
    x = make_row(req)
    if kind == "CatBoostRegressor":
        for c in CAT: x[c] = x[c].fillna("未知").astype(str)
        for c in NUM: x[c] = pd.to_numeric(x[c],errors="coerce").fillna(MEDIANS.get(c,0.0))
        pred = float(model.predict(x)[0])
    else:
        pred = float(model.predict(x)[0])
    pred = max(0.0, min(40.0, pred))
    return {"pet_id":req.pet_id,"status":"experimental_estimate",
            "predicted_age_years":round(pred,2),"interval_low":None,"interval_high":None,
            "chronological_age_years":req.chronological_age_years,
            "age_deviation_years":round(pred-req.chronological_age_years,2) if req.chronological_age_years is not None else None,
            "model_version":"bioage-synthetic-v0.1",
            "warning":"Exploratory estimate from a small-data model. Not a validated biological age or veterinary diagnosis.",
            "quality_note":"This prototype does not yet provide a calibrated confidence interval."}
