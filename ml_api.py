"""Local inference API for the separately trained AirGuard AQI model."""

from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from ml_model import fetch_cams_history, predict_next_24


ARTIFACT_PATH = Path(__file__).resolve().parent / "artifacts" / "aqi_model.joblib"
app = FastAPI(title="AirGuard custom AQI model", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def load_bundle() -> dict[str, Any]:
    if not ARTIFACT_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail="Custom ML model is not trained. Run train_model.py first.",
        )
    try:
        bundle = joblib.load(ARTIFACT_PATH)
    except (OSError, ValueError, EOFError) as error:
        raise HTTPException(status_code=500, detail=f"Could not load trained model: {error}") from error
    if not isinstance(bundle, dict) or "model" not in bundle or "metadata" not in bundle:
        raise HTTPException(status_code=500, detail="Trained model artifact has an invalid format.")
    return bundle


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ready" if ARTIFACT_PATH.is_file() else "model_not_trained",
        "model_artifact": str(ARTIFACT_PATH),
    }


@app.get("/api/ml-forecast")
def ml_forecast(
    latitude: float = Query(..., ge=-90, le=90),
    longitude: float = Query(..., ge=-180, le=180),
) -> dict[str, Any]:
    bundle = load_bundle()
    try:
        history, current = fetch_cams_history(
            latitude,
            longitude,
            past_days=8,
            forecast_days=1,
        )
        predictions = predict_next_24(
            bundle,
            history,
            current,
            latitude,
            longitude,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error

    return {
        "model": bundle["metadata"]["model_name"],
        "source": "AirGuard custom ML trained on Open-Meteo CAMS model estimates",
        "training_location": bundle["metadata"]["training_location"],
        "input_updated_at": current["time"],
        "timezone": history.timezone,
        "evaluation": bundle["metadata"]["evaluation"],
        "predictions": predictions,
    }
