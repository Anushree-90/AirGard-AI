"""CAMS-based AQI model training and inference helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import requests
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
HISTORY_DAYS = 92
MIN_HISTORY_DAYS = 30
DEFAULT_TRAINING_LATITUDE = 26.8467
DEFAULT_TRAINING_LONGITUDE = 80.9462
MAX_LAG_HOURS = 168
FORECAST_HORIZON_HOURS = 24
FEATURE_NAMES = (
    *(f"aqi_lag_{lag}" for lag in (0, 1, 2, 3, 6, 12, 24, 48, 72, 168)),
    *(f"pm25_lag_{lag}" for lag in (0, 1, 2, 3, 6, 12, 24, 48, 72, 168)),
    *(f"aqi_mean_{window}" for window in (6, 24, 72, 168)),
    *(f"pm25_mean_{window}" for window in (6, 24, 72, 168)),
    "aqi_std_24",
    "aqi_std_168",
    "pm25_std_24",
    "pm25_std_168",
    "horizon",
    "target_hour_sin",
    "target_hour_cos",
    "target_weekday_sin",
    "target_weekday_cos",
    "target_dayofyear_sin",
    "target_dayofyear_cos",
)


@dataclass(frozen=True)
class History:
    times: list[datetime]
    aqi: np.ndarray
    pm25: np.ndarray
    latitude: float
    longitude: float
    timezone: str


def fetch_cams_history(
    latitude: float,
    longitude: float,
    *,
    past_days: int,
    forecast_days: int = 0,
) -> tuple[History, dict[str, Any] | None]:
    """Fetch hourly CAMS values; reject incomplete or irregular series."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "us_aqi,pm2_5" if forecast_days else None,
        "hourly": "us_aqi,pm2_5",
        "past_days": past_days,
        "forecast_days": forecast_days,
        "timezone": "UTC",
    }
    params = {key: value for key, value in params.items() if value is not None}
    try:
        response = requests.get(AIR_QUALITY_URL, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as error:
        raise RuntimeError(f"Could not retrieve CAMS air-quality history: {error}") from error

    if not isinstance(payload, dict):
        raise RuntimeError("CAMS response is not a JSON object.")
    hourly = payload.get("hourly")
    if not isinstance(hourly, dict):
        raise RuntimeError("CAMS response has no hourly data.")
    times_raw = hourly.get("time")
    aqi_raw = hourly.get("us_aqi")
    pm25_raw = hourly.get("pm2_5")
    if not isinstance(times_raw, list) or not isinstance(aqi_raw, list) or not isinstance(pm25_raw, list):
        raise RuntimeError("CAMS response is missing hourly time, US AQI, or PM2.5 series.")
    if not times_raw or len(times_raw) != len(aqi_raw) or len(times_raw) != len(pm25_raw):
        raise RuntimeError("CAMS response contains empty or mismatched hourly series.")

    try:
        times = [datetime.fromisoformat(value) for value in times_raw]
        aqi = np.asarray(aqi_raw, dtype=float)
        pm25 = np.asarray(pm25_raw, dtype=float)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"CAMS response contains invalid hourly values: {error}") from error
    if not np.isfinite(aqi).all() or not np.isfinite(pm25).all():
        raise RuntimeError("CAMS response has missing hourly AQI or PM2.5 values; no gaps were filled.")
    if any((later - earlier) != timedelta(hours=1) for earlier, later in zip(times, times[1:])):
        raise RuntimeError("CAMS hourly timestamps are not continuous; no missing hours were fabricated.")
    try:
        grid_latitude = float(payload["latitude"])
        grid_longitude = float(payload["longitude"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("CAMS response has no valid grid coordinates.") from error
    if not math.isfinite(grid_latitude) or not math.isfinite(grid_longitude):
        raise RuntimeError("CAMS response has no valid grid coordinates.")
    timezone = payload.get("timezone")
    if not isinstance(timezone, str):
        raise RuntimeError("CAMS response has no timezone.")

    current = payload.get("current")
    return (
        History(
            times=times,
            aqi=aqi,
            pm25=pm25,
            latitude=grid_latitude,
            longitude=grid_longitude,
            timezone=timezone,
        ),
        current if isinstance(current, dict) else None,
    )


def build_feature_row(
    times: list[datetime],
    aqi: np.ndarray,
    pm25: np.ndarray,
    origin_index: int,
    horizon: int,
) -> list[float]:
    """Build features using values available at the forecast origin only."""
    if origin_index < MAX_LAG_HOURS or not 1 <= horizon <= FORECAST_HORIZON_HOURS:
        raise ValueError("Insufficient history or invalid forecast horizon.")

    features: list[float] = []
    for series in (aqi, pm25):
        features.extend(float(series[origin_index - lag]) for lag in (0, 1, 2, 3, 6, 12, 24, 48, 72, 168))
    for series in (aqi, pm25):
        for window in (6, 24, 72, 168):
            features.append(float(np.mean(series[origin_index - window + 1 : origin_index + 1])))
    for series in (aqi, pm25):
        for window in (24, 168):
            features.append(float(np.std(series[origin_index - window + 1 : origin_index + 1])))

    target_time = times[origin_index] + timedelta(hours=horizon)
    hour_angle = 2 * math.pi * target_time.hour / 24
    weekday_angle = 2 * math.pi * target_time.weekday() / 7
    day_angle = 2 * math.pi * target_time.timetuple().tm_yday / 365.25
    features.extend(
        [
            float(horizon),
            math.sin(hour_angle),
            math.cos(hour_angle),
            math.sin(weekday_angle),
            math.cos(weekday_angle),
            math.sin(day_angle),
            math.cos(day_angle),
        ]
    )
    return features


def _training_rows(
    history: History,
    origin_indices: range | list[int],
    *,
    target_before: int | None = None,
) -> tuple[list[list[float]], list[float], list[float], list[float]]:
    features: list[list[float]] = []
    targets: list[float] = []
    persistence: list[float] = []
    seasonal: list[float] = []
    for origin in origin_indices:
        for horizon in range(1, FORECAST_HORIZON_HOURS + 1):
            target_index = origin + horizon
            if target_index >= len(history.times) or (target_before is not None and target_index >= target_before):
                continue
            features.append(build_feature_row(history.times, history.aqi, history.pm25, origin, horizon))
            targets.append(float(history.aqi[target_index]))
            persistence.append(float(history.aqi[origin]))
            seasonal.append(float(history.aqi[target_index - 24]))
    return features, targets, persistence, seasonal


def _scores(actual: list[float], predicted: np.ndarray | list[float]) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(math.sqrt(mean_squared_error(actual, predicted))),
    }


def train_and_save(
    history: History,
    output_dir: Path,
    *,
    source: str,
) -> dict[str, Any]:
    count = len(history.times)
    if count < MIN_HISTORY_DAYS * 24:
        raise ValueError(
            f"Need at least {MIN_HISTORY_DAYS} complete hourly days; CAMS returned {count} records."
        )
    if count <= MAX_LAG_HOURS + 2 * FORECAST_HORIZON_HOURS:
        raise ValueError("Not enough records remain after lag and forecast-window requirements.")

    split_index = int(count * 0.8)
    validation_index = int(split_index * 0.875)
    selection_x, selection_y, selection_persistence, _ = _training_rows(
        history,
        range(MAX_LAG_HOURS, validation_index),
        target_before=validation_index,
    )
    validation_origins = list(
        range(validation_index, split_index - FORECAST_HORIZON_HOURS, FORECAST_HORIZON_HOURS)
    )
    validation_x, validation_y, validation_persistence, _ = _training_rows(
        history,
        validation_origins,
        target_before=split_index,
    )
    train_origins = range(MAX_LAG_HOURS, split_index)
    train_x, train_y, train_persistence, _ = _training_rows(
        history,
        train_origins,
        target_before=split_index,
    )
    test_origins = list(range(split_index, count - FORECAST_HORIZON_HOURS, FORECAST_HORIZON_HOURS))
    test_x, test_y, persistence, seasonal = _training_rows(history, test_origins)
    if (
        len(selection_y) < 1000
        or len(validation_y) < FORECAST_HORIZON_HOURS
        or len(train_y) < 1000
        or len(test_y) < FORECAST_HORIZON_HOURS
    ):
        raise ValueError("Not enough valid chronological train/test examples were produced.")

    model_options = {
        "max_iter": 150,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 30,
        "l2_regularization": 1.0,
        "early_stopping": False,
        "random_state": 42,
    }
    selection_direct = HistGradientBoostingRegressor(**model_options)
    selection_direct.fit(selection_x, selection_y)
    validation_direct = np.clip(selection_direct.predict(validation_x), 0, 500)
    selection_residual = HistGradientBoostingRegressor(**model_options)
    selection_residual.fit(
        selection_x,
        np.asarray(selection_y) - np.asarray(selection_persistence),
    )
    validation_residual = selection_residual.predict(validation_x)

    direct_validation_scores = _scores(validation_y, validation_direct)
    candidates = [
        (
            direct_validation_scores["mae"],
            direct_validation_scores["rmse"],
            "direct",
            1.0,
        )
    ]
    for scale in np.arange(0.1, 1.01, 0.1):
        prediction = np.clip(
            np.asarray(validation_persistence) + scale * validation_residual,
            0,
            500,
        )
        scores = _scores(validation_y, prediction)
        candidates.append((scores["mae"], scores["rmse"], "persistence_residual", float(scale)))
    _, _, selected_mode, residual_scale = min(candidates, key=lambda candidate: (candidate[0], candidate[1]))
    validation_selection_scores = {
        "direct": direct_validation_scores,
        "selected_residual_scale": residual_scale,
        "selected_mode": selected_mode,
        "selected": {
            "mae": min(candidates, key=lambda candidate: (candidate[0], candidate[1]))[0],
            "rmse": min(candidates, key=lambda candidate: (candidate[0], candidate[1]))[1],
        },
    }

    deployed_model = HistGradientBoostingRegressor(**model_options)
    deployed_target = (
        np.asarray(train_y) - np.asarray(train_persistence)
        if selected_mode == "persistence_residual"
        else train_y
    )
    deployed_model.fit(train_x, deployed_target)
    test_prediction = deployed_model.predict(test_x)
    if selected_mode == "persistence_residual":
        test_prediction = np.asarray(persistence) + residual_scale * test_prediction
    test_prediction = np.clip(test_prediction, 0, 500)
    selected_scores = _scores(test_y, test_prediction)
    persistence_scores = _scores(test_y, persistence)
    seasonal_scores = _scores(test_y, seasonal)
    model_beats_persistence = (
        selected_scores["mae"] < persistence_scores["mae"]
        and selected_scores["rmse"] < persistence_scores["rmse"]
    )
    evaluation = {
        "split": "chronological 80/20; test origins spaced 24 hours apart",
        "selection": "chronological validation within the first 80%; test period used only for final evaluation",
        "split_time_utc": history.times[split_index].isoformat(),
        "validation_split_time_utc": history.times[validation_index].isoformat(),
        "test_forecast_origin_start_utc": history.times[test_origins[0]].isoformat(),
        "test_forecast_origin_end_utc": history.times[test_origins[-1]].isoformat(),
        "test_period_start_utc": history.times[test_origins[0] + 1].isoformat(),
        "test_period_end_utc": history.times[test_origins[-1] + FORECAST_HORIZON_HOURS].isoformat(),
        "archive_end_utc": history.times[-1].isoformat(),
        "validation": validation_selection_scores,
        "training_examples": len(train_y),
        "test_examples": len(test_y),
        "test_forecast_origins": len(test_origins),
        "selected_model": "HistGradientBoostingRegressor",
        "selected_mode": selected_mode,
        "beats_persistence_on_both_metrics": model_beats_persistence,
        "models": {
            "selected": selected_scores,
            "persistence_last_aqi": persistence_scores,
            "seasonal_naive_previous_day_same_hour": seasonal_scores,
        },
    }
    evaluation["residual_scale"] = residual_scale if selected_mode == "persistence_residual" else None

    full_x, full_y, full_persistence, _ = _training_rows(
        history,
        range(MAX_LAG_HOURS, count - FORECAST_HORIZON_HOURS),
    )
    deployed_target = (
        np.asarray(full_y) - np.asarray(full_persistence)
        if selected_mode == "persistence_residual"
        else full_y
    )
    deployed_model = HistGradientBoostingRegressor(**model_options)
    deployed_model.fit(full_x, deployed_target)

    metadata = {
        "model_name": (
            "HistGradientBoostingRegressor (persistence residual)"
            if selected_mode == "persistence_residual"
            else "HistGradientBoostingRegressor"
        ),
        "prediction_mode": selected_mode,
        "residual_scale": residual_scale if selected_mode == "persistence_residual" else 1.0,
        "target": "Open-Meteo CAMS US AQI",
        "features": list(FEATURE_NAMES),
        "forecast_horizon_hours": FORECAST_HORIZON_HOURS,
        "history_records": count,
        "history_start_utc": history.times[0].isoformat(),
        "history_end_utc": history.times[-1].isoformat(),
        "training_location": {
            "latitude": history.latitude,
            "longitude": history.longitude,
            "timezone": history.timezone,
        },
        "training_source": source,
        "evaluation": evaluation,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": deployed_model, "metadata": metadata}, output_dir / "aqi_model.joblib")
    (output_dir / "evaluation.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def predict_next_24(
    bundle: dict[str, Any],
    history: History,
    current: dict[str, Any] | None,
    requested_latitude: float,
    requested_longitude: float,
) -> list[dict[str, Any]]:
    metadata = bundle["metadata"]
    grid = metadata["training_location"]
    if abs(requested_latitude - grid["latitude"]) > 0.2 or abs(requested_longitude - grid["longitude"]) > 0.2:
        raise ValueError(
            "This model is trained only for its configured CAMS grid; retrain it for this location."
        )
    if abs(history.latitude - grid["latitude"]) > 0.001 or abs(history.longitude - grid["longitude"]) > 0.001:
        raise ValueError("The live CAMS request resolved to a different grid than the trained model.")
    if current is None or not isinstance(current.get("time"), str):
        raise RuntimeError("CAMS did not return a current timestamp for the custom forecast.")
    try:
        current_time = datetime.fromisoformat(current["time"])
    except ValueError as error:
        raise RuntimeError("CAMS returned an invalid current timestamp.") from error

    usable = [index for index, timestamp in enumerate(history.times) if timestamp <= current_time]
    if not usable or history.times[usable[-1]] != current_time:
        raise RuntimeError("CAMS hourly inputs do not include the current forecast origin.")
    origin = usable[-1]
    if origin < MAX_LAG_HOURS:
        raise RuntimeError("CAMS returned too little recent history for the trained lag features.")

    inputs = [
        build_feature_row(history.times, history.aqi, history.pm25, origin, horizon)
        for horizon in range(1, FORECAST_HORIZON_HOURS + 1)
    ]
    predictions = bundle["model"].predict(inputs)
    if metadata.get("prediction_mode") == "persistence_residual":
        predictions = predictions * metadata["residual_scale"] + history.aqi[origin]
    predictions = np.clip(predictions, 0, 500)
    return [
        {
            "time": (current_time + timedelta(hours=step)).isoformat(timespec="minutes"),
            "aqi": float(value),
        }
        for step, value in enumerate(predictions, start=1)
    ]
