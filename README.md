# AirGuard AI

A bilingual air-quality dashboard that keeps Open-Meteo's CAMS forecast and a separately trained AirGuard machine-learning forecast visibly distinct. Both are model estimates, not local sensor readings.

## Dashboard only

The static dashboard can still be opened directly in a browser. To use its API-backed live AQI and weather panels, serve it locally:

```powershell
py -m http.server 8001
```

Open `http://localhost:8001`. The existing CAMS API 24-hour forecast does not require the Python service. The custom ML section stays unavailable until the training and service steps below are completed.

## Train the custom 24-hour model

Requirements: Python 3.10 or newer and internet access to the Open-Meteo Air Quality API.

In PowerShell, from the project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python train_model.py --latitude 26.8467 --longitude 80.9462 --history-days 92
```

The dashboard defaults to Lucknow (26.8467, 80.9462). Training downloads up to 92 days of hourly US AQI and PM2.5 CAMS estimates for those coordinates, records the resolved CAMS grid in the artifact, and writes the fitted model to `artifacts\aqi_model.joblib` plus its data window, chronological holdout scores, and baseline scores to `artifacts\evaluation.json`. The script fails rather than filling missing values or producing success-shaped results if the archive is incomplete or too short.

The pipeline uses lagged AQI/PM2.5, rolling statistics, forecast horizon, and target-time calendar features. A chronological 80/20 split reserves later data for evaluation; test forecasts are issued at non-overlapping 24-hour origins. It reports MAE and RMSE for the trained model, last-value persistence, and previous-day same-hour seasonal-naive baselines. Only after evaluation is the deployment model refit on all available training history.

The archive is currently limited to roughly three months at one CAMS grid location. These scores measure agreement with held-out CAMS model estimates—not real-world sensor accuracy—and do not establish seasonal or city-wide performance. Retrain for a different location.

## Run the dashboard and inference API

Keep the virtual environment active and start the API in one PowerShell window:

```powershell
python -m uvicorn ml_api:app --host 127.0.0.1 --port 8000
```

In another PowerShell window, serve the dashboard:

```powershell
py -m http.server 8001
```

Open `http://localhost:8001`. The “CAMS model forecast” remains the existing Open-Meteo forecast. “AirGuard ML predictions” are returned separately by the local Python service at `127.0.0.1:8000`, with chronological holdout MAE/RMSE and the persistence baseline shown below them. The service uses current CAMS hourly inputs and returns an unavailable/error state if the model, archive, or trained location is not available.

Run the focused checks with:

```powershell
python -m unittest test_ml_model.py
```

## Other interactions

- Use the EN / हि control to switch languages.
- Change the pollution chart between 24 hours, 7 days, and 30 days.
- Use the microphone in a supported browser (such as Chrome or Edge) to ask about AQI or forecasts. Speech recognition requests microphone access and requires a secure context (`localhost` or HTTPS). Allow microphone access when prompted; if it was previously blocked, update the site's microphone permission in browser settings and reload.
- The voice assistant shows Listening, Processing, Success, or Error while handling a spoken question. If speech input is unavailable or permission is denied, use the text question field as a fallback.
- The map is a schematic of CAMS grid estimates, not a station map.
