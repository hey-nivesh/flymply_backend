# Hackathon CAT Prediction System - Setup Guide

## What Was Built

A hackathon-ready real-time turbulence prediction system with:
- **Backend**: Flask API with weather service, ML model, and AI reasoning
- **Frontend**: React dashboard with real-time updates

## Backend Files Created

1. **services/weather_service.py** - Mock weather data generation
2. **services/ml_service.py** - RandomForest classifier for turbulence prediction
3. **services/gemma_service.py** - Google Gemini AI for reasoning and adjustment
4. **utils/feature_engineering.py** - Feature extraction from weather data
5. **app.py** - Updated with new endpoints:
   - `GET /weather` - Fetch weather data
   - `POST /predict-cat` - Predict turbulence risk (note: named `predict-cat` to avoid conflict with existing `/predict`)

## Frontend Files Created

1. **components/CATDashboard.tsx** - Main dashboard component
2. **pages/CAT.tsx** - Page route for the dashboard
3. **App.tsx** - Updated with `/cat` route

## Setup Instructions

### 1. Install Backend Dependencies

```bash
cd flymply_backend
pip install -r requirements.txt
```

### 2. Create .env File

Create `flymply_backend/.env` with:
```
GOOGLE_API_KEY=your_google_api_key_here
```

Get your API key from: https://aistudio.google.com/apikey

### 3. Run Backend

```bash
cd flymply_backend
python app.py
```

Backend will run on `http://localhost:5000` (or port from config)

### 4. Run Frontend

```bash
cd flymply-frontend
npm install
npm run dev
```

Frontend will run on `http://localhost:3000` (or Vite default port)

### 5. Access Dashboard

Navigate to: `http://localhost:3000/cat`

## API Endpoints

### GET /weather
Fetch current weather data (mock data for hackathon)

**Query Parameters:**
- `lat` (optional): Latitude (default: 40.7128)
- `lon` (optional): Longitude (default: -74.0060)

**Response:**
```json
{
  "wind_speed": 25.5,
  "temperature": -15.3,
  "pressure": 1013.2,
  "humidity": 65.0,
  "timestamp": "2024-01-01T12:00:00"
}
```

### POST /predict-cat
Predict turbulence risk

**Request Body:**
```json
{
  "lat": 40.7128,
  "lon": -74.0060
}
```

**Response:**
```json
{
  "turbulence_probability": 45,
  "risk_level": "Medium",
  "weather": { ... },
  "features": { ... },
  "ml_score": 42,
  "gemma_adjustment": 3,
  "reasoning": "Wind conditions suggest moderate risk..."
}
```

## Testing

### Test Weather Endpoint
```bash
curl http://localhost:5000/weather
```

### Test Prediction Endpoint
```bash
curl -X POST http://localhost:5000/predict-cat \
  -H "Content-Type: application/json" \
  -d '{}'
```

## Notes

- The ML model trains automatically on first run (synthetic data)
- Weather data is mocked for hackathon purposes
- Gemma service uses Google Gemini API (fallback if unavailable)
- Dashboard auto-refreshes every 30 seconds
- All services include error handling fallbacks

## Troubleshooting

1. **Import errors**: Make sure all dependencies are installed
2. **Gemma errors**: Check your GOOGLE_API_KEY in .env
3. **CORS errors**: Flask-CORS is configured, should work by default
4. **Model training fails**: Check that `models/` directory is writable

