# Turbulence Prediction Backend

Production-ready Flask backend for turbulence probability prediction using PyTorch LSTM Autoencoder, with Gemma LLM fine-tuning for natural-language pilot advisories.

## Features

- **Training Pipeline**: Train LSTM autoencoder from CSV files in `data/raw/`
- **Model Persistence**: Save trained model weights and training score distribution
- **REST API**: Predict turbulence probability, severity, and confidence
- **LLM Advisory Generation**: Fine-tuned Gemma-2B model for generating cockpit-safe turbulence advisories
- **Production Ready**: Gunicorn support, CORS enabled, structured logging

## Project Structure

```
turbulence_backend/
├── app.py                    # Flask application with API endpoints
├── config.py                 # Configuration and environment variables
├── model.py                  # PyTorch LSTM Autoencoder model
├── utils.py                  # Utility functions (preprocessing, scoring)
├── requirements.txt          # Python dependencies
├── Procfile                  # Gunicorn configuration for deployment
├── README.md                 # This file
├── data/
│   ├── raw/                 # Place your training CSV files here
│   ├── processed/           # Processed data (auto-generated)
│   └── llm_train.jsonl      # LLM training dataset (auto-generated)
├── models/
│   └── saved/               # Saved model weights and training scores
├── llm/
│   ├── __init__.py
│   ├── prompts.py          # Prompt templates
│   ├── dataset_builder.py  # Generate synthetic training data
│   ├── gemma_finetune.py   # LoRA fine-tuning script
│   ├── gemma_infer.py      # Inference module
│   └── adapters/           # LoRA adapters (saved after fine-tuning)
│       └── gemma_turbulence_advisor/
├── scripts/
│   ├── preprocess.py        # Data preprocessing script
│   ├── train.py             # Model training script
│   └── compute_training_scores.py  # Compute training anomaly scores
└── tests/
    └── test_api.py          # API tests

## Setup
python -m venv venv

venv\Scripts\activate


### 1. Install Dependencies
pip install -r requirements.txt
```

### 2. Prepare Training Data

Place your CSV files in `data/raw/`. Each CSV must contain the following columns:

- `time` (timestamp or integer)
- `altitude`
- `velocity`
- `vertical_rate`
- `u_wind`
- `v_wind`
- `temperature`

Example CSV structure:
```csv
time,altitude,velocity,vertical_rate,u_wind,v_wind,temperature
0,35000,450,0,25,-10,220
1,35001,451,1,26,-9,220
...
```

### 3. Setup Gemma LLM (Optional but Recommended)

The LLM module generates natural-language pilot advisories. To use it:

#### 3a. Login to HuggingFace

```bash
huggingface-cli login
```

You'll need a HuggingFace account and access token to download the Gemma model.

#### 3b. Build LLM Training Dataset

Generate synthetic training examples:

```bash
python -m llm.dataset_builder
```

This creates `data/llm_train.jsonl` with 300 training examples.

#### 3c. Fine-tune Gemma Model

Fine-tune the Gemma-2B model using LoRA:

**Basic Training (Default: 3 epochs):**
```bash
python -m llm.gemma_finetune
```

**Train for More Epochs:**

```bash
# 5 epochs
python -m llm.gemma_finetune --epochs 5

# 10 epochs
python -m llm.gemma_finetune --epochs 10

# 20 epochs
python -m llm.gemma_finetune --epochs 20
```

**Advanced Training Options:**

```bash
# More epochs with larger batch size
python -m llm.gemma_finetune --epochs 10 --batch-size 8

# More epochs with custom learning rate
python -m llm.gemma_finetune --epochs 10 --lr 1e-4

# Full training configuration
python -m llm.gemma_finetune --epochs 15 --batch-size 4 --lr 2e-4

# CPU training (disable 4-bit quantization)
python -m llm.gemma_finetune --epochs 5 --no-4bit
```

**Recommended for Better Results:**
```bash
python -m llm.gemma_finetune --epochs 10 --batch-size 4 --lr 2e-4
```

This will:
- Load the base `google/gemma-2b-it` model
- Apply 4-bit quantization (QLoRA) if GPU available
- Fine-tune using LoRA adapters
- Save adapters to `llm/adapters/gemma_turbulence_advisor/`

**Training Parameters:**
- `--epochs`: Number of training epochs (default: 3)
- `--batch-size`: Training batch size (default: 4)
- `--lr`: Learning rate (default: 2e-4)
- `--dataset`: Path to training JSONL file (default: `data/llm_train.jsonl`)
- `--output-dir`: Output directory for adapters (default: `llm/adapters/gemma_turbulence_advisor/`)
- `--no-4bit`: Disable 4-bit quantization (for CPU training)

**Note**: Fine-tuning requires a GPU with at least 8GB VRAM. CPU training is possible but very slow. More epochs may improve performance but monitor for overfitting.

### 4. Environment Variables (Optional)

You can configure the following environment variables:

**Turbulence Model:**
- `PORT`: Server port (default: 5000)
- `MODEL_PATH`: Path to saved model (default: `models/saved/lstm_autoencoder.pth`)
- `SCORES_PATH`: Path to training scores (default: `models/saved/training_scores.npy`)
- `WINDOW_SIZE`: Sliding window size (default: 50)
- `BATCH_SIZE`: Training batch size (default: 32)
- `EPOCHS`: Number of training epochs (default: 50)
- `LR`: Learning rate (default: 0.001)
- `HIDDEN_SIZE`: LSTM hidden size (default: 64)
- `NUM_LAYERS`: Number of LSTM layers (default: 2)

**LLM Model:**
- `GEMMA_MODEL_ID`: HuggingFace model ID (default: `google/gemma-2b-it`)
- `LORA_ADAPTER_PATH`: Path to LoRA adapter directory (default: `llm/adapters/gemma_turbulence_advisor/`)

## Usage

### Training the Model

Train the model using the `/train` endpoint:

```bash
curl -X POST http://localhost:5000/train
```

Or run the training scripts manually:

```bash
# Preprocess data
python scripts/preprocess.py

# Train model
python scripts/train.py

# Compute training scores
python scripts/compute_training_scores.py
```

### Running the Server

**Development:**
```bash
python app.py
```

**Production (with Gunicorn):**
```bash
gunicorn app:app
```

The server will start on `http://localhost:5000` (or the port specified by `PORT`).

### API Endpoints

#### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

#### POST /predict

Predict turbulence probability from a time-series window.

**Request:**
```json
{
  "window": [
    [altitude, velocity, vertical_rate, u_wind, v_wind, temperature],
    [altitude, velocity, vertical_rate, u_wind, v_wind, temperature],
    ...
  ],
  "include_advisory": true,  // Optional: generate LLM advisory
  "time_horizon_min": 10,    // Optional: for advisory (default: 10)
  "altitude_band": "FL360"   // Optional: for advisory (default: "FL360")
}
```

The window must have exactly `WINDOW_SIZE` rows (default: 50) and 6 features per row.

**Response (without advisory):**
```json
{
  "turbulence_probability": 0.65,
  "severity": "Moderate",
  "confidence": "Medium",
  "anomaly_score": 0.0234
}
```

**Response (with advisory):**
```json
{
  "turbulence_probability": 0.65,
  "severity": "Moderate",
  "confidence": "Medium",
  "anomaly_score": 0.0234,
  "advisory": "Moderate turbulence expected at FL360 within 10 minutes. Probability 65%. Consider altitude change or route adjustment. Moderate confidence.",
  "model": "gemma-2b-it+lora"
}
```

**Example:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "window": [
      [35000, 450, 0, 25, -10, 220],
      [35001, 451, 1, 26, -9, 220],
      ...
    ],
    "include_advisory": true,
    "time_horizon_min": 8,
    "altitude_band": "FL360"
  }'
```

#### POST /advisory

Generate turbulence advisory directly from prediction inputs (without running turbulence model).

**Request:**
```json
{
  "probability": 0.74,
  "severity": "Moderate",
  "confidence": "High",
  "time_horizon_min": 8,
  "altitude_band": "FL360"
}
```

**Response:**
```json
{
  "advisory": "Moderate turbulence expected at FL360 within 8 minutes. Probability 74%. Consider altitude change or route adjustment. High confidence.",
  "model": "gemma-2b-it+lora"
}
```

**Example:**
```bash
curl -X POST http://localhost:5000/advisory \
  -H "Content-Type: application/json" \
  -d '{
    "probability": 0.74,
    "severity": "Moderate",
    "confidence": "High",
    "time_horizon_min": 8,
    "altitude_band": "FL360"
  }'
```

#### POST /train

Trigger the training pipeline. This will:
1. Preprocess all CSV files in `data/raw/`
2. Train the LSTM autoencoder
3. Compute and save training scores

**Response:**
```json
{
  "status": "trained",
  "num_windows": 1500,
  "loss": 0.001234
}
```

**Example:**
```bash
curl -X POST http://localhost:5000/train
```

## Model Details

### Architecture

- **Type**: LSTM Autoencoder
- **Input**: Time-series window of shape `(window_size, 6)`
- **Features**: `[altitude, velocity, vertical_rate, u_wind, v_wind, temperature]`
- **Output**: Reconstructed window (same shape as input)

### Training Strategy

1. Load all CSV files from `data/raw/`
2. Extract and normalize features
3. Create sliding windows of size `WINDOW_SIZE`
4. Train autoencoder to reconstruct normal flight data
5. Compute anomaly scores (MSE) on training windows
6. Save model weights and training score distribution

### Anomaly Detection

- **Anomaly Score**: MSE between input window and reconstructed output
- **Probability**: Fraction of training scores less than current score
- **Severity Mapping**:
  - `probability < 0.40` → "Low"
  - `0.40 ≤ probability < 0.70` → "Moderate"
  - `probability ≥ 0.70` → "High"
- **Confidence Mapping**:
  - `score ≤ 50th percentile` → "High"
  - `50th < score ≤ 90th percentile` → "Medium"
  - `score > 90th percentile` → "Low"

## Testing

Run the test suite:

```bash
pytest tests/
```

## Deployment

### Render.com

1. Push your code to a Git repository
2. Create a new Web Service on Render
3. Connect your repository
4. Set build command: `pip install -r requirements.txt`
5. Set start command: `gunicorn app:app`
6. Add environment variables if needed
7. Deploy!

The `Procfile` is already configured for Gunicorn.

## LLM Module Details

### Architecture

- **Base Model**: `google/gemma-2b-it` (2B parameter instruction-tuned Gemma model)
- **Fine-tuning**: LoRA (Low-Rank Adaptation) for efficient fine-tuning
- **Quantization**: 4-bit QLoRA when GPU available, full precision on CPU
- **Output**: Max 2 lines, cockpit-safe natural language advisories

### Training Data Format

Each training example in `data/llm_train.jsonl` follows this format:

```json
{
  "input": {
    "probability": 0.65,
    "severity": "Moderate",
    "confidence": "High",
    "time_horizon_min": 10,
    "altitude_band": "FL360"
  },
  "output": "Moderate turbulence expected at FL360 within 10 minutes. Probability 65%. Consider altitude change or route adjustment. High confidence."
}
```

### Fine-tuning Process

1. **Dataset Generation**: `python -m llm.dataset_builder` creates 300 synthetic examples
2. **Fine-tuning**: `python -m llm.gemma_finetune` trains LoRA adapters
3. **Inference**: Adapters are loaded automatically when generating advisories

### GPU vs CPU

- **GPU (Recommended)**: Uses 4-bit quantization, ~8GB VRAM required, fast inference
- **CPU**: Full precision, slower but works without GPU

The system automatically detects available hardware and adjusts accordingly.

## Notes

- The turbulence model assumes training data is mostly "normal" flight data
- Input windows are automatically normalized using z-score normalization
- The turbulence model is loaded lazily on first prediction request
- Training scores are required for probability and confidence calculations
- LLM module is optional - API works without it, but advisory generation will be disabled
- If LoRA adapters are missing, the system falls back to base Gemma model

## License

MIT

