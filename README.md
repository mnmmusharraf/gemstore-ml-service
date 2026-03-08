<div align="center">

# 💎 GemStore ML Service

### Precious Gemstone Price Prediction API

![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109.0-009688?style=flat-square&logo=fastapi)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0.3-FF6600?style=flat-square)

A **FastAPI** machine learning service that predicts gemstone prices in **LKR and USD** with live exchange rates.
Supports Sapphires, Rubies, Emeralds, and Diamonds with **85.83% R² accuracy** and **9.38% median error**.

Part of the [GemStore](https://github.com/mnmmusharraf/gemstore-backend) marketplace platform.

[API Docs](#-api-documentation) · [Getting Started](#-getting-started) · [Model Details](#-model-details) · [Related Repos](#-related-repositories)

</div>

---

## 📦 Related Repositories

| Repository | Description |
|---|---|
| [gemstore-backend](https://github.com/mnmmusharraf/gemstore-backend) | Spring Boot backend API |
| [gemstore-web](https://github.com/mnmmusharraf/gemstore-web) | React user-facing application |
| [gemstore-admin](https://github.com/mnmmusharraf/gemstore-admin) | React admin dashboard |

---

## ✨ Features

- **Single and batch gem price prediction**
- **Live USD ↔ LKR exchange rate** with automatic hourly refresh and fallback caching
- **Confidence levels** (high / medium / low) per prediction
- **Quality grading** (AAA → C) based on clarity, cut, color, treatment, and origin
- **Price range bounds** (±10–25% depending on confidence)
- **Price factor breakdown** showing origin, color quality, and treatment impact
- **Automatic warnings** for rare gems or limited training data scenarios
- **Swagger UI** available out of the box at `/docs`

---

## 🛠 Technology Stack

| Layer | Technology |
|---|---|
| Framework | FastAPI 0.109 |
| ML Model | XGBoost 2.0.3 |
| Data Processing | NumPy, Pandas, scikit-learn |
| Model Serialization | joblib |
| HTTP Client | httpx (exchange rate fetching) |
| Validation | Pydantic v2 |
| Server | Uvicorn |

---

## 📁 Project Structure

```
.
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI app, routes, lifespan
│   ├── model.py         # GemPriceModel, FeatureEngineer, ExchangeRateService
│   └── schemas.py       # Pydantic request/response models
├── models/
│   ├── feature_config.json               # Feature configuration and scoring weights
│   ├── precious_gems_features.joblib     # Trained feature names list
│   ├── precious_gems_final_model.joblib  # Trained XGBoost model
│   └── precious_gems_metadata.joblib     # Model metadata and version info
└── requirements.txt
```

---

## 🤖 Model Details

| Metric | Value |
|---|---|
| Algorithm | XGBoost |
| R² Accuracy | 85.83% |
| Median Error | 9.38% |
| Target | `log1p(price_lkr)` |
| Output | `expm1(prediction)` → LKR price |

### Supported Gem Types
`sapphire` · `ruby` · `emerald` · `diamond`

### Feature Engineering

The model uses **60+ engineered features** including:

- Carat weight, dimensions (X/Y/Z), volume, XY ratio
- Clarity score (1–5), cut grade (1–5)
- Gem rarity, hardness, base value scores
- Color desirability and rarity scores
- Color quality premium factors
- Origin quality and premium multipliers
- Treatment scores and value impact factors
- Size threshold multipliers and carat flags
- Special combination flags (e.g. `Is_Ceylon_Sapphire`, `Is_Pigeon_Blood`, `Is_Burma_Ruby`, `Is_Padparadscha`)
- Quality grade score and rarity index
- One-hot encoded categoricals: gem type, color, color quality, shape, origin, treatment

### Quality Grading

| Grade | Description |
|---|---|
| AAA | Score ≥ 85 — Museum / Collector quality |
| AA | Score ≥ 70 — Investment grade |
| A | Score ≥ 55 — Premium quality |
| B | Score ≥ 40 — Standard quality |
| C | Score < 40 — Entry level |

### Origin Premiums

| Origin | Premium |
|---|---|
| Myanmar (Burma) | +50% |
| Colombia | +50% |
| Sri Lanka (Ceylon) | +30% |
| Mozambique | +20% |
| Zambia | +10% |

### Treatment Impact

| Treatment | Impact |
|---|---|
| Unheated | +50% (Natural) |
| Heated | Baseline |
| Oiled | −10% |

---

## 🔗 API Documentation

Interactive Swagger UI: `http://localhost:8000/docs`

---

### `GET /`
Root endpoint — returns links to docs and health check.

---

### `GET /health`
Returns service and model status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0",
  "features_count": 87
}
```

---

### `GET /model/info`
Returns detailed model metadata and current exchange rate.

---

### `POST /predict`
Predict the price of a single gemstone.

**Request:**
```json
{
  "gem_type": "sapphire",
  "carat_weight": 2.5,
  "gem_color": "blue",
  "color_quality": "royal",
  "clarity_score": 4,
  "cut_grade_score": 4,
  "shape": "oval",
  "origin": "sri lanka",
  "treatment": "Heated",
  "x": 9.1,
  "y": 7.2,
  "z": 4.8
}
```

| Field | Type | Required | Description |
|---|---|---|---|
| `gem_type` | enum | ✅ | `sapphire`, `ruby`, `emerald`, `diamond` |
| `carat_weight` | float | ✅ | Weight in carats (0–100) |
| `gem_color` | enum | ✅ | `blue`, `red`, `green`, `pink`, `yellow`, `white`, `orange`, `purple`, `teal`, `padparadscha`, `other` |
| `color_quality` | enum | ✅ | `vivid`, `royal`, `cornflower`, `normal`, `light` |
| `clarity_score` | int | ✅ | 1 (lowest) – 5 (highest) |
| `cut_grade_score` | int | ✅ | 1 (lowest) – 5 (highest) |
| `shape` | enum | ✅ | `oval`, `cushion`, `round`, `emerald`, `pear`, `heart`, `marquise`, `princess`, `radiant`, `asscher` |
| `origin` | enum | ✅ | `sri lanka`, `myanmar`, `colombia`, `madagascar`, `mozambique`, `zambia`, `afghanistan`, `tanzania`, `russia`, `pakistan`, `other`, `unknown` |
| `treatment` | enum | ✅ | `Heated`, `Unheated`, `Oiled` |
| `x`, `y`, `z` | float | ❌ | Dimensions in mm — auto-calculated if omitted |

**Response:**
```json
{
  "predicted_price_lkr": 1250000.0,
  "predicted_price_usd": 3846.15,
  "price_range_low_lkr": 1125000.0,
  "price_range_high_lkr": 1375000.0,
  "price_range_low_usd": 3461.54,
  "price_range_high_usd": 4230.77,
  "confidence": "high",
  "quality_grade": "AA",
  "gem_summary": {
    "type": "Blue Sapphire",
    "weight": "2.5 carats",
    "origin": "Sri Lanka (Ceylon)",
    "treatment": "Heated",
    "color_quality": "Royal",
    "clarity": "Score 4/5"
  },
  "price_factors": {
    "origin_impact": "+30% (Ceylon)",
    "color_quality_impact": "+150% (Royal)",
    "treatment_impact": "Standard"
  },
  "warnings": [],
  "exchange_rate": {
    "usd_to_lkr": 325.0,
    "last_updated": "2025-01-01T10:00:00"
  }
}
```

---

### `POST /predict/batch`
Predict prices for multiple gemstones in a single request.

**Request:**
```json
{
  "gems": [
    { "gem_type": "ruby", "carat_weight": 1.2, "gem_color": "red", ... },
    { "gem_type": "emerald", "carat_weight": 3.0, "gem_color": "green", ... }
  ]
}
```

**Response:**
```json
{
  "predictions": [ ... ],
  "total_gems": 2,
  "processing_time_ms": 12.5
}
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- pip

---

### 1. Clone the repository

```bash
git clone https://github.com/mnmmusharraf/gemstore-ml-service
cd gemstore-ml-service
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the service

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Service starts at: `http://localhost:8000`
Swagger UI: `http://localhost:8000/docs`

---

### Running with auto-reload (development)

```bash
uvicorn app.main:app --reload --port 8000
```

---

## 💱 Live Exchange Rates

The service fetches live **USD → LKR** exchange rates automatically with an hourly cache. It tries three free APIs in order:

1. `open.er-api.com`
2. `api.frankfurter.app`
3. `cdn.jsdelivr.net` (fawazahmed0 currency API)

If all APIs fail, a fallback rate of **325.0 LKR/USD** is used. The current rate and last update time are included in every prediction response.

---

## 🔗 Integration with GemStore Backend

This service is called by the Spring Boot backend via `WebClient`.

**Backend configuration:**
```properties
gem-price-api.base-url=http://localhost:8000
gem-price-api.predict-endpoint=/predict
gem-price-api.health-endpoint=/health
```

**Backend endpoint:** `POST /api/v1/gems/price/predict`

---
