from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import logging

from app.schemas import (
    GemInput, PriceResponse, HealthResponse,
    BatchGemInput, BatchPriceResponse
)
from app.model import GemPriceModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model variable
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events"""
    global model
    # Startup
    try:
        model = GemPriceModel(model_dir="models")
        logger.info("✅ Model loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise

    yield  # App is running

    # Shutdown (cleanup if needed)
    logger.info("Shutting down...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="💎 Gem Price Prediction API",
    description="""
    ## Precious Gems Price Prediction Service

    Predicts prices for **Sapphires, Rubies, Emeralds, and Diamonds**

    ### Performance:
    - 🎯 **85.83% R² accuracy**
    - 📊 **9.38% median error**

    Built with ❤️ by mnm-musharraf
    """,
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "💎 Welcome to Gem Price Prediction API!",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    info = model.get_model_info()
    return {
        "status": "healthy",
        "model_loaded": info["model_loaded"],
        "model_version": info["model_version"],
        "features_count": info["features_count"]
    }


@app.post("/predict", response_model=PriceResponse, tags=["Prediction"])
async def predict_price(gem: GemInput):
    """Predict gem price"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        gem_data = {
            "gem_type": gem.gem_type.value,
            "carat_weight": gem.carat_weight,
            "gem_color": gem.gem_color.value,
            "color_quality": gem.color_quality.value,
            "clarity_score": gem.clarity_score,
            "cut_grade_score": gem.cut_grade_score,
            "shape": gem.shape.value,
            "origin": gem.origin.value,
            "treatment": gem.treatment.value,
            "x": gem.x,
            "y": gem.y,
            "z": gem.z
        }

        result = model.predict(gem_data)
        return PriceResponse(**result)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPriceResponse, tags=["Prediction"])
async def predict_batch(batch: BatchGemInput):
    """Predict prices for multiple gems"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start_time = time.time()
    predictions = []

    for gem in batch.gems:
        try:
            gem_data = {
                "gem_type": gem.gem_type.value,
                "carat_weight": gem.carat_weight,
                "gem_color": gem.gem_color.value,
                "color_quality": gem.color_quality.value,
                "clarity_score": gem.clarity_score,
                "cut_grade_score": gem.cut_grade_score,
                "shape": gem.shape.value,
                "origin": gem.origin.value,
                "treatment": gem.treatment.value,
                "x": gem.x,
                "y": gem.y,
                "z": gem.z
            }
            result = model.predict(gem_data)
            predictions.append(PriceResponse(**result))
        except Exception as e:
            logger.error(f"Batch error: {e}")

    return BatchPriceResponse(
        predictions=predictions,
        total_gems=len(batch.gems),
        processing_time_ms=round((time.time() - start_time) * 1000, 2)
    )


@app.get("/model/info", tags=["Model"])
async def get_model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model.get_model_info()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)