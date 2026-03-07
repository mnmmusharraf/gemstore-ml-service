from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Optional, List
from enum import Enum


class GemType(str, Enum):
    sapphire = "sapphire"
    ruby = "ruby"
    emerald = "emerald"
    diamond = "diamond"


class GemColor(str, Enum):
    blue = "blue"
    red = "red"
    green = "green"
    pink = "pink"
    yellow = "yellow"
    white = "white"
    orange = "orange"
    purple = "purple"
    teal = "teal"
    padparadscha = "padparadscha"
    other = "other"


class ColorQuality(str, Enum):
    vivid = "vivid"
    royal = "royal"
    cornflower = "cornflower"
    normal = "normal"
    light = "light"


class Shape(str, Enum):
    oval = "oval"
    cushion = "cushion"
    round = "round"
    emerald_cut = "emerald"
    pear = "pear"
    heart = "heart"
    marquise = "marquise"
    princess = "princess"
    radiant = "radiant"
    asscher = "asscher"


class Origin(str, Enum):
    sri_lanka = "sri lanka"
    myanmar = "myanmar"
    colombia = "colombia"
    madagascar = "madagascar"
    mozambique = "mozambique"
    zambia = "zambia"
    afghanistan = "afghanistan"
    tanzania = "tanzania"
    russia = "russia"
    pakistan = "pakistan"
    other = "other"
    unknown = "unknown"


class Treatment(str, Enum):
    heated = "Heated"
    unheated = "Unheated"
    oiled = "Oiled"


class GemInput(BaseModel):
    """Input schema for gem price prediction"""

    # Use ConfigDict instead of Config class (Pydantic v2)
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "gem_type": "sapphire",
                "carat_weight": 2.5,
                "gem_color": "blue",
                "color_quality": "royal",
                "clarity_score": 4,
                "cut_grade_score": 4,
                "shape": "oval",
                "origin": "sri lanka",
                "treatment": "Heated"
            }
        }
    )

    # Required fields
    gem_type: GemType = Field(..., description="Type of gemstone")
    carat_weight: float = Field(..., gt=0, le=100, description="Weight in carats")
    gem_color: GemColor = Field(..., description="Color of the gem")
    color_quality: ColorQuality = Field(..., description="Quality of the color")
    clarity_score: int = Field(..., ge=1, le=5, description="Clarity score (1-5)")
    cut_grade_score: int = Field(..., ge=1, le=5, description="Cut grade (1-5)")
    shape: Shape = Field(..., description="Shape of the gem")
    origin: Origin = Field(..., description="Origin/source of the gem")
    treatment: Treatment = Field(..., description="Treatment applied")

    # Optional dimensions
    x: Optional[float] = Field(None, ge=0, description="Length in mm")
    y: Optional[float] = Field(None, ge=0, description="Width in mm")
    z: Optional[float] = Field(None, ge=0, description="Depth in mm")

    @field_validator('carat_weight')
    @classmethod
    def validate_carat(cls, v):
        if v <= 0:
            raise ValueError('Carat weight must be positive')
        return v


class PriceResponse(BaseModel):
    """Response schema for price prediction"""

    # Fix: Use protected_namespaces to avoid conflict
    model_config = ConfigDict(
        protected_namespaces=(),  # Disable protected namespace check
        json_schema_extra={
            "example": {
                "predicted_price_lkr": 1250000,
                "predicted_price_usd": 3850,
                "price_range_low_lkr": 1062500,
                "price_range_high_lkr": 1437500,
                "confidence": "high",
                "quality_grade": "AA",
                "gem_summary": {
                    "type": "Blue Sapphire",
                    "weight": "2.5 carats",
                    "origin": "Sri Lanka (Ceylon)",
                    "treatment": "Heated"
                },
                "price_factors": {
                    "origin_premium": "+30%",
                    "color_quality": "+100% (Royal Blue)",
                    "treatment_impact": "Standard (Heated)"
                },
                "warnings": []
            }
        }
    )

    predicted_price_lkr: float = Field(..., description="Predicted price in LKR")
    predicted_price_usd: float = Field(..., description="Predicted price in USD")
    price_range_low_lkr: float = Field(..., description="Lower bound of price range")
    price_range_high_lkr: float = Field(..., description="Upper bound of price range")
    confidence: str = Field(..., description="Prediction confidence level")
    quality_grade: str = Field(..., description="Overall quality grade (AAA, AA, A, B, C)")
    gem_summary: dict = Field(..., description="Summary of gem characteristics")
    price_factors: dict = Field(..., description="Factors affecting the price")
    warnings: List[str] = Field(default=[], description="Any warnings about the prediction")


class HealthResponse(BaseModel):
    """Health check response"""

    # Fix: Rename fields to avoid 'model_' prefix conflict
    model_config = ConfigDict(protected_namespaces=())

    status: str
    is_model_loaded: bool = Field(..., alias="model_loaded")
    version: str = Field(..., alias="model_version")
    features_count: int


class BatchGemInput(BaseModel):
    """Batch input for multiple gems"""
    gems: List[GemInput]


class BatchPriceResponse(BaseModel):
    """Batch response for multiple gems"""
    predictions: List[PriceResponse]
    total_gems: int
    processing_time_ms: float