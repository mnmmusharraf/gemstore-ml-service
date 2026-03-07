import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging
import httpx
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


# =============================================================================
# EXCHANGE RATE SERVICE
# =============================================================================

class ExchangeRateService:
    """Fetches and caches USD to LKR exchange rate"""

    def __init__(self):
        self.cached_rate = 325.0  # Fallback rate
        self.last_updated = None
        self.cache_duration = timedelta(hours=1)  # Update every hour

    def get_usd_to_lkr_rate(self) -> float:
        """Get current USD to LKR rate with caching"""

        # Check if cache is still valid
        if self.last_updated and datetime.now() - self.last_updated < self.cache_duration:
            return self.cached_rate

        # Try to fetch new rate
        rate = self._fetch_rate()
        if rate:
            self.cached_rate = rate
            self.last_updated = datetime.now()
            logger.info(f"Updated exchange rate: 1 USD = {rate} LKR")

        return self.cached_rate

    def _fetch_rate(self) -> float | None:
        """Fetch rate from free API"""

        # List of free APIs to try
        apis = [
            self._fetch_from_exchangerate_api,
            self._fetch_from_frankfurter,
            self._fetch_from_fawazahmed,
        ]

        for api_func in apis:
            try:
                rate = api_func()
                if rate and rate > 0:
                    return rate
            except Exception as e:
                logger.warning(f"API failed: {e}")
                continue

        logger.warning("All exchange rate APIs failed, using cached rate")
        return None

    def _fetch_from_exchangerate_api(self) -> float | None:
        """Free API - exchangerate-api.com (no key required for basic)"""
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get("https://open.er-api.com/v6/latest/USD")
                if response.status_code == 200:
                    data = response.json()
                    return data.get("rates", {}).get("LKR")
        except:
            return None

    def _fetch_from_frankfurter(self) -> float | None:
        """Free API - frankfurter.app"""
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get("https://api.frankfurter.app/latest?from=USD&to=LKR")
                if response.status_code == 200:
                    data = response.json()
                    return data.get("rates", {}).get("LKR")
        except:
            return None

    def _fetch_from_fawazahmed(self) -> float | None:
        """Free API - fawazahmed0 currency API"""
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(
                    "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@latest/v1/currencies/usd.json")
                if response.status_code == 200:
                    data = response.json()
                    return data.get("usd", {}).get("lkr")
        except:
            return None


# Global exchange rate service instance
exchange_service = ExchangeRateService()


# =============================================================================
# FEATURE ENGINEER
# =============================================================================

class FeatureEngineer:
    """Handles all feature engineering for gem price prediction"""

    def __init__(self, config: dict = None):
        self.config = config or self._get_default_config()

    def _get_default_config(self) -> dict:
        return {
            "gem_rarity_scores": {"sapphire": 3, "ruby": 5, "emerald": 4, "diamond": 5},
            "gem_base_values": {"sapphire": 3, "ruby": 5, "emerald": 4, "diamond": 5},
            "gem_hardness": {"sapphire": 9, "ruby": 9, "emerald": 7.5, "diamond": 10},
            "color_desirability": {
                "padparadscha": 10, "red": 9, "blue": 8, "green": 7, "pink": 7,
                "teal": 6, "yellow": 5, "purple": 5, "orange": 4, "white": 3, "other": 2
            },
            "color_rarity": {
                "padparadscha": 10, "red": 8, "teal": 7, "blue": 5, "green": 5,
                "pink": 4, "purple": 4, "yellow": 3, "orange": 3, "white": 2, "other": 1
            },
            "color_quality_scores": {"vivid": 5, "royal": 4, "cornflower": 3, "normal": 2, "light": 1},
            "premium_factors": {"vivid": 3.0, "royal": 2.0, "cornflower": 1.5, "normal": 1.0, "light": 0.7},
            "origin_quality": {
                "sri lanka": 5, "myanmar": 5, "colombia": 5,
                "mozambique": 4, "madagascar": 3, "zambia": 4, "afghanistan": 3,
                "tanzania": 3, "russia": 3, "pakistan": 3, "other": 2, "unknown": 1
            },
            "origin_premium": {
                "sri lanka": 1.3, "myanmar": 1.5, "colombia": 1.5,
                "mozambique": 1.2, "madagascar": 1.0, "zambia": 1.1, "afghanistan": 1.0,
                "tanzania": 1.0, "russia": 1.0, "pakistan": 1.0, "other": 0.9, "unknown": 0.8
            },
            "treatment_scores": {"Unheated": 5, "Heated": 3, "Oiled": 2},
            "natural_factors": {"Unheated": 1.0, "Heated": 0.7, "Oiled": 0.5},
            "treatment_value_impacts": {"Unheated": 1.5, "Heated": 1.0, "Oiled": 0.8}
        }

    def calculate_dimensions(self, carat_weight: float, shape: str) -> Dict[str, float]:
        shape_factors = {
            'oval': 1.0, 'cushion': 0.95, 'round': 1.05, 'emerald': 0.9,
            'pear': 0.98, 'heart': 0.95, 'marquise': 1.1, 'princess': 0.92
        }
        factor = shape_factors.get(shape, 1.0)
        base_size = (carat_weight * 0.2) ** (1 / 3) * 6.5 * factor
        return {'x': round(base_size * 1.1, 2), 'y': round(base_size, 2), 'z': round(base_size * 0.6, 2)}

    def process_gem_features(self, gem_data: Dict[str, Any]) -> Dict[str, Any]:
        features = {}

        # Basic
        carat = gem_data['carat_weight']
        features['Carat Weight'] = carat

        # Dimensions
        if gem_data.get('x') and gem_data.get('y') and gem_data.get('z'):
            features['X'], features['Y'], features['Z'] = gem_data['x'], gem_data['y'], gem_data['z']
        else:
            dims = self.calculate_dimensions(carat, gem_data['shape'])
            features['X'], features['Y'], features['Z'] = dims['x'], dims['y'], dims['z']

        features['Volume'] = features['X'] * features['Y'] * features['Z']
        features['XY_Ratio'] = features['X'] / features['Y'] if features['Y'] > 0 else 1

        # Quality
        features['Clarity_Score'] = gem_data['clarity_score']
        features['Cut_Grade_Score'] = gem_data['cut_grade_score']

        # Gem type
        gem_type = gem_data['gem_type']
        features['Gem_Rarity_Score'] = self.config['gem_rarity_scores'].get(gem_type, 3)
        features['Gem_Base_Value'] = self.config['gem_base_values'].get(gem_type, 3)
        features['Gem_Hardness'] = self.config['gem_hardness'].get(gem_type, 7)

        # Color
        gem_color = gem_data['gem_color']
        features['Color_Desirability'] = self.config['color_desirability'].get(gem_color, 5)
        features['Color_Rarity'] = self.config['color_rarity'].get(gem_color, 5)
        features['Color_Warmth'] = 1 if gem_color in ['red', 'orange', 'yellow', 'pink'] else 0

        # Color quality
        color_quality = gem_data['color_quality']
        features['ColorQuality_Score'] = self.config['color_quality_scores'].get(color_quality, 2)
        features['Premium_Factor'] = self.config['premium_factors'].get(color_quality, 1.0)

        # Shape
        shape = gem_data['shape']
        shape_pop = {'oval': 5, 'cushion': 4, 'round': 5, 'emerald': 3, 'pear': 3, 'heart': 2, 'marquise': 2,
                     'princess': 3}
        features['Shape_Popularity'] = shape_pop.get(shape, 3)
        features['Cutting_Difficulty'] = 3
        features['Brilliance_Score'] = 4 if shape in ['round', 'oval', 'cushion'] else 3
        features['Yield_Loss'] = 0.3

        # Origin
        origin = gem_data['origin']
        features['Origin_Quality_Rep'] = self.config['origin_quality'].get(origin, 3)
        features['Origin_Premium'] = self.config['origin_premium'].get(origin, 1.0)

        # Treatment
        treatment = gem_data['treatment']
        features['Treatment_Score'] = self.config['treatment_scores'].get(treatment, 3)
        features['Natural_Factor'] = self.config['natural_factors'].get(treatment, 0.7)
        features['Treatment_Value_Impact'] = self.config['treatment_value_impacts'].get(treatment, 1.0)

        # Size
        features['Size_Score'] = min(5, int(carat / 2) + 1)
        features['Size_Premium'] = 1.0 + (carat - 1) * 0.1 if carat > 1 else 1.0

        # Combined
        features['Quality_Score'] = (features['Clarity_Score'] + features['Cut_Grade_Score'] + features[
            'ColorQuality_Score']) / 3
        features['Overall_Quality'] = (features['Quality_Score'] + features['Gem_Rarity_Score']) / 2
        features['Value_Potential'] = features['Overall_Quality'] * features['Size_Premium']
        features['Combined_Premium'] = features['Premium_Factor'] * features['Origin_Premium']
        features['Rarity_Index'] = features['Gem_Rarity_Score'] * features['Color_Rarity'] / 10

        # Business rules
        features['Origin_Match_Premium'] = self._get_origin_match_premium(gem_type, origin)
        features['Color_Match_Premium'] = self._get_color_match_premium(gem_type, gem_color)
        features['Treatment_Gem_Factor'] = self._get_treatment_gem_factor(gem_type, treatment)
        features['Size_Threshold_Mult'] = self._get_size_threshold_mult(carat)
        features['Quality_Grade_Score'] = self._get_quality_grade_score(gem_data)
        features['Rarity_Score_100'] = self._calculate_rarity_score(gem_data)
        features['Ultimate_Premium'] = self._calculate_ultimate_premium(features)
        features['Clarity_Premium'] = self._get_clarity_premium(features['Clarity_Score'])

        # Flags
        features['Above_1ct'] = 1 if carat >= 1 else 0
        features['Above_2ct'] = 1 if carat >= 2 else 0
        features['Above_3ct'] = 1 if carat >= 3 else 0
        features['Above_5ct'] = 1 if carat >= 5 else 0
        features['Above_10ct'] = 1 if carat >= 10 else 0

        features['Is_Ceylon_Sapphire'] = 1 if gem_type == 'sapphire' and origin == 'sri lanka' else 0
        features['Is_Blue_Sapphire'] = 1 if gem_type == 'sapphire' and gem_color == 'blue' else 0
        features['Is_Padparadscha'] = 1 if gem_color == 'padparadscha' else 0
        features['Is_Royal_Blue'] = 1 if color_quality == 'royal' and gem_color == 'blue' else 0
        features['Is_Cornflower'] = 1 if color_quality == 'cornflower' else 0
        features['Is_Burma_Ruby'] = 1 if gem_type == 'ruby' and origin == 'myanmar' else 0
        features['Is_Pigeon_Blood'] = 1 if gem_type == 'ruby' and gem_color == 'red' and color_quality == 'vivid' else 0
        features['Is_Colombian_Emerald'] = 1 if gem_type == 'emerald' and origin == 'colombia' else 0
        features['Is_Unheated'] = 1 if treatment == 'Unheated' else 0
        features['Is_Investment_Grade'] = 1 if features['Quality_Grade_Score'] >= 4 and carat >= 2 else 0
        features['Is_Museum_Quality'] = 1 if features['Quality_Grade_Score'] >= 5 and carat >= 3 and color_quality in [
            'vivid', 'royal'] else 0
        features['Is_Collectors_Grade'] = 1 if features[
                                                   'Quality_Grade_Score'] >= 4 and carat >= 2 and treatment == 'Unheated' else 0
        features['Is_Perfect_Blue'] = 1 if gem_type == 'sapphire' and gem_color == 'blue' and color_quality in ['royal',
                                                                                                                'cornflower',
                                                                                                                'vivid'] and origin == 'sri lanka' else 0
        features['Is_Loupe_Clean'] = 1 if features['Clarity_Score'] >= 5 else 0
        features['Is_Eye_Clean'] = 1 if features['Clarity_Score'] >= 4 else 0

        # Store categorical values for one-hot encoding
        features['_gem_type'] = gem_type
        features['_gem_color'] = gem_color
        features['_color_quality'] = color_quality
        features['_shape'] = shape
        features['_origin'] = origin
        features['_treatment'] = treatment
        features['_quality_grade'] = self._get_quality_grade(gem_data)

        return features

    def _get_origin_match_premium(self, gem_type: str, origin: str) -> float:
        premiums = {('sapphire', 'sri lanka'): 1.5, ('ruby', 'myanmar'): 2.0, ('ruby', 'mozambique'): 1.8,
                    ('emerald', 'colombia'): 2.0, ('emerald', 'zambia'): 1.4}
        return premiums.get((gem_type, origin), 1.0)

    def _get_color_match_premium(self, gem_type: str, gem_color: str) -> float:
        if gem_color == 'padparadscha': return 3.0
        premiums = {('sapphire', 'blue'): 1.5, ('sapphire', 'pink'): 1.4, ('ruby', 'red'): 1.8,
                    ('emerald', 'green'): 1.5}
        return premiums.get((gem_type, gem_color), 1.0)

    def _get_treatment_gem_factor(self, gem_type: str, treatment: str) -> float:
        factors = {('sapphire', 'Unheated'): 2.0, ('ruby', 'Unheated'): 2.5, ('emerald', 'Unheated'): 1.5}
        return factors.get((gem_type, treatment), 1.0)

    def _get_size_threshold_mult(self, carat: float) -> float:
        if carat >= 10:
            return 3.0
        elif carat >= 5:
            return 2.0
        elif carat >= 3:
            return 1.5
        elif carat >= 2:
            return 1.3
        elif carat >= 1:
            return 1.1
        return 1.0

    def _get_quality_grade_score(self, gem_data: Dict) -> int:
        score = {'vivid': 30, 'royal': 25, 'cornflower': 20, 'normal': 10, 'light': 5}.get(gem_data['color_quality'],
                                                                                           10)
        score += gem_data['clarity_score'] * 5 + gem_data['cut_grade_score'] * 5
        score += {'Unheated': 25, 'Heated': 15, 'Oiled': 10}.get(gem_data['treatment'], 10)
        if score >= 85:
            return 5
        elif score >= 70:
            return 4
        elif score >= 55:
            return 3
        elif score >= 40:
            return 2
        return 1

    def _get_quality_grade(self, gem_data: Dict) -> str:
        return {5: 'AAA', 4: 'AA', 3: 'A', 2: 'B', 1: 'C'}.get(self._get_quality_grade_score(gem_data), 'B')

    def _calculate_rarity_score(self, gem_data: Dict) -> int:
        score = {'ruby': 20, 'emerald': 18, 'diamond': 15, 'sapphire': 12}.get(gem_data['gem_type'], 10)
        score += {'padparadscha': 25, 'red': 20, 'teal': 18, 'blue': 15, 'pink': 14, 'green': 12, 'purple': 10,
                  'yellow': 8, 'orange': 7, 'white': 5, 'other': 3}.get(gem_data['gem_color'], 5)
        score += {'Unheated': 20, 'Heated': 10}.get(gem_data['treatment'], 5)
        carat = gem_data['carat_weight']
        score += 20 if carat >= 10 else 15 if carat >= 5 else 10 if carat >= 3 else 7 if carat >= 2 else 3
        score += {'vivid': 15, 'royal': 12, 'cornflower': 10, 'normal': 5, 'light': 2}.get(gem_data['color_quality'], 5)
        return score

    def _calculate_ultimate_premium(self, features: Dict) -> float:
        return features['Origin_Match_Premium'] * features['Color_Match_Premium'] * features['Treatment_Gem_Factor'] * \
            features['Size_Threshold_Mult'] * features['Premium_Factor']

    def _get_clarity_premium(self, clarity_score: int) -> float:
        return {5: 2.0, 4: 1.5, 3: 1.2, 2: 1.0, 1: 0.8}.get(clarity_score, 1.0)


# =============================================================================
# GEM PRICE MODEL
# =============================================================================

class GemPriceModel:
    """Handles model loading and predictions"""

    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model = None
        self.feature_names = None
        self.metadata = None
        self.feature_engineer = FeatureEngineer()
        self.exchange_service = exchange_service  # Use global instance
        self._load_model()

    @property
    def usd_rate(self) -> float:
        """Get current USD to LKR rate (live)"""
        return self.exchange_service.get_usd_to_lkr_rate()

    def _load_model(self):
        try:
            # Get absolute path
            base_path = Path(__file__).parent.parent / self.model_dir
            if not base_path.exists():
                base_path = Path(self.model_dir)

            logger.info(f"Looking for models in: {base_path.absolute()}")

            # Load model
            model_path = base_path / "precious_gems_final_model.joblib"
            if model_path.exists():
                self.model = joblib.load(model_path)
                logger.info(f"✅ Model loaded")
            else:
                raise FileNotFoundError(f"Model not found: {model_path}")

            # Load features
            feat_path = base_path / "precious_gems_features.joblib"
            if feat_path.exists():
                self.feature_names = joblib.load(feat_path)
                logger.info(f"✅ Features loaded: {len(self.feature_names)}")
            else:
                raise FileNotFoundError(f"Features not found: {feat_path}")

            # Load metadata
            meta_path = base_path / "precious_gems_metadata.joblib"
            if meta_path.exists():
                self.metadata = joblib.load(meta_path)

            logger.info("✅ Model ready!")

        except Exception as e:
            logger.error(f"Error: {e}")
            raise

    def _prepare_features(self, gem_data: Dict[str, Any]) -> pd.DataFrame:
        """Prepare feature vector matching the training data exactly"""

        # Get processed features
        features = self.feature_engineer.process_gem_features(gem_data)

        # Create feature dictionary
        feature_vector = {}

        # Numerical features - direct mapping
        numerical_features = [
            'Carat Weight', 'X', 'Y', 'Z', 'Volume', 'XY_Ratio',
            'Clarity_Score', 'Cut_Grade_Score', 'Quality_Score',
            'Gem_Rarity_Score', 'Gem_Base_Value', 'Gem_Hardness',
            'Color_Desirability', 'Color_Rarity', 'Color_Warmth',
            'ColorQuality_Score', 'Premium_Factor',
            'Shape_Popularity', 'Cutting_Difficulty', 'Brilliance_Score', 'Yield_Loss',
            'Origin_Quality_Rep', 'Origin_Premium',
            'Treatment_Score', 'Natural_Factor', 'Treatment_Value_Impact',
            'Size_Score', 'Size_Premium',
            'Overall_Quality', 'Value_Potential', 'Combined_Premium', 'Rarity_Index',
            'Origin_Match_Premium', 'Color_Match_Premium', 'Treatment_Gem_Factor',
            'Size_Threshold_Mult', 'Quality_Grade_Score', 'Rarity_Score_100',
            'Ultimate_Premium', 'Clarity_Premium',
            'Above_1ct', 'Above_2ct', 'Above_3ct', 'Above_5ct', 'Above_10ct',
            'Is_Ceylon_Sapphire', 'Is_Blue_Sapphire', 'Is_Padparadscha',
            'Is_Royal_Blue', 'Is_Cornflower', 'Is_Burma_Ruby', 'Is_Pigeon_Blood',
            'Is_Colombian_Emerald', 'Is_Unheated', 'Is_Investment_Grade',
            'Is_Museum_Quality', 'Is_Collectors_Grade', 'Is_Perfect_Blue',
            'Is_Loupe_Clean', 'Is_Eye_Clean'
        ]

        # Map numerical features
        for feat in self.feature_names:
            if feat in numerical_features:
                feature_vector[feat] = features.get(feat, 0)
            elif feat.startswith('Master_Gem_Type_'):
                gem_type = feat.replace('Master_Gem_Type_', '')
                feature_vector[feat] = 1 if features['_gem_type'] == gem_type else 0
            elif feat.startswith('Gem_Color_'):
                color = feat.replace('Gem_Color_', '')
                feature_vector[feat] = 1 if features['_gem_color'] == color else 0
            elif feat.startswith('Color_Quality_'):
                quality = feat.replace('Color_Quality_', '')
                feature_vector[feat] = 1 if features['_color_quality'] == quality else 0
            elif feat.startswith('Shape_'):
                shape = feat.replace('Shape_', '')
                feature_vector[feat] = 1 if features['_shape'] == shape else 0
            elif feat.startswith('Origin_') and not feat.startswith('Origin_Match') and not feat.startswith(
                    'Origin_Quality') and not feat.startswith('Origin_Premium'):
                origin = feat.replace('Origin_', '').replace('_', ' ')
                feature_vector[feat] = 1 if features['_origin'] == origin else 0
            elif feat.startswith('Treatment_') and not feat.startswith('Treatment_Score') and not feat.startswith(
                    'Treatment_Value') and not feat.startswith('Treatment_Gem'):
                treatment = feat.replace('Treatment_', '')
                feature_vector[feat] = 1 if features['_treatment'] == treatment else 0
            elif feat.startswith('Quality_Grade_') and not feat.startswith('Quality_Grade_Score'):
                grade = feat.replace('Quality_Grade_', '')
                feature_vector[feat] = 1 if features['_quality_grade'] == grade else 0
            else:
                feature_vector[feat] = 0

        # Create DataFrame with correct column order
        df = pd.DataFrame([feature_vector])
        df = df.reindex(columns=self.feature_names, fill_value=0)

        # Debug: print some values
        logger.info(f"Sample features: Carat={df['Carat Weight'].values[0]}, Volume={df['Volume'].values[0]:.2f}")

        return df

    def predict(self, gem_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make price prediction with live exchange rate"""

        X = self._prepare_features(gem_data)

        # Debug: check input
        logger.info(f"Input shape: {X.shape}, non-zero features: {(X.values != 0).sum()}")

        # Predict log price
        log_price = self.model.predict(X)[0]
        logger.info(f"Log price prediction: {log_price}")

        # Convert from log - model was trained on np.log1p(price)
        price_lkr = np.expm1(log_price)
        logger.info(f"Price LKR: {price_lkr}")

        # If price is still too small, use fallback
        if log_price < 5:
            logger.warning(f"Log price {log_price} seems too low, using fallback...")
            price_lkr = self._estimate_price_fallback(gem_data)
            logger.info(f"Using fallback estimate: {price_lkr}")

        confidence, confidence_level = self._calculate_confidence(gem_data)
        range_factor = 0.10 if confidence_level == "high" else 0.15 if confidence_level == "medium" else 0.25

        # Get LIVE exchange rate
        current_usd_rate = self.usd_rate

        return {
            "predicted_price_lkr": round(float(price_lkr), 2),
            "predicted_price_usd": round(float(price_lkr / current_usd_rate), 2),
            "price_range_low_lkr": round(float(price_lkr * (1 - range_factor)), 2),
            "price_range_high_lkr": round(float(price_lkr * (1 + range_factor)), 2),
            "price_range_low_usd": round(float(price_lkr * (1 - range_factor) / current_usd_rate), 2),
            "price_range_high_usd": round(float(price_lkr * (1 + range_factor) / current_usd_rate), 2),
            "confidence": confidence_level,
            "quality_grade": self.feature_engineer._get_quality_grade(gem_data),
            "gem_summary": self._get_gem_summary(gem_data),
            "price_factors": self._get_price_factors(gem_data),
            "warnings": self._get_warnings(gem_data),
            "exchange_rate": {
                "usd_to_lkr": current_usd_rate,
                "last_updated": self.exchange_service.last_updated.isoformat() if self.exchange_service.last_updated else None
            }
        }

    def _estimate_price_fallback(self, gem_data: Dict) -> float:
        """Fallback price estimation based on business rules"""

        # Base prices per carat (LKR)
        base_prices = {
            'sapphire': 300000,
            'ruby': 800000,
            'emerald': 500000,
            'diamond': 1000000
        }

        base = base_prices.get(gem_data['gem_type'], 400000)
        carat = gem_data['carat_weight']

        # Carat multiplier (exponential for larger gems)
        carat_mult = carat ** 1.5 if carat > 1 else carat

        # Color quality
        color_mult = {'vivid': 4.0, 'royal': 2.5, 'cornflower': 1.8, 'normal': 1.0, 'light': 0.6}.get(
            gem_data['color_quality'], 1.0)

        # Origin
        origin_mult = {'sri lanka': 1.3, 'myanmar': 1.5, 'colombia': 1.5, 'madagascar': 1.0}.get(gem_data['origin'],
                                                                                                 1.0)

        # Treatment
        treatment_mult = {'Unheated': 1.5, 'Heated': 1.0, 'Oiled': 0.9}.get(gem_data['treatment'], 1.0)

        # Clarity
        clarity_mult = 0.8 + (gem_data['clarity_score'] * 0.1)

        # Special combinations
        special_mult = 1.0
        if gem_data['gem_type'] == 'sapphire' and gem_data['gem_color'] == 'blue' and gem_data[
            'color_quality'] == 'royal':
            special_mult = 1.3
        if gem_data['gem_color'] == 'padparadscha':
            special_mult = 2.5

        price = base * carat_mult * color_mult * origin_mult * treatment_mult * clarity_mult * special_mult

        return price

    def _calculate_confidence(self, gem_data: Dict) -> Tuple[float, str]:
        confidence = 0.85
        if gem_data['treatment'] == 'Unheated' and gem_data['color_quality'] in ['vivid', 'royal']:
            if gem_data['gem_type'] == 'sapphire': confidence -= 0.2
        if gem_data['carat_weight'] > 10: confidence -= 0.1
        if gem_data['gem_type'] == 'diamond': confidence -= 0.15
        if confidence >= 0.8:
            return confidence, "high"
        elif confidence >= 0.6:
            return confidence, "medium"
        return confidence, "low"

    def _get_warnings(self, gem_data: Dict) -> List[str]:
        warnings = []
        if gem_data['treatment'] == 'Unheated' and gem_data['gem_type'] == 'sapphire' and gem_data['color_quality'] in [
            'vivid', 'royal']:
            warnings.append("⚠️ Unheated premium sapphire: Limited training data")
        if gem_data['gem_type'] == 'diamond':
            warnings.append("⚠️ Limited diamond data")
        if gem_data['carat_weight'] > 10:
            warnings.append("⚠️ Very large gem")
        return warnings

    def _get_gem_summary(self, gem_data: Dict) -> Dict:
        origin_names = {'sri lanka': 'Sri Lanka (Ceylon)', 'myanmar': 'Myanmar (Burma)', 'colombia': 'Colombia'}
        return {
            "type": f"{gem_data['gem_color'].title()} {gem_data['gem_type'].title()}",
            "weight": f"{gem_data['carat_weight']} carats",
            "origin": origin_names.get(gem_data['origin'], gem_data['origin'].title()),
            "treatment": gem_data['treatment'],
            "color_quality": gem_data['color_quality'].title(),
            "clarity": f"Score {gem_data['clarity_score']}/5"
        }

    def _get_price_factors(self, gem_data: Dict) -> Dict:
        return {
            "origin_impact": {
                'sri lanka': '+30% (Ceylon)', 'myanmar': '+50% (Burma)', 'colombia': '+50% (Colombian)'
            }.get(gem_data['origin'], 'Standard'),
            "color_quality_impact": {
                'vivid': '+300% (Exceptional)', 'royal': '+150% (Royal)', 'cornflower': '+80%', 'normal': 'Baseline',
                'light': '-40%'
            }.get(gem_data['color_quality'], 'Standard'),
            "treatment_impact": {
                'Unheated': '+50% (Natural)', 'Heated': 'Standard', 'Oiled': '-10%'
            }.get(gem_data['treatment'], 'Standard')
        }

    def get_model_info(self) -> Dict:
        return {
            "model_loaded": self.model is not None,
            "model_version": self.metadata.get('version', '1.0') if self.metadata else '1.0',
            "features_count": len(self.feature_names) if self.feature_names else 0,
            "exchange_rate": {
                "usd_to_lkr": self.usd_rate,
                "last_updated": self.exchange_service.last_updated.isoformat() if self.exchange_service.last_updated else None
            }
        }