import joblib
import pandas as pd
import yaml
from datetime import datetime
from schemas import HousePredictionRequest, PredictionResponse

# Paths
MODEL_PATH = "models/trained/house_price_model.pkl"
PREPROCESSOR_PATH = "models/trained/preprocessor.pkl"
CONFIG_PATH = "models/trained/model_config.yaml"

# Load model and preprocessor
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading model: {str(e)}")

try:
    preprocessor = joblib.load(PREPROCESSOR_PATH)
except:
    preprocessor = None  # Optional

# Load config
try:
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    selected_features = config['model']['feature_sets']['rfe']
    # Ensure selected features are human-readable names, not indices
    if isinstance(selected_features, list) and selected_features and str(selected_features[0]).isdigit():
        sample_input = pd.DataFrame([{'bedrooms': 0, 'bathrooms': 1, 'sqft': 1000, 'year_built': 2000}])
        all_columns = list(sample_input.columns) + ['house_age', 'bed_bath_ratio', 'price_per_sqft']
        selected_features = [all_columns[int(i)] for i in selected_features]
except Exception as e:
    raise RuntimeError(f"Error loading model config: {str(e)}")

def preprocess_input(input_data: pd.DataFrame) -> pd.DataFrame:
    input_data['house_age'] = datetime.now().year - input_data['year_built']
    input_data['bed_bath_ratio'] = input_data['bedrooms'] / input_data['bathrooms']
    input_data['price_per_sqft'] = 0  # Dummy value for compatibility
    
    input_data = input_data[selected_features]  # Align with selected features
    
    if preprocessor:
        return preprocessor.transform(input_data)
    else:
        return input_data

def predict_price(request: HousePredictionRequest) -> PredictionResponse:
    input_data = pd.DataFrame([request.dict()])
    processed_features = preprocess_input(input_data)
    predicted_price = model.predict(processed_features)[0]
    predicted_price = round(float(predicted_price), 2)

    confidence_interval = [round(predicted_price * 0.9, 2), round(predicted_price * 1.1, 2)]

    return PredictionResponse(
        predicted_price=predicted_price,
        confidence_interval=confidence_interval,
        features_importance={},
        prediction_time=datetime.now().isoformat()
    )

def batch_predict(requests: list[HousePredictionRequest]) -> list[float]:
    input_data = pd.DataFrame([req.dict() for req in requests])
    processed_features = preprocess_input(input_data)
    predictions = model.predict(processed_features)
    return predictions.tolist()
