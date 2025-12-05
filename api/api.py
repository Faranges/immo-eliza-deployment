# ---------------------------------------------------------
# FASTAPI PRICE PREDICTION API
# ---------------------------------------------------------
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import pickle
from .predict import make_prediction   # ← import from predict.py

app = FastAPI(
    title="Immo Price Prediction API",
    description="API for predicting real estate prices"
)


# ----------------------------------------
# ALLOWED OPTIONS 
# ----------------------------------------
PROPERTY_TYPES = ["Apartment", "House"]

PROPERTY_SUBTYPES = sorted([
    "Apartment", "Residence", "Villa", "Ground floor", "Penthouse",
    "Duplex", "Mixed building", "Studio", "Chalet", "Bungalow",
    "Cottage", "Loft", "Triplex", "Mansion", "Masterhouse"
])

PROVINCES = sorted([
    "Brussels", "Antwerp", "West-Flanders", "East-Flanders",
    "Flemish-Brabant", "Limburg", "Liège", "Brabant-Wallon",
    "Hainaut", "Luxembourg", "Namur"
])

STATE_OF_BUILDING = sorted([
    "To demolish", "Under construction", "To restore", "To renovate",
    "To be renovated", "Normal", "Fully renovated", "Excellent", "New"
])

YES_NO = ["Yes", "No"]

# ----------------------------------------
# PYDANTIC MODEL 
# ----------------------------------------
class PropertyInput(BaseModel):
    # Step 1 fields
    type: str = Field(..., description="Property type", pattern="|".join(PROPERTY_TYPES))
    subtype: str = Field(..., description="Property subtype", pattern="|".join(PROPERTY_SUBTYPES))
    province: str = Field(..., description="Province", pattern="|".join(PROVINCES))
    state_of_building: str = Field(..., description="State of the building", pattern="|".join(STATE_OF_BUILDING))

    # Step 2 fields
    living_area: float = Field(..., ge=18, le=2670, description="Living area (m²)")
    number_of_bedrooms: int = Field(..., ge=1, le=50)
    has_equiped_kitchen: str = Field(..., pattern="|".join(YES_NO))
    is_furnished: str = Field(..., pattern="|".join(YES_NO))
    has_open_fire: str = Field(..., pattern="|".join(YES_NO))

    # Step 3 fields
    has_terrace: str = Field(..., pattern="|".join(YES_NO))
    terrace_area: float = Field(..., ge=0, le=150)
    has_garden: str = Field(..., pattern="|".join(YES_NO))
    number_facades: int = Field(..., ge=1, le=4)
    has_swimming_pool: str = Field(..., pattern="|".join(YES_NO))


# ----------------------------------------
# HEALTH CHECK ENDPOINT
# ----------------------------------------
@app.get("/")
def alive():
    return {"status": "alive", "message": "FastAPI backend running!"}


# ----------------------------------------
# PREDICTION ENDPOINT
# ----------------------------------------
@app.post("/predict")
def predict_price(data: PropertyInput):

    # Convert Yes/No → 1/0
    def yn(x):
        return 1 if x == "Yes" else 0

    # Build DataFrame exactly like Streamlit
    input_df = pd.DataFrame([{
        "type": data.type,
        "subtype": data.subtype,
        "province": data.province,
        "state_of_building": data.state_of_building,
        "living_area (m²)": data.living_area,
        "number_of_bedrooms": data.number_of_bedrooms,
        "number_facades": data.number_facades,
        "equiped_kitchen (yes:1, no:0)": yn(data.has_equiped_kitchen),
        "furnished (yes:1, no:0)": yn(data.is_furnished),
        "open_fire (yes:1, no:0)": yn(data.has_open_fire),
        "terrace (yes:1, no:0)": yn(data.has_terrace),
        "terrace_area (m²)": data.terrace_area,
        "garden (yes:1, no:0)": yn(data.has_garden),
        "swimming_pool (yes:1, no:0)": yn(data.has_swimming_pool),
    }])

    # Reorder columns as required by trained model
    model_order = [
        'number_of_bedrooms', 'living_area (m²)', 'equiped_kitchen (yes:1, no:0)',
        'furnished (yes:1, no:0)', 'open_fire (yes:1, no:0)', 'terrace (yes:1, no:0)',
        'terrace_area (m²)', 'garden (yes:1, no:0)', 'number_facades',
        'swimming_pool (yes:1, no:0)', 'state_of_building', 'type', 'subtype', 'province'
    ]

    input_df = input_df[model_order]

    # Try model prediction
    try:
        prediction = model.predict(input_df)[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    return {
        "predicted_price": float(prediction),
        "status": "success"
    }


