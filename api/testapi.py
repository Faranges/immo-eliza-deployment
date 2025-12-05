from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel
import pickle
import pandas as pd 


# ---------------------------
# 1. Load trained model
# ---------------------------

with open("../models/xgb_pipeline.pkl", "rb") as f:
    model = pickle.load(f)


# ---------------------------
# 2. Create FastAPI app
# ---------------------------

app = FastAPI(
    title="Immo Eliza Price Prediction API",
    description="API for predicting real estate prices"
)

# ---------------------------
# 3. Define the input format
# ---------------------------

class PropertyInput(BaseModel):
# STEP 1 — General info
    type: Literal["Apartment", "House"] = Field(..., description="Property type")
    subtype: Literal[
        "Apartment", "Residence", "Villa", "Ground floor", "Penthouse", "Duplex",
        "Mixed building", "Studio", "Chalet", "Bungalow", "Cottage", "Loft",
        "Triplex", "Mansion", "Masterhouse"
    ] = Field(..., description="Subtype of property")

    province: Literal[
        "Brussels", "Antwerp", "West-Flanders", "East-Flanders", "Flemish-Brabant",
        "Limburg", "Liège", "Brabant-Wallon", "Hainaut", "Luxembourg", "Namur"
    ] = Field(..., description="Province in Belgium")

    state_of_building: Literal[
        "To demolish", "Under construction", "To restore", "To renovate",
        "To be renovated", "Normal", "Fully renovated", "Excellent", "New"
    ] = Field(..., description="Building state")






    type: 
    subtype: 
    province: 
    state_of_building: 
    living_area: float = Field(..., ge=1, le=1000) # enforce min/max values
    number_of_bedrooms: 
    number_facades: 
    equiped_kitchen: 
    furnished:
    open_fire: 
    terrace: 
    terrace_area: 
    garden: 
    swimming_pool: 

# Mapping from clean names to raw model column names
COLUMN_MAP = {
    "living_area": "living_area (m²)",
    "terrace": "terrace (yes:1, no:0)",
    "open_fire": "open_fire (yes:1, no:0)",
    "equiped_kitchen": "equipped_kitchen (yes:1 no:0)",
    "furnished": "furnished (yes:1, no:0)",
    "open_fire": "open_fire",
    "terrace": "terrace (yes:1, no:0)",
    "terrace_area": "terrace_area (m²)",
    "garden": "garden (yes:1, no:0)",
    "swimming_pool": "swimming_pool (yes:1, no:0)"
}

# Convert clean inputs → model formats
def convert_input_to_model_format(data: PropertyInput):
    d = data.dict()

    transformed = {}

    for clean_name, model_name in COLUMN_MAP.items():
        value = d[clean_name]

        # Convert booleans to 1/0
        if isinstance(value, bool):
            value = int(value)

        # Example kitchen conversion
        if clean_name == "kitchen":
            value = 1 if value.lower() == "installed" else 0

        transformed[model_name] = value

    return transformed

# Convert into DataFrame
input_df = pd.DataFrame([convert_input_to_model_format(data)])

# ---------------------------
# 4. Alive endpoint
# ---------------------------
@app.get("/")
def alive():
    return {"status": "alive"}

# ---------------------------
# 5. Prediction endpoint
# ---------------------------
@app.post("/predict")
def predict(data: PropertyInput):

    # Convert Pydantic object → DataFrame
    input_df = pd.DataFrame([data.dict()])

    # Run model prediction
    prediction = model.predict(input_df)[0]

    # Return JSON result
    return {"predicted_price": float(prediction)}
    
        