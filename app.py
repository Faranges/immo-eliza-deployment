from utils import FullXGBPipeline
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model
with open("xgb_pipeline.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit initial UI
st.set_page_config(page_title="Property Pricing Predictor", page_icon="🏡", layout="centered")
st.write("Fill in the required fields to get a price prediction of your property. There are 14 questions in total.")

# ------------------------------
# Initialize session state
# ------------------------------
if "step" not in st.session_state:
    st.session_state.step = 1

# Total steps
TOTAL_STEPS = 3

# Progress bar value
progress = st.session_state.step / TOTAL_STEPS
st.progress(progress)

st.write(f"### Step {st.session_state.step} of {TOTAL_STEPS}")

# Required Input for user (dropdown or yes/no binary features or number input)

# -----------------------------------
# STEP 1 of 3 — General property info
# -----------------------------------
if st.session_state.step == 1:

    st.session_state.type = st.selectbox("Select property type", ["House", "Apartment"], 
        index=None,  # forces the user to pick something (no default)
        placeholder="Choose a property type")

    st.session_state.subtype = st.selectbox("Select property subtype", 
        ["Apartment", "Residence", "Villa", "Ground floor", "Penthouse", "Duplex", "Mixed building", "Studio", "Chalet", "Bungalow", "Cottage", "Loft", "Triplex", "Mansion", "Masterhouse"],
        index = None,
        placeholder = "Choose a property subtype")

    st.session_state.province = st.selectbox(
        "Select province",
        ["Brussels", "Antwerp", "West-Flanders", "East-Flanders", "Flemish-Brabant" "Limburg", "Liège", "Brabant-Wallon", "Hainaut", "Luxembourg", "Namur"],
        index=None,
        placeholder="Choose a province")


    st.session_state.state_of_building = st.selectbox(
        "Select state of the building",
        ["To demolish", "Under construction", "To restore", "To renovate", "To be renovated", "Normal", "Fully renovated", "Excellent", "New"],
        index=None,
        placeholder="Choose the state of the building")
    
    if st.button("Next →"):
        st.session_state.step = 2
        st.rerun()

# ---------------------------------------------------------------
# STEP 2 of 3 — info about the inside of the property
# ---------------------------------------------------------------
elif st.session_state.step == 2:
    st.session_state.living_area = st.number_input('Living area (square meter)', 
                            min_value= 18, 
                            max_value= 2670, 
                            value= 100)
    st.session_state.number_of_bedrooms = st.number_input('Number of bedrooms', 
                            min_value= 1, 
                            max_value= 50, 
                            value= 2)

    st.session_state.has_equiped_kitchen = st.radio(
        "Does the property have an equiped kitchen?",
        options=["Yes", "No"],
        index=None  # forces selection
    )

    st.session_state.is_furnished = st.radio(
        "Is the property furnished?",
        options=["Yes", "No"],
        index=None  
    )

    st.session_state.has_open_fire = st.radio(
        "Does the property have a an open fire?",
        options=["Yes", "No"],
        index=None  
    )
    col1, col2 = st.columns(2)
    if col1.button("← Back"):
        st.session_state.step = 1
        st.rerun()
    if col2.button("Next →"):
        st.session_state.step = 3
        st.rerun()


# -----------------------------------------------------------------
# STEP 3 of 3 — info about the outside of the property + validation
# ------------------------------------------------------------------
elif st.session_state.step == 3:

    st.session_state.has_terrace = st.radio(
        "Does the property have a terrace?",
        options=["Yes", "No"],
        index=None  
    )

    st.session_state.terrace_area = st.number_input('Terrace area (square meter)', 
                            min_value= 1, 
                            max_value= 150, 
                            value= 21)
    st.session_state.has_garden = st.radio(
        "Does the property have a garden?",
        options=["Yes", "No"],
        index=None  
    )

    st.session_state.number_facades = st.number_input('Number of facades', 
                            min_value= 1, 
                            max_value= 4, 
                            value=3)
    
    st.session_state.has_swimming_pool = st.radio(
        "Does the property have a swimming pool?",
        options=["Yes", "No"],
        index=None  
    )

    # --- VALIDATION: ensure all fields exist ---
    all_inputs_valid = (
        st.session_state.type is not None and
        st.session_state.subtype is not None and
        st.session_state.province is not None and
        st.session_state.state_of_building is not None and
        st.session_state.living_area is not None and
        st.session_state.number_of_bedrooms is not None and
        st.session_state.has_equiped_kitchen is not None and
        st.session_state.is_furnished is not None and
        st.session_state.has_open_fire is not None and
        st.session_state.has_garden is not None and
        st.session_state.has_terrace is not None and
        st.session_state.terrace_area is not None and
        st.session_state.number_facades is not None and
        st.session_state.has_swimming_pool is not None
    )
    # Buttons (back + forward)
    col1, col2 = st.columns(2)

    if col1.button("← Back"):
        st.session_state.step = 2
        st.rerun()

    # Show warning if something is missing
    if not all_inputs_valid:
        st.warning("⚠️ Please fill in all fields to continue.")
        st.stop()


    # --- PREDICT ---
    if col2.button("Predict Price"):

        # Convert Yes/No to 1/0
        garden = 1 if st.session_state.has_garden == "Yes" else 0
        terrace = 1 if st.session_state.has_terrace == "Yes" else 0
        equiped_kitchen = 1 if st.session_state.has_equiped_kitchen == "Yes" else 0
        furnished = 1 if st.session_state.is_furnished == "Yes" else 0
        open_fire = 1 if st.session_state.has_open_fire == "Yes" else 0
        swimming_pool = 1 if st.session_state.has_swimming_pool == "Yes" else 0

        # Build dataframe
        input_df = pd.DataFrame([{
            "type": st.session_state.type,
            "subtype": st.session_state.subtype,
            "province": st.session_state.province,
            "state_of_building": st.session_state.state_of_building,
            "living_area (m²)": st.session_state.living_area,
            "number_of_bedrooms": st.session_state.number_of_bedrooms,
            "number_facades": st.session_state.number_facades,
            "equiped_kitchen (yes:1, no:0)": equiped_kitchen,
            "furnished (yes:1, no:0)": furnished,
            "open_fire (yes:1, no:0)": open_fire,
            "terrace (yes:1, no:0)": terrace,
            "terrace_area (m²)": st.session_state.terrace_area,
            "garden (yes:1, no:0)": garden,
            "swimming_pool (yes:1, no:0)": swimming_pool
        }])

        # Reorder input_df to match the numeric + categorical column order pipeline expects
        # The pipeline will create the _le and _oe columns internally
        numeric_and_cat_cols = [
            'number_of_bedrooms', 'living_area (m²)', 'equiped_kitchen (yes:1, no:0)',
            'furnished (yes:1, no:0)', 'open_fire (yes:1, no:0)', 'terrace (yes:1, no:0)',
            'terrace_area (m²)', 'garden (yes:1, no:0)', 'number_facades', 'swimming_pool (yes:1, no:0)',
            'state_of_building', 'type', 'subtype', 'province'
        ]
        input_df = input_df[numeric_and_cat_cols]


        # Run model prediction and show result
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted price: €{prediction:,.0f}")
    
# To launch the streamlit app locally: run "streamlit run app.py" in terminal

