from utils import FullXGBPipeline
from PIL import Image
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- 1. CSS INJECTION FOR THE BUTTON STYLE AND ANIMATION ---
st.markdown("""
<style>
/* CSS for the animated gradient button */
.predict-button-container {
    display: flex; /* Enables flex for centering */
    justify-content: center; /* Centers the content (the button) */
    margin-top: 20px;
    margin-bottom: 20px;
    width: 100%; /* Ensure the container spans full width to center effectively */
}

.stButton > button {
    /* Base styles for the Streamlit button element */
    width: 250px;
    height: 60px;
    font-size: 20px;
    font-weight: bold;
    color: white; /* Text color */
    border: none;
    border-radius: 12px;
    cursor: pointer;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    transition: all 0.3s ease-in-out; /* Smooth transition for hover effects */

    /* Gradient definition */
    background: linear-gradient(135deg, #FF6B6B, #FFD166, #118AB2, #073B4C);
    background-size: 400% 400%; /* Make the gradient large for smooth animation */
    
    /* Apply the movement animation */
    animation: gradientShift 10s ease infinite; 
}

/* Hover Effect: The "Jump Out" and speed up animation */
.stButton > button:hover {
    transform: scale(1.08) translateY(-3px); /* Jump out (scale and slight lift) */
    box-shadow: 0 8px 15px rgba(0, 0, 0, 0.4); /* Deeper shadow */
    animation: gradientShift 4s ease infinite; /* Speed up gradient movement on hover */
}

/* Define the slow, continuous gradient movement */
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
</style>
""", unsafe_allow_html=True)

# Load the trained model
with open("xgb_pipeline.pkl", "rb") as file:
    model = pickle.load(file)

# ----------------------------
# SIDEBAR CONTENT
# ----------------------------
with st.sidebar:
    # Load image from project folder
    image = Image.open("assets/sidebar_image_immo_eliza.png")

    # Center the image using markdown container
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.image(image, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # Centered & styled text
    st.markdown(
        """
        <div style='text-align: center; color: #1D4ED8; font-weight: bold; font-size: 16px;'>
            This application uses machine learning to predict Belgian real estate prices.
            <br><br>
            To get an instant prediction, fill in the property details.
        </div>
        """,
        unsafe_allow_html=True
    )
# Streamlit initial UI
st.set_page_config(page_title="Property Pricing Predictor", page_icon="🏡", layout="centered")
st.title("Belgian Real Estate Price Predictor")
st.write("Fill in the 14 required fields to get a price prediction of your property.")

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

    st.session_state.type = st.selectbox("Select property type", sorted(["House", "Apartment"]), 
        index=None,  # forces the user to pick something (no default)
        placeholder="Choose a property type")

    st.session_state.subtype = st.selectbox("Select property subtype", 
        sorted(["Apartment", "Residence", "Villa", "Ground floor", "Penthouse", "Duplex", "Mixed building", "Studio", "Chalet", "Bungalow", "Cottage", "Loft", "Triplex", "Mansion", "Masterhouse"]),
        index = None,
        placeholder = "Choose a property subtype")

    st.session_state.province = st.selectbox(
        "Select province",
        sorted([
            "Brussels", "Antwerp", "West-Flanders", "East-Flanders", "Flemish-Brabant",
            "Limburg", "Liège", "Brabant-Wallon", "Hainaut", "Luxembourg", "Namur"
        ]),
        index=None,
        placeholder="Choose a province")
    

    st.session_state.state_of_building = st.selectbox(
        "Select state of the building",
        sorted(["To demolish", "Under construction", "To restore", "To renovate", "To be renovated", "Normal", "Fully renovated", "Excellent", "New"]),
        index=None,
        placeholder="Choose the state of the building")
    
    if st.button("Next →"):
        st.session_state.step = 2
        st.rerun()

# ---------------------------------------------------------------
# STEP 2 of 3 — info about the inside of the property
# ---------------------------------------------------------------
elif st.session_state.step == 2:
    st.session_state.living_area = st.number_input('Living area in m²', 
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
        "Does the property have a an open fireplace?",
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

    st.session_state.terrace_area = st.number_input('Terrace area in m² (Enter zero if there is no terrace) ', 
                            min_value= 0, 
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

    # --- VALIDATION: ensure all fields are filled in ---
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
    col_back, _ = st.columns([1, 5])

    with col_back:
        # Back Button logic (remains left-aligned)
        if st.button("← Back"):
            st.session_state.step = 2
            st.rerun()
        
    # Show warning if something is missing
    if not all_inputs_valid:
        st.warning("⚠️ Please fill in all fields to get a prediction.")
        st.stop()


    # --- PREDICT BUTTON IMPLEMENTATION (Centered, Below Back Button) ---
    # We use the custom CSS class 'predict-button-container' to apply centering 
    # across the full width of the Streamlit page.
    st.markdown('<div class="predict-button-container">', unsafe_allow_html=True)
    
    # This is the actual Streamlit button that triggers your Python logic. 
    if st.button("💰 Predict Price"):
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
        # Calculate 15% range
        lower = prediction * 0.85
        upper = prediction * 1.15

            # Full width thin grey horizontal line
        st.markdown("<hr style='border: 1px solid grey; margin-top:40px;'>", unsafe_allow_html=True)

        # Predicted price section
        st.markdown(f"""
            <h2 style='text-align:center; color:white;'>
                Predicted Property Price:
            </h2>

            <h1 style='text-align:center; color:#0077cc;'>
                €{prediction:,.0f}
            </h1>
        """, unsafe_allow_html=True)

        # Price range section
        st.markdown(f"""
            <h4 style='text-align:center; color:white;'>
                Estimated Price Range in Euros (€):
            </h4>

            <p style='text-align:center; font-size:20px; color:#0077cc;'>
                €{lower:,.0f} – €{upper:,.0f}
            </p>
        """, unsafe_allow_html=True)

    # Overview section using an expander
    with st.expander("Show overview of answers"):
        
        # Display all user inputs when the expander is opened
        for col in input_df.columns:
            value = input_df[col].values[0]

            st.markdown(
                f"""
                <div style='padding:8px; margin-bottom:10px;'>
                    <span style='font-weight:600; color:#ffffff; font-size:17px;'>
                        {col}
                    </span>
                    <br>
                    <span style='color:#0077cc; font-size:19px;'>
                        {value}
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

    
# To launch the streamlit app locally: run "streamlit run app.py" in terminal

