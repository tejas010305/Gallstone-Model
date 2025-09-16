import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Apply custom CSS for a beautiful dark theme
st.markdown("""
<style>
    /* Overall Page Styling with a new dark blue palette */
    body {
        background-color: #0A192F;
        color: #E6F1FF;
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar and Main Content */
    .main .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
    }

    /* Header and Titles */
    h1 {
        color: #64FFDA; /* A vibrant accent color */
        font-weight: 700;
        text-align: center;
        border-bottom: 2px solid #64FFDA;
        padding-bottom: 10px;
    }

    h2 {
        color: #CCD6F6;
        font-weight: 600;
    }

    /* Input Fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background-color: #1A283C;
        color: #E6F1FF;
        border: 1px solid #303C52;
        border-radius: 8px;
        padding: 12px 18px;
        transition: border 0.3s;
    }
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus {
        border-color: #64FFDA;
    }

    /* Selectbox */
    .stSelectbox > div > div > div {
        background-color: #1A283C;
        color: #E6F1FF;
        border: 1px solid #303C52;
        border-radius: 8px;
    }

    /* Buttons */
    .stButton > button {
        background-color: #64FFDA;
        color: #0A192F;
        font-weight: bold;
        border-radius: 8px;
        padding: 12px 25px;
        border: none;
        transition: background-color 0.3s;
    }
    .stButton > button:hover {
        background-color: #4ECDC4;
    }

    /* Info Box for Prediction Result */
    div[data-baseweb="alert"] {
        background-color: #1A283C !important;
        color: #E6F1FF !important;
        border-left: 5px solid #64FFDA !important;
        border-radius: 8px !important;
        margin-top: 20px;
    }

    /* Markdown Separator */
    hr {
        border-top: 2px solid #303C52;
    }

    /* General Text and Disclaimer */
    .stMarkdown p {
        color: #A8B2D1;
    }

    /* Sidebar Styling for the Nav Bar */
    .css-1d391kg {
        background-color: #0A192F;
        border-right: 1px solid #303C52;
    }
    .css-1d391kg .stRadio > label {
        color: #A8B2D1;
        font-size: 1.1rem;
        font-weight: 500;
    }
    .css-1d391kg .stRadio > label:hover {
        color: #64FFDA;
    }
</style>
""", unsafe_allow_html=True)

# Load the model
@st.cache_resource
def load_model():
    try:
        with open('Gallstone_Model.pkl', 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error("The model file 'Gallstone_Model.pkl' was not found. Please make sure it's in the same directory.")
        st.stop()

model = load_model()

# Get the feature names from the model
try:
    feature_names = model.feature_names_in_
except AttributeError:
    st.error("Could not retrieve feature names from the model. The model might not be compatible.")
    st.stop()

# --- Navigation Bar ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Prediction", "About", "Team"])
st.sidebar.markdown("---")

# --- Page Logic ---
if page == "Prediction":
    st.title("Gallstone Prediction App")
    st.write("This application predicts the likelihood of having gallstones based on various health indicators.")
    st.markdown("---")
    st.header("Patient Information")

    # Helper function to create input fields
    def create_input_field(feature_name):
        if 'Gender' in feature_name or 'Comorbidity' in feature_name or 'Disease' in feature_name:
            options = ['No', 'Yes']
            return st.selectbox(f"Select {feature_name}:", options=options, index=options.index('No'))
        else:
            return st.number_input(f"Enter {feature_name}:", value=0.0, step=0.01)

    # Dynamically create input fields based on model features
    user_input = {}
    col1, col2 = st.columns(2)
    for i, feature in enumerate(feature_names):
        if i % 2 == 0:
            with col1:
                user_input[feature] = create_input_field(feature)
        else:
            with col2:
                user_input[feature] = create_input_field(feature)

    # Convert categorical inputs to numerical (0 or 1)
    def process_input(input_dict):
        processed_dict = input_dict.copy()
        for key, value in processed_dict.items():
            if value == 'Yes':
                processed_dict[key] = 1.0
            elif value == 'No':
                processed_dict[key] = 0.0
        return processed_dict

    processed_input = process_input(user_input)

    # Prediction button
    if st.button("Predict Gallstone Likelihood"):
        input_df = pd.DataFrame([processed_input])
        input_df = input_df[feature_names]
        input_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        input_df.fillna(0, inplace=True)

        try:
            prediction_proba = model.predict_proba(input_df)
            predicted_class_index = np.argmax(prediction_proba, axis=1)[0]
            predicted_class = model.classes_[predicted_class_index]

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.stop()

        st.markdown("---")
        st.header("Prediction Results")
        st.info(f"The model predicts the likelihood of having gallstones is: **{predicted_class}**")

        if 'Yes' in model.classes_ and 'No' in model.classes_:
            yes_index = list(model.classes_).index('Yes')
            no_index = list(model.classes_).index('No')
            yes_prob = prediction_proba[0, yes_index] * 100
            no_prob = prediction_proba[0, no_index] * 100
        else:
            yes_prob = prediction_proba[0, 1] * 100
            no_prob = prediction_proba[0, 0] * 100

        st.write(f"Confidence (Yes): {yes_prob:.2f}%")
        st.write(f"Confidence (No): {no_prob:.2f}%")

        st.markdown("""
        *Disclaimer: This is a machine learning model prediction and should not be used as a substitute for professional medical advice. Please consult a healthcare professional for diagnosis and treatment.*
        """)

elif page == "About":
    st.title("About This Application")
    st.markdown("---")
    st.write("""
    This application is an educational tool developed to demonstrate the use of machine learning in healthcare. 
    It uses a **Logistic Regression model** trained on a comprehensive dataset of patient health metrics to predict the presence of gallstones.
    """)
    st.subheader("How It Works")
    st.write("""
    The model analyzes various features, including demographic information, body composition data, and laboratory results, to make a prediction.
    The prediction is presented along with a confidence score, indicating the model's certainty.
    """)
    st.subheader("Disclaimer")
    st.warning("""
    **This tool is for informational purposes only.** It is not a diagnostic tool and should not be used to replace professional medical advice, diagnosis, or treatment. 
    Always seek the advice of a qualified healthcare provider with any questions you may have regarding a medical condition.
    """)

elif page == "Team":
    st.title("Meet the Team")
    st.markdown("---")
    st.header("Maruti Margale")
    st.markdown("""
    **Data Scientist** 
    **Email:** [margalemaruti3@gmail.com](mailto:margalemaruti3@gmail.com)
    """)
    st.write("Maruti is the data scientist responsible for developing and implementing the machine learning model used in this application.")
