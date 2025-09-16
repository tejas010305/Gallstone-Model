import streamlit as st
import pickle
import pandas as pd
import numpy as np

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Prediction", "About", "Team"])
st.sidebar.markdown("---")

# --- Sidebar Theme Switcher ---
st.sidebar.title("Theme Settings")
theme_type = st.sidebar.radio("Select Theme", ["Professional", "Playful"], index=0)  # Default Professional
theme_mode = st.sidebar.radio("Mode", ["Light", "Dark"], index=0)  # Default Light

# --- Theme CSS ---
professional_light = """<style>
body { background-color: #F7FBFF; color: #1A2B48; font-family: 'Helvetica Neue', sans-serif; }
h1 { color: #0077B6; border-bottom: 2px solid #90E0EF; text-align: center; padding-bottom: 10px; }
.stButton > button { background-color: #0077B6; color: #fff; border-radius: 8px; }
.stButton > button:hover { background-color: #0096C7; }
</style>"""

professional_dark = """<style>
body { background-color: #0A192F; color: #E6F1FF; font-family: 'Helvetica Neue', sans-serif; }
h1 { color: #64FFDA; border-bottom: 2px solid #64FFDA; text-align: center; padding-bottom: 10px; }
.stButton > button { background-color: #64FFDA; color: #0A192F; border-radius: 8px; font-weight: bold; }
.stButton > button:hover { background-color: #4ECDC4; }
</style>"""

playful_light = """<style>
body { background: linear-gradient(135deg,#FDEBEB,#E3FDFD); color: #2B2D42; font-family: 'Poppins',sans-serif; }
h1 { background: linear-gradient(90deg,#FF6B6B,#6C63FF); -webkit-background-clip:text; -webkit-text-fill-color:transparent; text-align:center; }
.stButton > button { background: linear-gradient(90deg,#6C63FF,#48C6EF); color:#fff; border-radius:25px; }
.stButton > button:hover { transform:scale(1.05); }
</style>"""

playful_dark = """<style>
body { background: linear-gradient(135deg,#2B2D42,#1A1A2E); color: #EDF2F4; font-family: 'Poppins',sans-serif; }
h1 { background: linear-gradient(90deg,#FF6B6B,#FFD166); -webkit-background-clip:text; -webkit-text-fill-color:transparent; text-align:center; }
.stButton > button { background: linear-gradient(90deg,#FF6B6B,#FFD166); color:#fff; border-radius:25px; }
.stButton > button:hover { transform:scale(1.05); box-shadow:0 0 10px rgba(255,255,255,0.3); }
</style>"""

# --- Apply selected theme ---
if theme_type == "Professional" and theme_mode == "Light":
    st.markdown(professional_light, unsafe_allow_html=True)
elif theme_type == "Professional" and theme_mode == "Dark":
    st.markdown(professional_dark, unsafe_allow_html=True)
elif theme_type == "Playful" and theme_mode == "Light":
    st.markdown(playful_light, unsafe_allow_html=True)
else:
    st.markdown(playful_dark, unsafe_allow_html=True)

# --- Load the model ---
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

    # Dynamically create input fields
    user_input = {}
    col1, col2 = st.columns(2)
    for i, feature in enumerate(feature_names):
        if i % 2 == 0:
            with col1:
                user_input[feature] = create_input_field(feature)
        else:
            with col2:
                user_input[feature] = create_input_field(feature)

    # Convert categorical inputs to numerical
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
    st.header("Tejas Narkhede")
    st.markdown("""
    **Data Scientist**  
    **Email:** [tejasnarkhede03@gmail.com](mailto:tejasnarkhede03@gmail.com)
    """)
    st.write("Tejas is the data scientist responsible for developing and implementing the machine learning model used in this application.")
