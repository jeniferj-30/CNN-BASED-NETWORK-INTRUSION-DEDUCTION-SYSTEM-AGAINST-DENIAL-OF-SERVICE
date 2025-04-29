import streamlit as st
import pandas as pd
import numpy as np
import joblib
from streamlit_option_menu import option_menu

# Set page configuration
st.set_page_config(
    page_title="CNN Based Network Intrusion Deduction System against DOS",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Attack classification information
attack_info = {
    "Normal": "No intrusion detected. Network is safe.",
    "DoS": "Denial of Service attack detected. Prevent unauthorized access to network.",
    "Probe": "Port scanning detected. Monitor network traffic closely.",
    "R2L": "Remote to Local attack detected. Restrict external access to the network.",
    "U2R": "User to Root attack detected. Review system logs for suspicious activity."
}

# Map numeric indices to attack types
index_to_attack = {
    0: "Normal",
    1: "DoS",
    2: "Probe", 
    3: "R2L", 
    4: "U2R"
}

# Attack colors for visual identification
attack_colors = {
    "Normal": "#4CAF50",  # Green
    "DoS": "#F44336",     # Red
    "Probe": "#FFC107",   # Yellow
    "R2L": "#FF9800",     # Orange
    "U2R": "#9C27B0"      # Purple
}

# Custom CSS to make the app look better
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #26A69A;
        font-weight: 600;
        margin-top: 1rem;
    }
    .card {
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        background-color: #FFFFFF;
    }
    .info-text {
        color: #616161;
        font-size: 1.1rem;
    }
    .highlight {
        color: #D81B60;
        font-weight: 600;
    }
    .feature-box {
        background-color: #f1f8e9;
        padding: 1rem;
        border-radius: 5px;
        margin-bottom: 0.5rem;
    }
    .alert-box {
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
        font-weight: 500;
    }
    .alert-normal {
        background-color: #E8F5E9;
        border-left: 5px solid #4CAF50;
        color: #2E7D32;
    }
    .alert-dos {
        background-color: #FFEBEE;
        border-left: 5px solid #F44336;
        color: #C62828;
    }
    .alert-probe {
        background-color: #FFF8E1;
        border-left: 5px solid #FFC107;
        color: #FF8F00;
    }
    .alert-r2l {
        background-color: #FFF3E0;
        border-left: 5px solid #FF9800;
        color: #E65100;
    }
    .alert-u2r {
        background-color: #F3E5F5;
        border-left: 5px solid #9C27B0;
        color: #6A1B9A;
    }
</style>
""", unsafe_allow_html=True)

# Load the model and scaler
@st.cache_resource
def load_models():
    try:
        scaler = joblib.load('scaler_top7.pkl')
        model = joblib.load('model_top7.pkl')
        return scaler, model
    except FileNotFoundError:
        st.error("Model or scaler files not found. Please make sure 'scaler_top7.pkl' and 'model_top7.pkl' exist in the current directory.")
        return None, None

# Function to get alert class based on attack type
def get_alert_class(attack_type):
    if attack_type == "Normal":
        return "alert-normal"
    elif attack_type == "DoS":
        return "alert-dos"
    elif attack_type == "Probe":
        return "alert-probe"
    elif attack_type == "R2L":
        return "alert-r2l"
    elif attack_type == "U2R":
        return "alert-u2r"
    else:
        return "alert-normal"

# Load model and scaler
scaler, model = load_models()

# Prediction page
st.markdown('<div class="main-header">CNN Based Network Intrusion Deduction System against DOS</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Make Predictions</div>', unsafe_allow_html=True)

if model is not None and scaler is not None:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Enter Feature Values")

    # Create input fields for each feature
    col1, col2 = st.columns(2)

    with col1:
        packets_matched = st.number_input("Packets Matched", min_value=37, value=1012438)
        packets_looked_up = st.number_input("Packets Looked Up", min_value=87, value=1012574)
        port_alive_duration = st.number_input("Port alive Duration (S)", min_value=26, value=3297)
        sent_packets = st.number_input("Sent Packets", min_value=41, value=421414)

    with col2:
        received_bytes = st.number_input("Received Bytes", min_value=856, value=222094697)
        sent_bytes = st.number_input("Sent Bytes", min_value=5899, value=232922967)
        received_packets = st.number_input("Received Packets", min_value=10, value=352740)

    # Create a dataframe with the input values
    input_data = pd.DataFrame({
        'Packets Matched': [packets_matched],
        'Packets Looked Up': [packets_looked_up],
        'Port alive Duration (S)': [port_alive_duration],
        'Sent Packets': [sent_packets],
        'Received Bytes': [received_bytes],
        'Sent Bytes': [sent_bytes],
        'Received Packets': [received_packets]
    })

    # Make prediction
    if st.button("Predict", key="predict_button", help="Click to make a prediction"):
        # Scale input data
        input_scaled = scaler.transform(input_data)

        # Get prediction
        prediction = model.predict(input_scaled)

        # Handle prediction probabilities if available
        try:
            prediction_proba = model.predict_proba(input_scaled)
        except:
            # For models that don't have predict_proba (like some neural networks)
            # Create a dummy prediction_proba with 1.0 for the predicted class
            num_classes = len(attack_info)
            prediction_proba = np.zeros((1, num_classes))
            prediction_proba[0, prediction[0]] = 1.0

        # Get predicted attack type based on prediction format
        if isinstance(prediction[0], (int, np.integer)):
            # If numeric, map to attack type using our mapping dictionary
            predicted_attack = index_to_attack.get(prediction[0], "Unknown")
        else:
            # If already a string/class name
            predicted_attack = prediction[0]

        st.markdown('</div>', unsafe_allow_html=True)

        # Display prediction result
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Prediction Result")

        # Get alert class and description
        alert_class = get_alert_class(predicted_attack)
        attack_description = attack_info.get(predicted_attack, "Unknown attack type detected.")

        # Display prediction with colored box matching the attack type
        st.markdown(f"""
        <div class="{alert_class}" style="padding: 25px; border-radius: 10px; margin-bottom: 20px;">
            <h2 style="margin-top: 0;">Detected: {predicted_attack}</h2>
            <p style="font-size: 1.2rem;">{attack_description}</p>
        </div>
        """, unsafe_allow_html=True)

        # Display recommendations based on attack type
        st.subheader("Recommendations")
        if predicted_attack == "Normal":
            st.success("✅ No action needed. Continue monitoring network traffic.")
        elif predicted_attack == "DoS":
            st.error("🛑 Immediate action required:")
            st.markdown("""
            - Activate DoS protection mechanisms
            - Implement rate limiting
            - Filter traffic from attacking IPs
            - Increase server resources if possible
            """)
        elif predicted_attack == "Probe":
            st.warning("⚠ Recommended actions:")
            st.markdown("""
            - Review firewall rules
            - Close unnecessary open ports
            - Update intrusion detection systems
            - Monitor for follow-up attacks
            """)
        elif predicted_attack == "R2L":
            st.warning("⚠ Recommended actions:")
            st.markdown("""
            - Verify authentication systems
            - Check for unauthorized access
            - Review remote access policies
            - Implement multi-factor authentication
            """)
        elif predicted_attack == "U2R":
            st.error("🛑 Critical - immediate action required:")
            st.markdown("""
            - Check for privilege escalation
            - Review all administrator accounts
            - Audit recent system changes
            - Consider system isolation until resolved
            """)