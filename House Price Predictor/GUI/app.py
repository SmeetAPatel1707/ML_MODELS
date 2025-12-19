import streamlit as st
import joblib
import numpy as np

# Load Model
model = joblib.load('house_price_model.pkl')

# Page Config
st.set_page_config(
    page_title="MacOS House Price Estimator",
    page_icon="💻",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Apply custom CSS
with open('style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# MacOS Style Header
st.markdown("""
<div class="mac-window">
    <div class="mac-buttons">
        <span class="red"></span>
        <span class="yellow"></span>
        <span class="green"></span>
    </div>
    <h1>🏠 House Price Estimator</h1>
</div>
""", unsafe_allow_html=True)

# Input Fields
st.subheader("Enter House Details")
area = st.number_input("Total Area (in sq ft):", min_value=200, max_value=10000)
bedrooms = st.slider("Number of Bedrooms:", 1, 10, 3)
bathrooms = st.slider("Number of Bathrooms:", 1, 10, 2)
location_index = st.selectbox("Location Quality Index (0 - low, 10 - high):", list(range(11)))

# Predict Button
if st.button("🔍 Predict Price"):
    input_data = np.array([[area, bedrooms, bathrooms, location_index]])
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated House Price: ₹ {prediction:,.2f}")

# Footer
st.markdown("""<div class="footer">MacOS UI styled by Smeet ✨</div>""", unsafe_allow_html=True)
