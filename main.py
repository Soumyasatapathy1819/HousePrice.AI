import streamlit as st
import requests
import pandas as pd

# 1. Page Configuration & Custom Styling
st.set_page_config(
    page_title="House Price Prediction Dashboard", 
    page_icon="🏡", 
    layout="wide"
)

# Injecting some custom CSS for a cleaner modern interface
st.markdown("""
    <style>
    .metric-box {
        background-color: #1E293B;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #334155;
        text-align: center;
    }
    div.stButton > button:first-child {
        background-color: #2563EB;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        height: 3em;
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #1D4ED8;
        transform: scale(1.02);
    }
    </style>
""", unsafe_allow_html=True)

# 2. Main App Header
st.title("🏡 HousePrice.AI")
st.markdown("### *Predictive Machine Learning Model for Real Estate Valuation*")
st.markdown("Adjust the house attributes below to generate an instant market price prediction using our trained model.")
st.write("---")

# 3. Main Dashboard Layout (Columns)
col_inputs, col_visuals = st.columns([4, 5], gap="large")

with col_inputs:
    st.subheader("🛠️ House Attributes")
    
    # Structural specs grouped in sub-columns
    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
        area = st.number_input("Total Living Area (sq ft)", min_value=100, max_value=20000, value=1800, step=50)
        bedrooms = st.slider("Number of Bedrooms", min_value=1, max_value=10, value=3)
        bathrooms = st.slider("Number of Bathrooms", min_value=1, max_value=10, value=2)
    with sub_col2:
        year_built = st.slider("Year Built", min_value=1800, max_value=2026, value=2010)
        floors = st.selectbox("Number of Floors", [1, 2, 3, 4, 5], index=0)
    
    st.markdown("#### 📍 Location & Condition Profile")
    location = st.segmented_control("Property Location", ["Downtown", "Suburban", "Rural"], default="Suburban")
    condition = st.select_slider("Overall Property Condition", options=["Poor", "Fair", "Good", "Excellent"], value="Good")
    garage = st.toggle("Includes an attached Garage?", value=True)
    garage_str = "Yes" if garage else "No"

    st.write(" ")
    predict_btn = st.button("📈 Predict House Price", use_container_width=True)

with col_visuals:
    st.subheader("📊 Price Prediction Analysis")
    
    # If button is pressed, make API call and render interactive charts
    if predict_btn:
        payload = {
            "Area": area, "Bedrooms": bedrooms, "Bathrooms": bathrooms,
            "Floors": floors, "YearBuilt": year_built, "Location": location,
            "Condition": condition, "Garage": garage_str
        }
        
        try:
            # Call our running Flask Backend API
            response = requests.post("http://127.0.0.1:5000/predict", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                predicted_price = result['price']
                
                # Dynamic KPI Cards
                st.balloons()
                st.markdown(f"""
                    <div class="metric-box">
                        <span style="font-size: 1.1rem; color: #94A3B8; font-weight: 500;">PREDICTED HOUSE VALUE</span><br>
                        <span style="font-size: 2.8rem; color: #34D399; font-weight: 800;">${predicted_price:,.2f}</span>
                    </div>
                """, unsafe_allow_html=True)
                
                st.write(" ")
                
                # Interactive Feature Breakdown Tabs
                tab1, tab2 = st.tabs(["📉 Price Comparables", "📋 Input Summary"])
                
                with tab1:
                    st.write("Estimated valuation impact of feature modifications:")
                    chart_data = pd.DataFrame({
                        "Scenario": ["Predicted Price", "With +1 Bedroom", "With +1 Bathroom"],
                        "Price ($)": [predicted_price, predicted_price * 1.08, predicted_price * 1.05]
                    })
                    st.bar_chart(chart_data, x="Scenario", y="Price ($)", color="#2563EB")
                    
                with tab2:
                    st.table(pd.DataFrame([payload]).T.rename(columns={0: "Feature Value"}))
                    
            else:
                st.error(f"Backend Error: {response.json().get('error', 'Unknown error occurred')}")
                
        except requests.exceptions.ConnectionError:
            st.error("🚨 Connection Failed! Make sure your Flask backend (`python app.py`) is running in your separate VS Code terminal tab.")
            
    else:
        st.info("💡 Adjust features on the left and click 'Predict House Price' to run the evaluation model.")