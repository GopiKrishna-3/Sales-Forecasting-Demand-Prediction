import streamlit as st
import pickle
import pandas as pd

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="Sales Forecasting System",
    page_icon="📈",
    layout="centered"
)

# ---------------- Custom Styling ----------------
st.markdown("""
<style>
.main {
    background-color: #f7f9fc;
}
.title {
    font-size: 36px;
    font-weight: 700;
    color: #2e5e50;
    text-align: center;
}
.subtitle {
    font-size: 18px;
    color: #555;
    text-align: center;
    margin-bottom: 30px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- Load Model Safely ----------------
@st.cache_resource
def load_model():
    try:
        with open("sales_model.pkl", "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error("❌ Failed to load model")
        st.write(e)
        st.stop()

model = load_model()

# ---------------- Title ----------------
st.markdown("<div class='title'>Sales Forecasting System</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Machine Learning Based Demand Prediction</div>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- USER INPUT ----------------
st.subheader("📥 User Input")

days = st.number_input(
    "Enter number of days to forecast",
    min_value=1,
    max_value=365,
    value=60,
    step=1
)

# ---------------- Prediction ----------------
if st.button("🔮 Predict Sales"):

    future = model.make_future_dataframe(periods=int(days))
    forecast = model.predict(future)
    forecast_future = forecast[['ds', 'yhat']].tail(int(days))

    st.success("Prediction completed successfully!")

    # ---------------- Plot ----------------
    st.subheader("📈 Forecasted Sales")

    st.subheader("📈 Forecasted Sales")

    chart_data = forecast_future.set_index("ds")[["yhat"]]
    st.line_chart(chart_data)

    # ---------------- Table ----------------
    st.subheader("📊 Forecasted Values")
    st.dataframe(forecast_future, use_container_width=True)

# ---------------- Footer ----------------
st.markdown("---")
st.write("📌 Deployed using Streamlit | ML Model: Prophet")
