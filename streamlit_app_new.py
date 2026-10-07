"""Interactive Streamlit demo for the food-delivery order-status model."""

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

st.set_page_config(page_title="Food Delivery AI", page_icon="🍽️", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "daily_food_delivery_orders.csv"
MODEL_PATH = BASE_DIR / "food_delivery_model.pkl"


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def prepare_features(data: pd.DataFrame) -> pd.DataFrame:
    """Apply exactly the feature engineering used in the notebook."""
    features = data.copy()
    features["order_date"] = pd.to_datetime(features["order_date"])
    features["order_month"] = features["order_date"].dt.month
    features["order_day_of_week"] = features["order_date"].dt.dayofweek
    features["is_weekend"] = (features["order_day_of_week"] >= 5).astype(int)
    return features.drop(columns=["order_id", "order_date"], errors="ignore")


def inject_css():
    st.markdown(
        """
        <style>
        .main { background: #f7f9fc; }
        .hero { padding: 1.5rem 1.8rem; border-radius: 18px; background: linear-gradient(135deg,#173b62,#246b8f); color: white; margin-bottom: 1rem; }
        .hero h1 { margin: 0; font-size: 2.35rem; }
        .hero p { margin: .45rem 0 0; font-size: 1.05rem; opacity: .92; }
        div[data-testid="stMetricValue"] { color: #173b62; }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()
st.markdown(
    '<div class="hero"><h1>🍽️ Food Delivery AI</h1><p>Predict the outcome of a food order and explore the patterns behind delivery performance.</p></div>',
    unsafe_allow_html=True,
)

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.info("Place daily_food_delivery_orders.csv beside this app and refresh the page.")
    st.stop()
if not MODEL_PATH.exists():
    st.error("Model file not found. Run all cells in food_delivery_ml.ipynb first.")
    st.info("The notebook creates food_delivery_model.pkl for this app.")
    st.stop()

df = load_data()
bundle = load_model()
model = bundle["model"]

with st.sidebar:
    st.header("Project controls")
    page = st.radio("Choose a view", ["Predict an order", "Explore the data", "About the project"])
    st.caption(f"Model: {bundle.get('model_name', 'Saved pipeline')}")

if page == "Predict an order":
    st.subheader("Predict order status")
    st.write("Enter an order profile below. The model returns its predicted status and the probability for every class.")
    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            order_date = st.date_input("Order date", value=pd.Timestamp("2024-06-15").date())
            customer_age = st.slider("Customer age", 18, 80, 35)
            restaurant_type = st.selectbox("Restaurant type", sorted(df["restaurant_type"].unique()))
        with c2:
            order_value = st.number_input("Order value", min_value=1.0, max_value=5000.0, value=650.0, step=10.0)
            distance = st.number_input("Delivery distance (km)", min_value=0.1, max_value=50.0, value=6.0, step=0.1)
            delivery_time = st.number_input("Estimated delivery time (minutes)", min_value=1, max_value=180, value=45, step=1)
        with c3:
            payment_method = st.selectbox("Payment method", sorted(df["payment_method"].unique()))
            rating = st.slider("Delivery partner rating", 1.0, 5.0, 4.0, step=0.1)
            submitted = st.form_submit_button("Predict order status", type="primary", use_container_width=True)

    if submitted:
        row = pd.DataFrame([{
            "order_date": pd.Timestamp(order_date), "customer_age": customer_age,
            "restaurant_type": restaurant_type, "order_value": order_value,
            "delivery_distance_km": distance, "delivery_time_minutes": delivery_time,
            "payment_method": payment_method, "delivery_partner_rating": rating,
        }])
        prepared = prepare_features(row)
        prediction = model.predict(prepared)[0]
        probabilities = model.predict_proba(prepared)[0]
        st.success(f"Predicted outcome: **{prediction}**")
        probability_df = pd.DataFrame({"Status": model.classes_, "Probability": probabilities}).sort_values("Probability", ascending=False)
        left, right = st.columns([1, 2])
        with left:
            st.metric("Top prediction", prediction)
            st.metric("Confidence", f"{probability_df.iloc[0]['Probability']:.1%}")
        with right:
            st.bar_chart(probability_df.set_index("Status"), y="Probability", color="#246b8f")

elif page == "Explore the data":
    st.subheader("Dataset overview")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Orders", f"{len(df):,}")
    k2.metric("Average order value", f"₹{df['order_value'].mean():,.0f}")
    k3.metric("Average delivery time", f"{df['delivery_time_minutes'].mean():.1f} min")
    k4.metric("Average rating", f"{df['delivery_partner_rating'].mean():.2f}/5")
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Order status distribution**")
        st.bar_chart(df["order_status"].value_counts())
    with c2:
        st.write("**Average delivery time by status**")
        st.bar_chart(df.groupby("order_status")["delivery_time_minutes"].mean().sort_values())
    st.write("**Sample records**")
    st.dataframe(df.head(15), use_container_width=True, hide_index=True)

else:
    st.subheader("About this machine learning project")
    st.markdown("""
    **Goal:** predict the final status of a food delivery order.

    **Target:** `order_status` with three classes: Delivered, Cancelled, and Delayed.

    **Workflow:** the notebook cleans the date, creates calendar features, one-hot encodes categorical columns, scales numeric features, compares Logistic Regression with Random Forest, evaluates the best pipeline, and exports it as a Joblib bundle.

    **Why a pipeline?** The same preprocessing used during training is automatically applied to new orders in this app. This prevents training-serving mismatches.

    **Important demo note:** the model is educational and should be treated as a demonstration of a supervised classification workflow, not as an operational decision system.
    """)
    st.info("Run the notebook before launching Streamlit so the saved model bundle exists.")
