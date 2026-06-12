import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

try:
    from streamlit_extras.metric_cards import style_metric_cards
except ModuleNotFoundError:
    style_metric_cards = None


# -----------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------
st.set_page_config(
    page_title="CreditLend - Digital Credit Scoring",
    page_icon="💳",
    layout="wide"
)


# -----------------------------------------------------------
# STYLING
# -----------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #e3f2fd 0%, #e8f5e9 100%);
        font-family: 'Poppins', sans-serif;
    }

    h1, h2, h3, h4 {
        color: #0d47a1;
        font-weight: 600;
    }

    div.stButton > button {
        background: linear-gradient(90deg, #007bff, #00bfa5);
        color: white;
        border-radius: 12px;
        font-size: 17px;
        height: 3em;
        width: 100%;
        font-weight: 600;
        transition: 0.3s;
        border: none;
    }

    div.stButton > button:hover {
        transform: scale(1.02);
        background: linear-gradient(90deg, #0056b3, #009688);
    }

    section[data-testid="stSidebar"] {
        background: #e3f2fd;
        border-right: 2px solid #bbdefb;
    }

    div[data-testid="stMetricValue"] {
        color: #1b5e20;
        font-weight: 700;
        font-size: 28px;
    }

    .credit-card {
        background: rgba(255,255,255,0.85);
        border-radius: 15px;
        padding: 20px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        margin-bottom: 25px;
        transition: all 0.2s ease-in-out;
    }

    .credit-card:hover {
        transform: scale(1.01);
        box-shadow: 0px 6px 15px rgba(0,0,0,0.15);
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------------------------------------
# HEADER
# -----------------------------------------------------------
st.markdown(
    """
    <div style='background:linear-gradient(90deg,#0d47a1,#00796b);
                padding:15px;border-radius:12px;margin-bottom:15px'>
        <h2 style='color:white;text-align:center;'>
            CreditLend - Credit Scoring for Rural Finance
        </h2>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/4836/4836990.png", width=100)
st.sidebar.title("CreditLend Portal")
st.sidebar.markdown("---")
st.sidebar.info(
    "This dashboard predicts borrower repayment probability using demographic, "
    "financial, agricultural, business, and digital transaction features."
)


# -----------------------------------------------------------
# LOAD MODEL ASSETS
# -----------------------------------------------------------
@st.cache_resource
def load_assets():
    model = joblib.load("models/xgb_model.pkl")
    ct = joblib.load("models/column_transformer.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, ct, scaler


model, ct, scaler = load_assets()


# -----------------------------------------------------------
# INPUT SECTION
# -----------------------------------------------------------
st.header("📋 Borrower Information")
st.markdown('<div class="credit-card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 70, 35)
    gender = st.selectbox("Gender", ["Male", "Female"])

    education = st.selectbox(
        "Education Level",
        ["None", "Primary", "Secondary", "Graduate"]
    )

    occupation = st.selectbox(
        "Occupation Type",
        ["Farmer", "Shopkeeper", "Daily Wage", "Service Provider", "Dairy"]
    )

    state = st.selectbox(
        "State",
        [
            "Maharashtra",
            "Andhra Pradesh",
            "Telangana",
            "Tamil Nadu",
            "Kerala",
            "Karnataka",
            "Madhya Pradesh",
            "Gujarat",
            "Odisha",
            "Bihar",
            "Uttar Pradesh",
        ]
    )

with col2:
    family_members = st.number_input("Family Members", 1, 10, 4)
    dependents = st.number_input("Dependents", 0, 8, 2)
    annual_income = st.number_input("Annual Income (₹)", 10000, 500000, 120000)
    mobile_txn = st.slider("Monthly Digital Transactions", 0, 100, 20)
    previous_loans = st.slider("Previous Loans Taken", 0, 5, 1)
    previous_defaults = st.selectbox("Any Previous Default?", [0, 1])

st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------
# OCCUPATION-SPECIFIC INPUTS
# -----------------------------------------------------------
if occupation == "Farmer":
    st.subheader("🌾 Agricultural Details")
    st.markdown('<div class="credit-card">', unsafe_allow_html=True)

    land_size = st.number_input("Land Size (acres)", 0.0, 20.0, 2.0)

    crop = st.selectbox(
        "Crop Type",
        [
            "Rice",
            "Wheat",
            "Sugarcane",
            "Cotton",
            "Maize",
            "Pulses",
            "Millets",
            "Groundnut",
        ]
    )

    rainfall = st.number_input("Average Annual Rainfall (mm)", 200, 2000, 900)
    irrigation = st.selectbox("Irrigation Type", ["Rain-fed", "Canal", "Borewell"])
    soil_quality = st.slider("Soil Quality Index (1–100)", 1, 100, 70)
    yield_tonnes = st.number_input("Annual Crop Yield (tonnes)", 0.0, 50.0, 3.0)

    shop_revenue = 0
    business_expense = 0
    inventory_value = 0

    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.subheader("🏪 Business / Non-Farm Details")
    st.markdown('<div class="credit-card">', unsafe_allow_html=True)

    land_size = 0
    crop = "None"
    rainfall = 0
    irrigation = "NA"
    soil_quality = 0
    yield_tonnes = 0

    shop_revenue = st.number_input("Monthly Shop Revenue (₹)", 0, 500000, 100000)
    business_expense = st.number_input("Monthly Business Expense (₹)", 0, 400000, 50000)
    inventory_value = st.number_input("Inventory Value (₹)", 0, 200000, 30000)

    st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------
# LOAN DETAILS
# -----------------------------------------------------------
st.subheader("💰 Loan Details")
loan_amount = st.number_input("Requested Loan Amount (₹)", 10000, 500000, 50000)


# -----------------------------------------------------------
# PREDICTION
# -----------------------------------------------------------
if st.button("🚀 Generate Credit Report"):
    user_df = pd.DataFrame(
        [
            {
                "age": age,
                "gender": gender,
                "education_level": education,
                "village_state": state,
                "occupation_type": occupation,
                "family_members": family_members,
                "dependents": dependents,
                "land_size_acres": land_size,
                "crop_type": crop,
                "rainfall_mm": rainfall,
                "irrigation_type": irrigation,
                "soil_quality_index": soil_quality,
                "annual_yield_tonnes": yield_tonnes,
                "shop_monthly_revenue": shop_revenue,
                "business_expenses": business_expense,
                "inventory_value": inventory_value,
                "annual_income": annual_income,
                "mobile_transactions_per_month": mobile_txn,
                "loan_amount_requested": loan_amount,
                "previous_loans": previous_loans,
                "previous_defaults": previous_defaults,
            }
        ]
    )

    X_transformed = ct.transform(user_df)
    X_input = scaler.transform(X_transformed)

    prob = model.predict_proba(X_input)[0][1]
    credit_score = int(300 + prob * 600)

    if prob >= 0.75:
        risk = "🟢 Low Risk"
        color = "green"
        recommendation = "The borrower shows strong repayment potential."
    elif prob >= 0.45:
        risk = "🟡 Medium Risk"
        color = "orange"
        recommendation = "The borrower may be considered with additional verification."
    else:
        risk = "🔴 High Risk"
        color = "red"
        recommendation = "The borrower shows higher repayment risk and needs careful review."

    # -------------------------------------------------------
    # DASHBOARD
    # -------------------------------------------------------
    st.markdown("### 📊 Credit Evaluation Dashboard")

    colA, colB, colC = st.columns(3)
    colA.metric("Repayment Probability", f"{prob * 100:.2f}%")
    colB.metric("Credit Score", f"{credit_score}/900")
    colC.metric("Risk Category", risk)

    if style_metric_cards is not None:
        style_metric_cards(
            background_color="#f1f8e9",
            border_left_color=color,
            border_color=color
        )

    st.progress(float(prob))

    # -------------------------------------------------------
    # FEATURE IMPORTANCE
    # -------------------------------------------------------
    st.markdown("### 🔍 Top Factors Influencing Credit Score")

    feature_importances = model.feature_importances_
    feature_names = ct.get_feature_names_out()

    importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": feature_importances,
        }
    )

    importance_df["Feature"] = (
        importance_df["Feature"]
        .str.replace("cat__", "", regex=False)
        .str.replace("num__", "", regex=False)
        .str.replace("_", " ", regex=False)
    )

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    ).head(8)

    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top Factors Influencing Repayment Prediction",
        color="Importance",
        color_continuous_scale="Tealgrn",
    )

    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, width="stretch")

    # -------------------------------------------------------
    # SUMMARY REPORT
    # -------------------------------------------------------
    st.markdown("### 🧾 Summary Credit Report")

    st.markdown(
        f"""
        <div class='credit-card'>
            <h4>📍 Borrower Location:</h4> {state}
            <h4>💼 Occupation:</h4> {occupation}
            <h4>💰 Annual Income:</h4> ₹{annual_income:,}
            <h4>🏦 Requested Loan:</h4> ₹{loan_amount:,}
            <hr>
            <h4>📈 Repayment Probability:</h4> <b>{prob * 100:.2f}%</b>
            <h4>🏅 Credit Score:</h4> <b>{credit_score}/900</b>
            <h4>⚠️ Risk Category:</h4> <b style='color:{color};'>{risk}</b>
            <h4>✅ Recommendation:</h4> {recommendation}
        </div>
        """,
        unsafe_allow_html=True
    )


# -----------------------------------------------------------
# DATASET INSIGHTS SECTION
# -----------------------------------------------------------
st.markdown("---")
st.header("📍 Dataset Insights")

try:
    df = pd.read_csv("data/rural_credit_dataset_mixed.csv")

    if "village_state" in df.columns and "loan_repayment_status" in df.columns:
        state_avg = (
            df.groupby("village_state")["loan_repayment_status"]
            .mean()
            .reset_index()
        )

        state_avg.columns = ["State", "Average Repayment Rate"]

        fig_state = px.bar(
            state_avg,
            x="State",
            y="Average Repayment Rate",
            color="Average Repayment Rate",
            color_continuous_scale="RdYlGn",
            title="State-wise Average Loan Repayment Rate",
        )

        fig_state.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_state, width="stretch")

    else:
        st.warning(
            "Dataset columns not found. Expected 'village_state' and "
            "'loan_repayment_status'."
        )

except FileNotFoundError:
    st.warning("Dataset not found. Make sure the CSV file exists inside the data/ folder.")