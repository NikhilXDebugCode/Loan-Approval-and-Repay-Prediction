# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# -------------------------
MODEL_PATH = r"C:\Users\user\Downloads\capStoneProject\Logistic_regression_model.pkl"
REPAYMENT_MODEL_PATH = r"C:\Users\user\Downloads\capStoneProject\loan_repayment_model.pkl"
# -------------------------

st.set_page_config(page_title="Loan Approval Predictor", page_icon="💳", layout="centered")
st.title("Loan Approval Predictor")

# Load model
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found at: {MODEL_PATH}")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
    st.success("Loan Approval Model Loaded.")
except Exception as e:
    st.error(f"Failed to load loan approval model: {e}")
    st.stop()

st.markdown("---")
st.subheader("Enter applicant details here ")

# --- Inputs (UI) ---
col1, col2 = st.columns(2)

with col1:
    Gender = st.selectbox("Gender", ["Male", "Female"], index=0)
    Married = st.selectbox("Married", ["No", "Yes"], index=1)
    Education = st.selectbox("Education", ["Graduate", "Not Graduate"], index=0)
    Self_Employed = st.selectbox("Self Employed", ["No", "Yes"], index=0)
    ApplicantIncome = st.number_input("Applicant Monthly Income", min_value=0.0, value=5000.0, step=100.0)

with col2:
    CoapplicantIncome = st.number_input("Coapplicant Monthly Income", min_value=0.0, value=0.0, step=100.0)
    LoanAmount = st.number_input("Loan Amount (in thousands)", min_value=1.0, value=100.0, step=1.0)
    Loan_Amount_Term = st.number_input("Loan Term (in months)", min_value=60.0, value=360.0, step=12.0)
    Credit_History = st.selectbox("Credit History (1.0 = good)", [1.0, 0.0], index=0)
    Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"], index=0)

Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"], index=0)

st.markdown("---")

# -------------------------
# Preprocess to match train
# -------------------------
def build_row():
    map_gender = {"Male": 1, "Female": 0}
    map_yesno = {"Yes": 1, "No": 0}
    map_education = {"Graduate": 1, "Not Graduate": 0}
    map_property = {"Urban": 2, "Semiurban": 1, "Rural": 0}

    dep_val = 3 if Dependents == "3+" else int(Dependents)

    row = {
        "Gender": map_gender[Gender],
        "Married": map_yesno[Married],
        "Education": map_education[Education],
        "Self_Employed": map_yesno[Self_Employed],
        "ApplicantIncome": float(ApplicantIncome),
        "CoapplicantIncome": float(CoapplicantIncome),
        "LoanAmount": float(LoanAmount),
        "Loan_Amount_Term": float(Loan_Amount_Term),
        "Credit_History": float(Credit_History),
        "Property_Area": map_property[Property_Area],
        "Dependents": int(dep_val)
    }

    cols_in_order = [
        "Gender","Married","Education","Self_Employed","ApplicantIncome",
        "CoapplicantIncome","LoanAmount","Loan_Amount_Term","Credit_History",
        "Property_Area","Dependents"
    ]
    return pd.DataFrame([row], columns=cols_in_order)

# Preview
input_df = build_row()
st.markdown("### Input preview (exact order & names)")
st.dataframe(input_df)

# -------------------------
# Loan Approval Prediction
# -------------------------
if st.button("Predict Approval"):
    try:
        pred = model.predict(input_df)
        prob = model.predict_proba(input_df)[0][1] * 100
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    if int(pred[0]) == 1:
        st.success(f"✅ Loan Approved — probability: {prob:.2f}%")
    else:
        st.error(f"❌ Loan Not Approved — probability: {prob:.2f}%")

    with st.expander("Raw model output"):
        st.write("model.predict ->", pred.tolist())
        st.write("model.predict_proba ->", prob)


# =========================================================
# 🔽🔽🔽  BELOW IS THE LOAN REPAYMENT SECTION (ADDED ONLY) 🔽🔽🔽
# =========================================================

st.markdown("---")
st.subheader("Loan Repayment Prediction")

# Load repayment model
if not os.path.exists(REPAYMENT_MODEL_PATH):
    st.error(f"Repayment model not found at: {REPAYMENT_MODEL_PATH}")
else:
    try:
        repay_model = joblib.load(REPAYMENT_MODEL_PATH)
        st.success("Repayment Model Loaded.")
    except Exception as e:
        st.error(f"Failed to load repayment model: {e}")
        repay_model = None

# -------------------------
# Repayment Inputs
# -------------------------
colA, colB = st.columns(2)
with colA:
    Credit_Score = st.slider("Credit Score", 300, 900, 700)
    Past_Defaults = st.number_input("Past Defaults", min_value=0, max_value=10, value=0)
    Active_Loans = st.number_input("Active Loans", min_value=0, max_value=10, value=0)
    Employment_Years = st.number_input("Employment Years", min_value=0, max_value=40, value=3)

with colB:
    Debt_To_Income_Ratio = st.slider("Debt to Income Ratio", 0.0, 1.0, 0.25, 0.01)
    Credit_Utilization = st.slider("Credit Utilization", 0.0, 1.0, 0.40, 0.01)
    EMI_to_Income_Ratio = st.slider("EMI to Income Ratio", 0.0, 1.0, 0.18, 0.01)
    Missed_Payments_Last_Year = st.number_input("Missed Payments (last year)", 0, 12, 0)

# Build repayment row
def build_repayment_row():
    LoanAmount_vs_Income = LoanAmount / (ApplicantIncome if ApplicantIncome > 0 else 1)

    row = {
        "Credit_Score": Credit_Score,
        "Past_Defaults": Past_Defaults,
        "Debt_To_Income_Ratio": Debt_To_Income_Ratio,
        "Active_Loans": Active_Loans,
        "Credit_Utilization": Credit_Utilization,
        "EMI_to_Income_Ratio": EMI_to_Income_Ratio,
        "Missed_Payments_Last_Year": Missed_Payments_Last_Year,
        "Employment_Years": Employment_Years,
        "Avg_Monthly_Balance": ApplicantIncome * 1.5,
        "LoanAmount_vs_Income": LoanAmount_vs_Income
    }
    cols = ["Credit_Score","Past_Defaults","Debt_To_Income_Ratio","Active_Loans",
            "Credit_Utilization","EMI_to_Income_Ratio","Missed_Payments_Last_Year",
            "Employment_Years","Avg_Monthly_Balance","LoanAmount_vs_Income"]
    return pd.DataFrame([row], columns=cols)

repayment_df = build_repayment_row()
st.write("Repayment model input preview:")
st.dataframe(repayment_df)

# Predict repayment
if st.button("Predict Repayment"):
    if repay_model is None:
        st.error("Repayment model not loaded.")
    else:
        try:
            rpred = repay_model.predict(repayment_df)[0]
            rprob = repay_model.predict_proba(repayment_df)[0][1] * 100
        except Exception as e:
            st.error(f"Repayment prediction error: {e}")
            st.stop()

        if int(rpred) == 1:
            st.success(f"🟢 Likely to REPAY — probability: {rprob:.2f}%")
        else:
            st.error(f"🔴 Likely to DEFAULT — probability (repay): {rprob:.2f}%")

        with st.expander("Raw repayment output"):
            st.write("predict ->", rpred)
            st.write("predict_proba ->", rprob)