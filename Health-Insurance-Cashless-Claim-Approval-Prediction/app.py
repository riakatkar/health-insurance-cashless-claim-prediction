import json
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from train_model import prepare_features

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "cashless_claim_prediction_dataset.csv"
MODEL_PATH = ROOT / "artifacts" / "cashless_claim_model.joblib"

st.set_page_config(page_title="Cashless Claim Estimator", page_icon="🏥", layout="wide")
st.title("🏥 Health Insurance Cashless Claim Estimator")
st.caption("A machine-learning demonstration for estimating a cashless claim approval amount.")
st.warning(
    "Educational prototype only. This estimate is not an insurer decision, medical advice, "
    "fraud determination, or a guarantee of reimbursement. Real approvals require policy "
    "wording, document review, clinical review and authorized human judgment."
)

if not MODEL_PATH.exists():
    st.error("Model artifact not found. From the project folder, run: python train_model.py")
    st.stop()

bundle = joblib.load(MODEL_PATH)
pipeline = bundle["pipeline"]
feature_columns = bundle["feature_columns"]
raw_data = pd.read_csv(DATA_PATH)

st.markdown("### Enter claim details")
st.write("Use the available case details. Fields are initialized with typical values from the supplied training dataset; replace them with a hypothetical claim.")

excluded = {"claim_id", "patient_id", "hospital_id", "approved_amount"}
raw_features = [c for c in raw_data.columns if c not in excluded]
categorical = raw_data[raw_features].select_dtypes(include="object").columns.tolist()
numeric = raw_data[raw_features].select_dtypes(include=np.number).columns.tolist()

with st.form("claim_form"):
    left, right = st.columns(2)
    input_values = {}
    with left:
        st.markdown("#### Patient & treatment")
        for col in ["age", "gender", "city_tier", "bmi", "smoker_flag",
                    "comorbidity_score", "prior_claim_count_2yr", "prior_claim_amount_2yr",
                    "admission_type", "treatment_type", "diagnosis_category",
                    "severity_level", "procedure_count", "procedure_complexity_score",
                    "icu_days", "length_of_stay_days", "pre_existing_disease_flag"]:
            if col not in raw_features:
                continue
            if col in categorical:
                options = sorted(raw_data[col].dropna().astype(str).unique().tolist())
                default = str(raw_data[col].mode().iloc[0]) if raw_data[col].notna().any() else options[0]
                input_values[col] = st.selectbox(col.replace("_", " ").title(), options,
                                                 index=options.index(default) if default in options else 0)
            else:
                series = pd.to_numeric(raw_data[col], errors="coerce")
                default = float(series.median()) if series.notna().any() else 0.0
                if col.endswith("_flag") or col == "smoker_flag":
                    input_values[col] = st.selectbox(col.replace("_", " ").title(), [0, 1],
                                                     index=int(round(default)) if round(default) in [0, 1] else 0)
                else:
                    input_values[col] = st.number_input(col.replace("_", " ").title(),
                                                        value=default, step=1.0 if series.dropna().mod(1).eq(0).all() else 0.1)
    with right:
        st.markdown("#### Hospital & policy")
        for col in ["hospital_type", "hospital_tier", "network_hospital_flag",
                    "hospital_historical_approval_rate", "hospital_billing_index",
                    "hospital_city_tier", "policy_type", "policy_tenure_months",
                    "sum_insured", "remaining_sum_insured", "room_rent_limit_per_day",
                    "copay_percent", "deductible_amount", "waiting_period_applicable_flag",
                    "claimed_amount", "room_charges", "doctor_fees", "medicine_charges",
                    "document_quality_score"]:
            if col not in raw_features:
                continue
            if col in categorical:
                options = sorted(raw_data[col].dropna().astype(str).unique().tolist())
                default = str(raw_data[col].mode().iloc[0]) if raw_data[col].notna().any() else options[0]
                input_values[col] = st.selectbox(col.replace("_", " ").title(), options,
                                                 index=options.index(default) if default in options else 0)
            else:
                series = pd.to_numeric(raw_data[col], errors="coerce")
                default = float(series.median()) if series.notna().any() else 0.0
                if col.endswith("_flag"):
                    input_values[col] = st.selectbox(col.replace("_", " ").title(), [0, 1],
                                                     index=int(round(default)) if round(default) in [0, 1] else 0)
                else:
                    step = 1.0 if series.dropna().mod(1).eq(0).all() else 0.1
                    input_values[col] = st.number_input(col.replace("_", " ").title(), value=default, step=step)
        if "claim_submitted_date" in raw_features:
            input_values["claim_submitted_date"] = st.date_input("Claim submitted date", value=date.today()).isoformat()

    submitted = st.form_submit_button("Estimate approved amount", type="primary", use_container_width=True)

if submitted:
    input_df = pd.DataFrame([input_values])
    transformed = prepare_features(input_df)
    transformed = transformed.reindex(columns=feature_columns)
    prediction = float(pipeline.predict(transformed)[0])
    prediction = max(0.0, prediction)
    st.markdown("---")
    st.subheader("Estimated approved claim amount")
    st.metric("Model estimate", f"₹{prediction:,.0f}")
    if "claimed_amount" in input_values:
        claimed = float(input_values["claimed_amount"])
        st.metric("Claimed amount", f"₹{claimed:,.0f}")
        st.metric("Estimated difference", f"₹{claimed-prediction:,.0f}")
    st.caption(f"Model used: {bundle['model_name']}. This is a statistical estimate from historical training data, not a policy-rule calculation.")
    with st.expander("Model validation information"):
        st.json(bundle["metrics"])
