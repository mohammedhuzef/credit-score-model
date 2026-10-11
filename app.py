"""CreditGuard — Credit Default Risk Predictor

Educational decision-support web application built with Streamlit, Scikit-learn, and Plotly.
Features a bold Neo-Brutalist fintech UI/UX design.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import sys
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.data_preprocessing import (
    CATEGORICAL_FEATURES,
    CATEGORICAL_MAPPINGS,
    NUMERIC_FEATURES,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    clean_data,
    download_uci_dataset,
    load_raw_data,
)
from src.evaluate_model import compute_confusion_matrix_data, compute_metrics_at_threshold
from src.train_model import MODELS_DIR, train_and_compare_models
from src.utils import (
    PALETTE,
    PERSONA_PRESETS,
    apply_neobrutalist_layout,
    generate_applicant_explanation,
    get_risk_tier,
)

# -----------------------------------------------------------------------------
# Streamlit App Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CreditGuard — Credit Default Risk Predictor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load Custom Neo-Brutalist CSS
CSS_PATH = BASE_DIR / "assets" / "style.css"
if CSS_PATH.exists():
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Cached Model & Data Loaders
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_all_artifacts() -> Optional[Dict[str, Any]]:
    """Loads all trained model pipelines, metrics, and feature importances."""
    required_files = [
        "best_model_pipeline.joblib",
        "logistic_pipeline.joblib",
        "random_forest_pipeline.joblib",
        "gradient_boosting_pipeline.joblib",
        "model_comparison_metrics.joblib",
        "feature_importances.joblib",
        "training_metadata.joblib",
    ]
    for rf in required_files:
        if not (MODELS_DIR / rf).exists():
            return None

    try:
        return {
            "best_pipeline": joblib.load(MODELS_DIR / "best_model_pipeline.joblib"),
            "logistic_pipeline": joblib.load(MODELS_DIR / "logistic_pipeline.joblib"),
            "rf_pipeline": joblib.load(MODELS_DIR / "random_forest_pipeline.joblib"),
            "gb_pipeline": joblib.load(MODELS_DIR / "gradient_boosting_pipeline.joblib"),
            "metrics": joblib.load(MODELS_DIR / "model_comparison_metrics.joblib"),
            "features": joblib.load(MODELS_DIR / "feature_importances.joblib"),
            "metadata": joblib.load(MODELS_DIR / "training_metadata.joblib"),
        }
    except Exception as e:
        st.error(f"Error loading model artifacts: {e}")
        return None


@st.cache_data(show_spinner=False)
def load_dataset_for_exploration() -> pd.DataFrame:
    """Loads cleaned dataset for visualizations and portfolio statistics."""
    try:
        raw = load_raw_data()
        return clean_data(raw)
    except Exception:
        return pd.DataFrame()


# -----------------------------------------------------------------------------
# Sidebar Navigation & Model Health Status
# -----------------------------------------------------------------------------
def render_sidebar(artifacts: Optional[Dict[str, Any]]) -> str:
    """Renders the Neo-Brutalist sidebar navigation."""
    with st.sidebar:
        # App Logo & Branding
        st.markdown(
            """
            <div style="padding: 6px 0 16px 0; border-bottom: 3px solid #111111; margin-bottom: 18px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="background: #B6FF4A; border: 2.5px solid #111111; box-shadow: 3px 3px 0 #111111; 
                                width: 38px; height: 38px; display: flex; align-items: center; justify-content: center; 
                                font-size: 20px; font-weight: 900;">🛡️</div>
                    <div>
                        <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.35rem; font-weight: 900; line-height: 1;">CREDITGUARD</div>
                        <div style="font-family: 'Space Grotesk', sans-serif; font-size: 0.75rem; font-weight: 700; color: #555555; letter-spacing: 0.5px;">RISK DECISION SUPPORT</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<p style='font-weight: 800; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px;'>Navigation</p>", unsafe_allow_html=True)
        page = st.radio(
            "Go to page:",
            [
                "🏠 Dashboard / Home",
                "⚡ Risk Prediction",
                "📊 Model Performance",
                "🔍 Feature Insights",
                "⚖️ Responsible AI & Limitations",
            ],
            index=0,
            label_visibility="collapsed",
        )

        st.markdown("<hr style='border: 1.5px solid #111111; margin: 20px 0;'>", unsafe_allow_html=True)

        # Model Status Box
        if artifacts is not None:
            meta = artifacts["metadata"]
            st.markdown(
                f"""
                <div style="background: #FFFFFF; border: 2.5px solid #111111; box-shadow: 4px 4px 0 #111111; padding: 12px; border-radius: 4px;">
                    <div style="font-family: 'Space Grotesk', sans-serif; font-weight: 800; font-size: 0.78rem; text-transform: uppercase; color: #111111; margin-bottom: 6px;">
                        SYSTEM HEALTH: <span style="background: #B6FF4A; padding: 2px 6px; border: 1.5px solid #111111; border-radius: 3px;">ACTIVE</span>
                    </div>
                    <div style="font-size: 0.8rem; font-weight: 600; line-height: 1.4;">
                        <b>Model:</b> {meta.get('best_model_name', 'Trained Ensemble')}<br>
                        <b>Cohort:</b> {meta.get('train_samples', 800) + meta.get('test_samples', 200)} Applicants<br>
                        <b>Evaluated:</b> {meta.get('trained_at', 'Online')[:10]}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="background: #FFD84D; border: 2.5px solid #111111; box-shadow: 4px 4px 0 #111111; padding: 12px; border-radius: 4px;">
                    <div style="font-family: 'Space Grotesk', sans-serif; font-weight: 800; font-size: 0.8rem; text-transform: uppercase;">
                        ⚠️ MODEL NOT TRAINED
                    </div>
                    <div style="font-size: 0.8rem; margin-top: 4px; font-weight: 600;">
                        Artifacts missing in models/ directory.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div style="margin-top: 25px; padding: 10px; background: #FAF8F2; border: 2px dashed #111111; font-size: 0.74rem; font-weight: 600; line-height: 1.35;">
                <b>DISCLAIMER:</b><br>
                Educational decision-support tool. Not an automated lending decision system. Does not replace human underwriters.
            </div>
            """,
            unsafe_allow_html=True,
        )

        return page


# -----------------------------------------------------------------------------
# Polished Empty State / Training Handler
# -----------------------------------------------------------------------------
def render_empty_state() -> None:
    """Displays a clear Neo-Brutalist empty state when models or data are missing."""
    st.markdown(
        """
        <div class="nb-hero" style="background: #FFD84D;">
            <div class="hero-pill">SYSTEM INITIALIZATION NEEDED</div>
            <h1 style="margin: 6px 0 12px 0;">Model Artifacts Not Found</h1>
            <p style="font-size: 1.1rem; font-weight: 600; max-width: 780px;">
                CreditGuard requires trained machine-learning pipelines and evaluation metrics to provide decision support. 
                You can generate the complete training suite right now with 1 click, or run the training script from the command line.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        st.markdown(
            """
            <div class="nb-card">
                <div class="nb-form-header">⚡ Option 1: One-Click Initialization</div>
                <p style="font-size: 0.92rem; font-weight: 500;">
                    Click the button below to download the official <b>UCI German Credit dataset</b>, 
                    execute preprocessing, train all 3 pipelines (Logistic Regression, Random Forest, Gradient Boosting), 
                    evaluate ROC-AUC, and save artifacts to <code>models/</code>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("RUN FULL ML PIPELINE & TRAIN MODELS NOW →", key="btn_train_models"):
            with st.spinner("Executing data ingestion, feature transformation, and model training..."):
                try:
                    res = train_and_compare_models()
                    st.success(f"Models successfully trained! Best model selected: {res['best_model_name']}")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Training failed: {ex}")

    with c2:
        st.markdown(
            """
            <div class="nb-card">
                <div class="nb-form-header">💻 Option 2: Command Line Run</div>
                <p style="font-size: 0.92rem; font-weight: 500;">
                    You can also execute the training pipeline directly from your terminal:
                </p>
                <pre style="background: #111111; color: #B6FF4A; padding: 12px; border-radius: 4px; font-weight: 700; font-family: monospace;">
# Step 1: Ingest & clean data
python -m src.data_preprocessing

# Step 2: Train & compare models
python -m src.train_model
                </pre>
                <p style="font-size: 0.85rem; font-weight: 600; margin-top: 8px;">
                    Raw dataset location: <code>data/raw/german_credit.csv</code>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# PAGE 1: Home / Dashboard
# -----------------------------------------------------------------------------
def render_home_page(artifacts: Dict[str, Any], df_clean: pd.DataFrame) -> None:
    """Renders the Home/Dashboard page with hero, KPIs, overview, and disclaimer."""
    metrics_data = artifacts["metrics"]
    evals = metrics_data["evaluations"]
    best_name = metrics_data.get("best_model_name", "Logistic Regression")
    best_eval = evals.get(best_name, list(evals.values())[0])
    m = best_eval["metrics"]

    # Hero Banner
    st.markdown(
        f"""
        <div class="nb-hero">
            <div class="hero-pill">⚡ EDUCATIONAL AI DECISION SUPPORT</div>
            <h1 style="margin: 8px 0 10px 0;">Make Credit Risk Decisions More Explainable</h1>
            <p style="font-size: 1.15rem; font-weight: 600; color: #333333; max-width: 850px; line-height: 1.45;">
                A machine-learning decision-support tool for estimating credit default risk. 
                Combines transparent probabilistic scoring, interactive threshold analysis, and interpretable historical drivers.
            </p>
            <div style="display: flex; gap: 12px; flex-wrap: wrap; margin-top: 18px;">
                <span class="nb-badge" style="background: #3B82F6; color: #FFFFFF;">
                    🎯 Active Model: {best_name}
                </span>
                <span class="nb-badge" style="background: #B6FF4A; color: #0B3D0B;">
                    📊 Baseline ROC-AUC: {m['ROC-AUC']:.3f}
                </span>
                <span class="nb-badge" style="background: #FFD84D; color: #593C00;">
                    ⚖️ Class Balanced (70% Non-Default / 30% Default)
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Mandatory Prominent Disclaimer
    st.markdown(
        """
        <div class="nb-callout">
            <div style="display: flex; align-items: center; gap: 8px; font-family: 'Space Grotesk', sans-serif; font-weight: 800; font-size: 1rem; color: #111111;">
                <span>⚠️</span> IMPORTANT REGULATORY & ETHICAL DISCLAIMER
            </div>
            <div style="font-size: 0.92rem; font-weight: 600; margin-top: 6px; line-height: 1.4;">
                This system is strictly an <b>educational decision-support tool</b> designed to assist qualified human underwriters. 
                It is <b>NOT an automated lending decision system</b> and must never be used to autonomously approve or reject credit applications. 
                All predictions represent statistical estimates derived from historical credit cohorts and require rigorous independent human verification.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # KPI Metric Cards
    st.markdown("<h3 style='margin-bottom: 12px;'>Core Model Benchmark Metrics</h3>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="nb-kpi-grid">
            <div class="nb-kpi-card">
                <span class="kpi-tag" style="background: #3B82F6; color: #FFFFFF;">DISCRIMINATION</span>
                <div class="kpi-value">{m['ROC-AUC']:.3f}</div>
                <div class="kpi-subtitle">Area Under ROC Curve</div>
            </div>
            <div class="nb-kpi-card">
                <span class="kpi-tag" style="background: #B6FF4A; color: #0B3D0B;">SAFETY / SENSITIVITY</span>
                <div class="kpi-value">{m['Recall'] * 100:.1f}%</div>
                <div class="kpi-subtitle">Default Recall (Catches Risky Borrowers)</div>
            </div>
            <div class="nb-kpi-card">
                <span class="kpi-tag" style="background: #FFD84D; color: #593C00;">RELIABILITY</span>
                <div class="kpi-value">{m['Precision'] * 100:.1f}%</div>
                <div class="kpi-subtitle">Precision on Flagged Defaults</div>
            </div>
            <div class="nb-kpi-card">
                <span class="kpi-tag" style="background: #8B5CF6; color: #FFFFFF;">HARMONIC BALANCE</span>
                <div class="kpi-value">{m['F1-Score']:.3f}</div>
                <div class="kpi-subtitle">F1-Score (Precision & Recall Tradeoff)</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Overview Section: Historical Portfolio & Model Health
    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        st.markdown(
            """
            <div class="nb-card">
                <div class="nb-form-header">📈 Historical Applicant Portfolio Overview</div>
                <p style="font-size: 0.9rem; font-weight: 500; margin-bottom: 12px;">
                    Distribution of loan outcomes in the benchmark training dataset (1,000 retail credit agreements).
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not df_clean.empty and TARGET_COLUMN in df_clean.columns:
            target_counts = df_clean[TARGET_COLUMN].value_counts().rename({0: "Non-Default (Good)", 1: "Default (Risky)"})
            fig_pie = go.Figure(
                data=[
                    go.Pie(
                        labels=target_counts.index,
                        values=target_counts.values,
                        hole=0.45,
                        marker=dict(
                            colors=[PALETTE["neon_green"], PALETTE["coral_red"]],
                            line=dict(color=PALETTE["black"], width=2.5),
                        ),
                        textinfo="label+percent",
                        textfont=dict(family="Space Grotesk, sans-serif", size=13, color=PALETTE["black"]),
                    )
                ]
            )
            fig_pie = apply_neobrutalist_layout(
                fig_pie,
                title="Class Distribution (70/30 Lending Imbalance)",
                height=320,
            )
            st.plotly_chart(fig_pie, width="stretch")

    with col_right:
        st.markdown(
            """
            <div class="nb-card">
                <div class="nb-form-header">🛡️ Decision Support Principles</div>
                <div style="display: flex; flex-direction: column; gap: 12px; font-size: 0.88rem; font-weight: 600;">
                    <div style="background: #FAF8F2; border: 2px solid #111111; padding: 10px; border-radius: 4px;">
                        <b>1. Probabilistic Estimates, Not Binary Commands</b><br>
                        CreditGuard outputs an estimated default likelihood (e.g. 24%) rather than an automated binary approve/deny order.
                    </div>
                    <div style="background: #FAF8F2; border: 2px solid #111111; padding: 10px; border-radius: 4px;">
                        <b>2. Asymmetric Default Cost Awareness</b><br>
                        A single unrecovered loan default often costs 5x to 10x more than the administrative review cost of a false alarm.
                    </div>
                    <div style="background: #FAF8F2; border: 2px solid #111111; padding: 10px; border-radius: 4px;">
                        <b>3. Human-in-the-Loop Safeguards</b><br>
                        Underwriters retain final discretion to evaluate compensating factors such as cosigners, verified assets, or career trajectory.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Quick Navigation Banner
    st.markdown(
        """
        <div class="nb-card" style="background: #FAF8F2; border: 3px solid #111111;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
                <div>
                    <h3 style="margin: 0;">Ready to evaluate a loan applicant?</h3>
                    <p style="margin: 4px 0 0 0; font-size: 0.9rem; font-weight: 600; color: #444444;">
                        Test applicant profiles, adjust risk thresholds, and inspect individual risk factors.
                    </p>
                </div>
                <div style="font-family: 'Space Grotesk', sans-serif; font-weight: 800; font-size: 0.95rem;">
                    👉 Switch to <b>"⚡ Risk Prediction"</b> in the left navigation sidebar!
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# PAGE 2: Risk Prediction
# -----------------------------------------------------------------------------
def render_risk_prediction_page(artifacts: Dict[str, Any]) -> None:
    """Renders the interactive loan-applicant risk prediction form and evaluation suite."""
    st.markdown(
        """
        <div class="nb-hero" style="padding: 22px 26px;">
            <div class="hero-pill">INTERACTIVE DECISION ENGINE</div>
            <h1 style="margin: 6px 0 8px 0;">Applicant Default Risk Assessment</h1>
            <p style="font-size: 1.05rem; font-weight: 600; color: #333333; margin: 0;">
                Input applicant credentials across four risk dimensions to generate real-time probability estimates and factor breakdowns.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Initialize session state for form presets if not set
    if "applicant_form" not in st.session_state:
        st.session_state["applicant_form"] = PERSONA_PRESETS["Prime Borrower (Low Risk)"].copy()

    # Persona Preset Bar
    st.markdown("<h4 style='margin-bottom: 8px;'>Quick-Load Benchmark Personas:</h4>", unsafe_allow_html=True)
    p_cols = st.columns([1, 1, 1])
    for idx, (p_name, p_data) in enumerate(PERSONA_PRESETS.items()):
        with p_cols[idx]:
            if st.button(f"👤 {p_name}", key=f"btn_persona_{idx}", width="stretch"):
                st.session_state["applicant_form"] = p_data.copy()
                st.rerun()

    current_values = st.session_state["applicant_form"]

    st.markdown("<hr style='border: 1.5px solid #111111; margin: 18px 0;'>", unsafe_allow_html=True)

    # Grouped Form
    with st.form("applicant_risk_form"):
        # Group 1: Personal Profile
        st.markdown(
            """
            <div class="nb-form-header">
                <span>👤</span> 1. Personal Profile
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            age = st.number_input(
                "Applicant Age (Years)",
                min_value=18,
                max_value=90,
                value=int(current_values.get("age_years", 35)),
                help="Legal applicant age at time of agreement.",
            )
            housing = st.selectbox(
                "Housing Arrangement",
                options=["Own", "Rent", "For Free"],
                index=["Own", "Rent", "For Free"].index(current_values.get("housing", "Own")),
            )
        with col_p2:
            status_sex_opts = [
                "Male: Single",
                "Male: Married / Widowed",
                "Male: Divorced / Separated",
                "Female: Single",
                "Female: Divorced / Married",
            ]
            personal_status_sex = st.selectbox(
                "Personal Status & Marital Profile",
                options=status_sex_opts,
                index=status_sex_opts.index(current_values.get("personal_status_sex", "Male: Single")),
            )
            num_dependents = st.selectbox(
                "Number of Dependents Liable to Maintain",
                options=[1, 2],
                index=[1, 2].index(int(current_values.get("num_dependents", 1))),
            )
        with col_p3:
            telephone = st.selectbox(
                "Telephone Registration",
                options=["Yes (Registered)", "None"],
                index=["Yes (Registered)", "None"].index(current_values.get("telephone", "Yes (Registered)")),
            )
            foreign_worker = st.selectbox(
                "Foreign Worker Status",
                options=["No", "Yes"],
                index=["No", "Yes"].index(current_values.get("foreign_worker", "No")),
            )

        # Group 2: Employment & Income
        st.markdown(
            """
            <div class="nb-form-header" style="margin-top: 18px;">
                <span>💼</span> 2. Employment & Income
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_e1, col_e2, col_e3 = st.columns(3)
        with col_e1:
            emp_opts = ["Unemployed", "< 1 year", "1 - 4 years", "4 - 7 years", ">= 7 years"]
            employment_duration = st.selectbox(
                "Present Employment Duration",
                options=emp_opts,
                index=emp_opts.index(current_values.get("employment_duration", ">= 7 years")),
            )
        with col_e2:
            job_opts = [
                "Management / Self-employed / Highly Qualified",
                "Skilled Employee / Official",
                "Unskilled Resident",
                "Unemployed / Unskilled Non-resident",
            ]
            job_type = st.selectbox(
                "Professional Job Classification",
                options=job_opts,
                index=job_opts.index(current_values.get("job_type", "Skilled Employee / Official")),
            )
        with col_e3:
            residence_duration = st.slider(
                "Residence at Current Address (Years)",
                min_value=1,
                max_value=4,
                value=int(current_values.get("residence_duration", 3)),
                help="Stability proxy: 1 represents under a year, 4 represents 4+ years.",
            )

        # Group 3: Loan Details
        st.markdown(
            """
            <div class="nb-form-header" style="margin-top: 18px;">
                <span>💰</span> 3. Loan Details
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_l1, col_l2, col_l3 = st.columns(3)
        with col_l1:
            purpose_opts = [
                "Car (new)",
                "Car (used)",
                "Furniture / Equipment",
                "Radio / Television",
                "Domestic appliances",
                "Repairs",
                "Education",
                "Vacation",
                "Retraining",
                "Business",
                "Others",
            ]
            purpose = st.selectbox(
                "Loan Purpose",
                options=purpose_opts,
                index=purpose_opts.index(current_values.get("purpose", "Car (used)")),
            )
        with col_l2:
            credit_amount = st.number_input(
                "Requested Credit Amount (DM / Currency)",
                min_value=250,
                max_value=20000,
                step=100,
                value=int(current_values.get("credit_amount", 2500)),
                help="Total principal borrowed.",
            )
        with col_l3:
            duration_months = st.number_input(
                "Repayment Duration (Months)",
                min_value=4,
                max_value=72,
                step=1,
                value=int(current_values.get("duration_months", 18)),
                help="Repayment term in months.",
            )

        installment_rate = st.slider(
            "Installment Rate (% of Disposable Income)",
            min_value=1,
            max_value=4,
            value=int(current_values.get("installment_rate", 2)),
            help="1 is light (< 20%), 4 is heavy debt service (> 35%).",
        )

        # Group 4: Credit History & Assets
        st.markdown(
            """
            <div class="nb-form-header" style="margin-top: 18px;">
                <span>🏛️</span> 4. Credit History & Financial Standing
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            chk_opts = [
                ">= 200 DM (Healthy)",
                "0 - 200 DM (Moderate)",
                "< 0 DM (Overdrawn)",
                "No Checking Account",
            ]
            checking_status = st.selectbox(
                "Checking Account Status",
                options=chk_opts,
                index=chk_opts.index(current_values.get("checking_status", ">= 200 DM (Healthy)")),
            )
            hist_opts = [
                "Critical account / other credits elsewhere",
                "Existing credits paid duly till now",
                "All credits at this bank paid duly",
                "No credits / all paid duly",
                "Delay in paying off in the past",
            ]
            credit_history = st.selectbox(
                "Historical Credit Standing",
                options=hist_opts,
                index=hist_opts.index(current_values.get("credit_history", "Existing credits paid duly till now")),
            )
            sav_opts = [
                ">= 1000 DM",
                "500 - 1000 DM",
                "100 - 500 DM",
                "< 100 DM",
                "Unknown / No Savings",
            ]
            savings_status = st.selectbox(
                "Liquid Savings Account / Bonds Buffer",
                options=sav_opts,
                index=sav_opts.index(current_values.get("savings_status", ">= 1000 DM")),
            )
            debtor_opts = ["None", "Co-applicant", "Guarantor"]
            other_debtors = st.selectbox(
                "Other Debtors / Guarantors Backing",
                options=debtor_opts,
                index=debtor_opts.index(current_values.get("other_debtors", "None")),
            )

        with col_c2:
            prop_opts = [
                "Real Estate",
                "Building Society / Life Insurance",
                "Car or Other Assets",
                "Unknown / No Property",
            ]
            property_type = st.selectbox(
                "Property Ownership & Asset Backing",
                options=prop_opts,
                index=prop_opts.index(current_values.get("property", "Real Estate")),
            )
            inst_plans_opts = ["None", "Bank", "Stores"]
            other_installment_plans = st.selectbox(
                "Other External Installment Obligations",
                options=inst_plans_opts,
                index=inst_plans_opts.index(current_values.get("other_installment_plans", "None")),
            )
            existing_credits = st.slider(
                "Number of Existing Credit Accounts at this Bank",
                min_value=1,
                max_value=4,
                value=int(current_values.get("existing_credits", 1)),
            )

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("CHECK RISK →", width="stretch")

    # Threshold & Model Selector Panel outside form
    st.markdown(
        """
        <div class="nb-card" style="background: #FFFFFF; border: 3px solid #111111; margin-top: 14px;">
            <div class="nb-form-header">⚙️ Decision Sensitivity & Model Choice</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    c_m1, c_m2 = st.columns([1, 1.4])
    with c_m1:
        model_choice = st.selectbox(
            "Select Inference Model Pipeline:",
            options=["Best Model (Logistic Regression)", "Random Forest", "Gradient Boosting"],
            index=0,
        )
    with c_m2:
        decision_threshold = st.slider(
            "Decision Risk Threshold (Cutoff):",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05,
            help="Applicants whose estimated default probability meets or exceeds this threshold are flagged for High-Risk review.",
        )
        st.caption(
            "💡 **Threshold Trade-off:** Lower threshold (e.g. 0.35) = More cautious, catches more defaults (high recall), but flags more reliable borrowers. "
            "Higher threshold (e.g. 0.65) = More lenient, approves more borrowers, but increases bad-debt risk."
        )

    # Process and Predict
    applicant_dict = {
        "checking_status": checking_status,
        "duration_months": duration_months,
        "credit_history": credit_history,
        "purpose": purpose,
        "credit_amount": credit_amount,
        "savings_status": savings_status,
        "employment_duration": employment_duration,
        "installment_rate": installment_rate,
        "personal_status_sex": personal_status_sex,
        "other_debtors": other_debtors,
        "residence_duration": residence_duration,
        "property": property_type,
        "age_years": age,
        "other_installment_plans": other_installment_plans,
        "housing": housing,
        "existing_credits": existing_credits,
        "job_type": job_type,
        "num_dependents": num_dependents,
        "telephone": telephone,
        "foreign_worker": foreign_worker,
    }

    # Pick pipeline
    if "Random Forest" in model_choice:
        active_pipe = artifacts["rf_pipeline"]
    elif "Gradient Boosting" in model_choice:
        active_pipe = artifacts["gb_pipeline"]
    else:
        active_pipe = artifacts["best_pipeline"]

    # Format input row into DataFrame
    input_df = pd.DataFrame([applicant_dict])

    try:
        raw_prob = active_pipe.predict_proba(input_df)[0, 1]
    except Exception as e:
        st.error(f"Inference error: {e}")
        return

    # Categorize Risk
    risk_info = get_risk_tier(raw_prob, threshold=decision_threshold)
    is_flagged = raw_prob >= decision_threshold

    # Prediction Output Display
    st.markdown("<hr style='border: 2px solid #111111; margin: 28px 0 20px 0;'>", unsafe_allow_html=True)
    st.markdown("<h2>Decision Support Evaluation</h2>", unsafe_allow_html=True)

    res_col1, res_col2 = st.columns([1.1, 1.2])

    with res_col1:
        st.markdown(
            f"""
            <div class="nb-risk-display" style="border-top: 10px solid {risk_info['badge_color']};">
                <span class="nb-badge" style="background: {risk_info['badge_color']}; color: {risk_info['text_color']};">
                    {risk_info['icon']} {risk_info['tier']}
                </span>
                <div class="nb-risk-pct">{raw_prob * 100:.1f}%</div>
                <div style="font-family: 'Space Grotesk', sans-serif; font-size: 1.05rem; font-weight: 800; text-transform: uppercase;">
                    ESTIMATED DEFAULT PROBABILITY
                </div>
                <div style="font-size: 0.85rem; font-weight: 600; color: #555555; margin-top: 4px;">
                    Decision Threshold: {decision_threshold * 100:.0f}% | Model: {model_choice.split(' ')[0]}
                </div>
                <div style="margin-top: 16px; padding: 10px; background: #F7F3E8; border: 2px solid #111111; border-radius: 4px; font-weight: 700; font-size: 0.9rem;">
                    {risk_info['action_text']}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with res_col2:
        st.markdown(
            f"""
            <div class="nb-card" style="height: 94%;">
                <div class="nb-form-header">📋 Summary & Plain-Language Assessment</div>
                <p style="font-size: 0.95rem; font-weight: 600; line-height: 1.45; margin-bottom: 12px;">
                    {risk_info['summary']}
                </p>
                <div style="background: {'#FFF1F0' if is_flagged else '#F0FFF4'}; border: 2px solid #111111; padding: 12px; border-radius: 4px; margin-bottom: 12px;">
                    <b>Policy Alignment:</b> At the current <b>{decision_threshold * 100:.0f}%</b> cutoff, 
                    this applicant is <b>{'FLAGGED FOR UNDERWRITING ESCALATION' if is_flagged else 'WITHIN STANDARD ACCEPTABLE RISK LIMITS'}</b>.
                </div>
                <div class="nb-disclaimer" style="margin-top: 8px;">
                    <b>Mandatory Notice:</b> This result supports human review. It is not an automatic approval or rejection.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Detailed Factor Breakdown
    st.markdown("<h3 style='margin-top: 18px;'>Key Contributing Applicant Drivers</h3>", unsafe_allow_html=True)
    factors = generate_applicant_explanation(applicant_dict, raw_prob, decision_threshold)

    fact_cols = st.columns(2)
    risk_factors = [f for f in factors if f["type"] == "risk"]
    pos_factors = [f for f in factors if f["type"] == "positive"]

    with fact_cols[0]:
        st.markdown(
            """
            <div class="nb-card" style="border-left: 8px solid #FF6B5E;">
                <div style="font-family: 'Space Grotesk', sans-serif; font-weight: 800; font-size: 1rem; color: #111111; margin-bottom: 8px;">
                    🔴 Risk Elevating Signals
                </div>
            """,
            unsafe_allow_html=True,
        )
        if risk_factors:
            for rf in risk_factors:
                st.markdown(
                    f"""
                    <div style="margin-bottom: 10px; padding-bottom: 8px; border-bottom: 1px dashed #CCCCCC;">
                        <b style="color: #111111;">{rf['title']}</b><br>
                        <span style="font-size: 0.85rem; font-weight: 500; color: #444444;">{rf['desc']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("<p style='font-size: 0.88rem; font-weight: 600; color: #2B7A0B;'>No severe elevated risk signals detected in this application.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with fact_cols[1]:
        st.markdown(
            """
            <div class="nb-card" style="border-left: 8px solid #B6FF4A;">
                <div style="font-family: 'Space Grotesk', sans-serif; font-weight: 800; font-size: 1rem; color: #111111; margin-bottom: 8px;">
                    🟢 Protective & Mitigating Signals
                </div>
            """,
            unsafe_allow_html=True,
        )
        if pos_factors:
            for pf in pos_factors:
                st.markdown(
                    f"""
                    <div style="margin-bottom: 10px; padding-bottom: 8px; border-bottom: 1px dashed #CCCCCC;">
                        <b style="color: #111111;">{pf['title']}</b><br>
                        <span style="font-size: 0.85rem; font-weight: 500; color: #444444;">{pf['desc']}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.markdown("<p style='font-size: 0.88rem; font-weight: 600; color: #666666;'>No strong protective factors identified.</p>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# PAGE 3: Model Performance
# -----------------------------------------------------------------------------
def render_model_performance_page(artifacts: Dict[str, Any]) -> None:
    """Renders model comparison tables, ROC & PR curves, and confusion matrices."""
    metrics_data = artifacts["metrics"]
    evals = metrics_data["evaluations"]
    summary_table = metrics_data["summary_table"]

    st.markdown(
        """
        <div class="nb-hero" style="padding: 22px 26px;">
            <div class="hero-pill">MODEL BENCHMARKING & VALIDATION</div>
            <h1 style="margin: 6px 0 8px 0;">Comprehensive Model Performance</h1>
            <p style="font-size: 1.05rem; font-weight: 600; color: #333333; margin: 0;">
                Rigorous evaluation of baseline Logistic Regression, Random Forest, and Gradient Boosting on an out-of-sample 20% holdout test set (200 credit agreements).
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 1. Performance Table
    st.markdown("<h3 style='margin-bottom: 10px;'>Model Comparison Table</h3>", unsafe_allow_html=True)
    table_df = pd.DataFrame(summary_table)

    # Render as Neo-Brutalist HTML Table
    table_html = "<table class='nb-table'><thead><tr>"
    for col in table_df.columns:
        table_html += f"<th>{col}</th>"
    table_html += "</tr></thead><tbody>"
    for _, row in table_df.iterrows():
        table_html += "<tr>"
        for col in table_df.columns:
            val = row[col]
            if col == "Model" and val == metrics_data.get("best_model_name"):
                table_html += f"<td><b>{val}</b> <span style='background: #B6FF4A; font-size: 0.75rem; padding: 2px 6px; border: 1.5px solid #111111; border-radius: 3px;'>SELECTED</span></td>"
            else:
                table_html += f"<td>{val}</td>"
        table_html += "</tr>"
    table_html += "</tbody></table>"

    st.markdown(table_html, unsafe_allow_html=True)

    # Beginner-Friendly Explanations of Metrics
    with st.expander("📖 Click to Read Beginner-Friendly Metric Definitions", expanded=False):
        st.markdown(
            """
            - **Accuracy:** The overall percentage of correct predictions (both defaults and non-defaults). *Can be misleading with imbalanced data!*
            - **Precision:** Of all applicants the model flagged as high-risk defaulters, how many actually defaulted? High precision means fewer false alarms.
            - **Recall (Sensitivity):** Of all actual defaulters in the portfolio, how many did the model successfully detect? High recall prevents costly bad debts.
            - **F1-Score:** The harmonic balance between Precision and Recall. Essential when false alarms and missed defaults are both important.
            - **ROC-AUC (Area Under ROC Curve):** Measures the model's ability to rank a random defaulter as higher risk than a random non-defaulter across all cutoffs. 0.5 is random chance; 1.0 is perfection.
            """
        )

    # Dedicated Section: Why Accuracy Alone is Not Enough
    st.markdown(
        """
        <div class="nb-callout" style="background: #FFF7D6; border-left: 10px solid #FFD84D;">
            <div style="font-family: 'Space Grotesk', sans-serif; font-weight: 800; font-size: 1.1rem; color: #111111;">
                ⚠️ Why Accuracy Alone Is Not Enough in Credit Risk
            </div>
            <div style="font-size: 0.92rem; font-weight: 600; line-height: 1.5; margin-top: 8px;">
                In the retail German Credit dataset, <b>70% of borrowers pay back cleanly and 30% default</b>. 
                A naive "dummy" model that mindlessly approves 100% of applicants would achieve an apparently high <b>70% Accuracy</b>—yet it would catch <b>0% of defaults (0% Recall)</b>, 
                guaranteeing catastrophic insolvency for any lending institution.
                <br><br>
                Therefore, responsible credit modeling prioritizes <b>ROC-AUC, Recall, and F1-score</b> under class-balanced loss functions.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Curves: ROC and Precision-Recall
    st.markdown("<h3 style='margin-top: 24px; margin-bottom: 12px;'>Interactive ROC & Precision-Recall Curves</h3>", unsafe_allow_html=True)
    c_curve1, c_curve2 = st.columns(2)

    with c_curve1:
        fig_roc = go.Figure()
        colors = {
            "Logistic Regression": PALETTE["blue"],
            "Random Forest": PALETTE["purple"],
            "Gradient Boosting": PALETTE["coral_red"],
        }
        for name, ev in evals.items():
            curves = ev["curves"]
            fig_roc.add_trace(
                go.Scatter(
                    x=curves["fpr"],
                    y=curves["tpr"],
                    mode="lines",
                    name=f"{name} (AUC={curves['roc_auc']:.3f})",
                    line=dict(color=colors.get(name, "#111111"), width=3.5),
                )
            )
        # Random Chance Line
        fig_roc.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Random Chance (AUC=0.500)",
                line=dict(color="#888888", width=2, dash="dash"),
            )
        )
        fig_roc = apply_neobrutalist_layout(
            fig_roc,
            title="ROC Curve Comparison",
            xaxis_title="False Positive Rate (1 - Specificity)",
            yaxis_title="True Positive Rate (Recall)",
            height=400,
        )
        st.plotly_chart(fig_roc, width="stretch")

    with c_curve2:
        fig_pr = go.Figure()
        for name, ev in evals.items():
            curves = ev["curves"]
            fig_pr.add_trace(
                go.Scatter(
                    x=curves["recalls"],
                    y=curves["precisions"],
                    mode="lines",
                    name=name,
                    line=dict(color=colors.get(name, "#111111"), width=3.5),
                )
            )
        # Baseline horizontal line (prevalence = 30%)
        fig_pr.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0.30, 0.30],
                mode="lines",
                name="Baseline Prevalence (30%)",
                line=dict(color="#888888", width=2, dash="dash"),
            )
        )
        fig_pr = apply_neobrutalist_layout(
            fig_pr,
            title="Precision-Recall Curve Comparison",
            xaxis_title="Recall (Default Catch Rate)",
            yaxis_title="Precision (Accuracy of Flagged Defaults)",
            height=400,
        )
        st.plotly_chart(fig_pr, width="stretch")

    # Interactive Confusion Matrix
    st.markdown("<hr style='border: 1.5px solid #111111; margin: 24px 0;'>", unsafe_allow_html=True)
    st.markdown("<h3 style='margin-bottom: 12px;'>Confusion Matrix & Lending Business Cost Breakdown</h3>", unsafe_allow_html=True)

    sel_col1, sel_col2 = st.columns([1, 1])
    with sel_col1:
        cm_model_choice = st.selectbox(
            "Select Model to Inspect Confusion Matrix:",
            options=list(evals.keys()),
            index=0,
        )
    with sel_col2:
        cm_threshold = st.slider(
            "Evaluation Cutoff Threshold for Confusion Matrix:",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05,
            key="cm_thresh_slider",
        )

    # Recompute confusion matrix at slider threshold
    target_eval = evals[cm_model_choice]
    y_prob = np.array(target_eval["y_prob"])
    y_true = np.array(target_eval["y_true"])
    cm_data = compute_confusion_matrix_data(y_true, y_prob, threshold=cm_threshold)
    dyn_metrics = compute_metrics_at_threshold(y_true, y_prob, threshold=cm_threshold)

    c_mat1, c_mat2 = st.columns([1.1, 1.2])

    with c_mat1:
        # Plotly Heatmap
        matrix_vals = cm_data["matrix"]
        fig_cm = go.Figure(
            data=go.Heatmap(
                z=matrix_vals,
                x=["Predicted Non-Default (0)", "Predicted Default (1)"],
                y=["Actual Non-Default (0)", "Actual Default (1)"],
                text=[
                    [f"TN: {matrix_vals[0][0]}<br>({matrix_vals[0][0]/len(y_true):.1%})", f"FP: {matrix_vals[0][1]}<br>({matrix_vals[0][1]/len(y_true):.1%})"],
                    [f"FN: {matrix_vals[1][0]}<br>({matrix_vals[1][0]/len(y_true):.1%})", f"TP: {matrix_vals[1][1]}<br>({matrix_vals[1][1]/len(y_true):.1%})"],
                ],
                texttemplate="%{text}",
                textfont=dict(family="Space Grotesk, sans-serif", size=16, color=PALETTE["black"]),
                colorscale=[[0, "#FAF8F2"], [0.5, "#FFD84D"], [1, "#B6FF4A"]],
                showscale=False,
            )
        )
        fig_cm = apply_neobrutalist_layout(
            fig_cm,
            title=f"Confusion Matrix ({cm_model_choice} @ {cm_threshold:.2f})",
            xaxis_title="Predicted Outcome",
            yaxis_title="True Historical Outcome",
            height=360,
        )
        st.plotly_chart(fig_cm, width="stretch")

    with c_mat2:
        st.markdown(
            f"""
            <div class="nb-card" style="height: 94%;">
                <div class="nb-form-header">💼 Underwriting Cost Translation</div>
                <div style="display: flex; flex-direction: column; gap: 8px; font-size: 0.86rem; font-weight: 600;">
                    <div style="background: #FAF8F2; border: 2px solid #111111; padding: 8px 12px; border-radius: 4px;">
                        <span style="color: #0B3D0B; font-weight: 800;">🟢 True Negatives ({cm_data['TN']}) [{cm_data['TN_pct']:.1f}%]:</span><br>
                        Reliable borrowers correctly classified. Approved smoothly, generating planned net interest income.
                    </div>
                    <div style="background: #FAF8F2; border: 2px solid #111111; padding: 8px 12px; border-radius: 4px;">
                        <span style="color: #92400E; font-weight: 800;">🟡 False Positives ({cm_data['FP']}) [{cm_data['FP_pct']:.1f}%]:</span><br>
                        Reliable borrowers flagged for review. Incurs manual underwriter hours and mild borrower friction.
                    </div>
                    <div style="background: #FFF1F0; border: 2px solid #FF6B5E; padding: 8px 12px; border-radius: 4px;">
                        <span style="color: #B91C1C; font-weight: 800;">🔴 False Negatives ({cm_data['FN']}) [{cm_data['FN_pct']:.1f}%]:</span><br>
                        <b>CRITICAL LOSS:</b> Defaulters mistakenly passed without review. Results in severe principal charge-offs.
                    </div>
                    <div style="background: #FAF8F2; border: 2px solid #111111; padding: 8px 12px; border-radius: 4px;">
                        <span style="color: #0B3D0B; font-weight: 800;">🟢 True Positives ({cm_data['TP']}) [{cm_data['TP_pct']:.1f}%]:</span><br>
                        Actual defaulters successfully intercepted. Capital preserved via structured guarantees or limits.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------------------
# PAGE 4: Feature Insights
# -----------------------------------------------------------------------------
def render_feature_insights_page(artifacts: Dict[str, Any], df_clean: pd.DataFrame) -> None:
    """Renders feature importance rankings, positive/negative coefficients, and insights."""
    features = artifacts["features"]
    rf_imps = pd.DataFrame(features["random_forest_importance"])
    lr_coefs = pd.DataFrame(features["logistic_coefficients"])

    st.markdown(
        """
        <div class="nb-hero" style="padding: 22px 26px;">
            <div class="hero-pill">TRANSPARENCY & EXPLAINABILITY</div>
            <h1 style="margin: 6px 0 8px 0;">Empirical Feature Insights</h1>
            <p style="font-size: 1.05rem; font-weight: 600; color: #333333; margin: 0;">
                Inspect which applicant characteristics hold the greatest statistical association with credit default risk across the historical dataset.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Causation vs Correlation Callout
    st.markdown(
        """
        <div class="nb-callout">
            <div style="display: flex; align-items: center; gap: 8px; font-family: 'Space Grotesk', sans-serif; font-weight: 800; font-size: 1rem; color: #111111;">
                <span>💡</span> CRITICAL METHODOLOGICAL DISTINCTION
            </div>
            <div style="font-size: 0.92rem; font-weight: 600; margin-top: 6px; line-height: 1.45;">
                <b>Feature importance shows patterns in historical data. It does NOT prove causation.</b><br>
                High association between a feature (such as checking status or loan duration) and credit default reflects 
                historical observational correlations. It does not establish that changing an individual attribute directly causes credit failure.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs([
        "🌲 Random Forest Importance",
        "⚖️ Logistic Regression Coefficients (Drivers vs Protective)",
        "🔎 Historical Distribution Explorer",
    ])

    with tab1:
        st.markdown("<h3 style='margin-bottom: 8px;'>Top 15 Most Influential Features (Random Forest)</h3>", unsafe_allow_html=True)
        top_rf = rf_imps.head(15).iloc[::-1]

        fig_rf = go.Figure(
            go.Bar(
                x=top_rf["importance"],
                y=top_rf["feature"],
                orientation="h",
                marker=dict(
                    color=PALETTE["blue"],
                    line=dict(color=PALETTE["black"], width=2),
                ),
            )
        )
        fig_rf = apply_neobrutalist_layout(
            fig_rf,
            title="Random Forest Feature Importance (Mean Decrease in Impurity)",
            xaxis_title="Importance Score",
            yaxis_title="",
            height=480,
        )
        st.plotly_chart(fig_rf, width="stretch")

        st.markdown(
            """
            <div class="nb-card" style="background: #FFFFFF;">
                <div class="nb-form-header">Key Observations from Tree Ensembles</div>
                <ul style="font-size: 0.9rem; font-weight: 600; line-height: 1.6; margin: 0; padding-left: 20px;">
                    <li><b>Continuous Exposure:</b> <code>Credit Duration (Months)</code> and <code>Credit Amount (DM)</code> dominate non-linear splits. Extended repayment timelines create exposure horizons with greater opportunity for economic disruption.</li>
                    <li><b>Applicant Age:</b> <code>Applicant Age</code> ranks among top structural features, capturing career maturation and accumulated asset cushions in historical cohorts.</li>
                    <li><b>Checking Account Liquidity:</b> Immediate checking balance signals (such as overdrawn accounts vs &gt; 200 DM reserves) serve as strong primary filters for repayment probability.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab2:
        st.markdown("<h3 style='margin-bottom: 8px;'>Directional Coefficients (Logistic Regression)</h3>", unsafe_allow_html=True)
        st.caption("Positive coefficients increase predicted default odds; Negative coefficients decrease default odds (protective).")

        col_c1, col_c2 = st.columns(2)
        top_pos = pd.DataFrame(features["top_risk_drivers"]).iloc[::-1]
        top_neg = pd.DataFrame(features["top_protective_factors"])

        with col_c1:
            fig_pos = go.Figure(
                go.Bar(
                    x=top_pos["coefficient"],
                    y=top_pos["feature"],
                    orientation="h",
                    marker=dict(
                        color=PALETTE["coral_red"],
                        line=dict(color=PALETTE["black"], width=2),
                    ),
                )
            )
            fig_pos = apply_neobrutalist_layout(
                fig_pos,
                title="Top Risk Escalators (Positive Odds)",
                xaxis_title="Logistic Coefficient (+ Log-Odds)",
                yaxis_title="",
                height=420,
            )
            st.plotly_chart(fig_pos, width="stretch")

        with col_c2:
            fig_neg = go.Figure(
                go.Bar(
                    x=top_neg["coefficient"],
                    y=top_neg["feature"],
                    orientation="h",
                    marker=dict(
                        color=PALETTE["neon_green"],
                        line=dict(color=PALETTE["black"], width=2),
                    ),
                )
            )
            fig_neg = apply_neobrutalist_layout(
                fig_neg,
                title="Top Protective Factors (Negative Odds)",
                xaxis_title="Logistic Coefficient (- Log-Odds)",
                yaxis_title="",
                height=420,
            )
            st.plotly_chart(fig_neg, width="stretch")

    with tab3:
        st.markdown("<h3 style='margin-bottom: 8px;'>Empirical Default Rates Across Cohorts</h3>", unsafe_allow_html=True)
        if not df_clean.empty:
            inspect_col = st.selectbox(
                "Select Feature Dimension to Inspect:",
                options=[
                    "checking_status",
                    "credit_history",
                    "savings_status",
                    "employment_duration",
                    "housing",
                    "purpose",
                ],
                index=0,
            )
            # Groupby and calculate default rate
            grouped = (
                df_clean.groupby(inspect_col)[TARGET_COLUMN]
                .agg(Total="count", Default_Count="sum", Default_Rate="mean")
                .reset_index()
            )
            grouped["Default_Rate_Pct"] = grouped["Default_Rate"] * 100

            fig_grp = go.Figure(
                go.Bar(
                    x=grouped[inspect_col],
                    y=grouped["Default_Rate_Pct"],
                    text=[f"{v:.1f}% ({c} loans)" for v, c in zip(grouped["Default_Rate_Pct"], grouped["Total"])],
                    textposition="auto",
                    marker=dict(
                        color=PALETTE["yellow"],
                        line=dict(color=PALETTE["black"], width=2),
                    ),
                )
            )
            fig_grp = apply_neobrutalist_layout(
                fig_grp,
                title=f"Empirical Default Rate by {inspect_col.replace('_', ' ').title()}",
                xaxis_title=inspect_col.replace("_", " ").title(),
                yaxis_title="Observed Default Rate (%)",
                height=420,
            )
            st.plotly_chart(fig_grp, width="stretch")


# -----------------------------------------------------------------------------
# PAGE 5: Responsible AI / Limitations
# -----------------------------------------------------------------------------
def render_responsible_ai_page() -> None:
    """Renders the comprehensive Responsible AI, limitations, and governance documentation."""
    st.markdown(
        """
        <div class="nb-hero" style="padding: 22px 26px;">
            <div class="hero-pill">ETHICS, COMPLIANCE & GOVERNANCE</div>
            <h1 style="margin: 6px 0 8px 0;">Responsible AI Framework & Model Limitations</h1>
            <p style="font-size: 1.05rem; font-weight: 600; color: #333333; margin: 0;">
                Rigorous accountability standards, legal guardrails, and fair lending constraints governing the deployment of predictive credit models.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Core Principles Grid
    st.markdown("<h3 style='margin-bottom: 12px;'>Foundational Risk Pillars in Automated Credit Support</h3>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            """
            <div class="nb-card">
                <div class="nb-form-header">⚖️ 1. False Positives vs False Negatives in Lending</div>
                <div style="font-size: 0.9rem; font-weight: 500; line-height: 1.5;">
                    Predictive credit errors carry starkly asymmetric consequences:
                    <ul style="margin-top: 8px; padding-left: 18px; font-weight: 600;">
                        <li><b>False Positive (Reliable borrower flagged as High Risk):</b> 
                        Causes borrower friction, delayed access to capital, potential unfair exclusion, and lost origination fee revenue for the lender.</li>
                        <li style="margin-top: 6px;"><b>False Negative (Defaulter approved as Low Risk):</b> 
                        Results in delinquent payments, expensive legal collection procedures, and unrecoverable principal write-offs (often 100% of unpaid capital).</li>
                    </ul>
                    Because the financial cost of a False Negative is typically much higher than a False Positive, lenders naturally calibrate conservative thresholds—yet doing so must never infringe on fair lending equity.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="nb-card">
                <div class="nb-form-header">📊 2. Class Imbalance in Portfolio Data</div>
                <div style="font-size: 0.9rem; font-weight: 500; line-height: 1.5;">
                    Healthy lending books are intrinsically skewed: non-defaulting loans heavily outnumber bad loans (70% vs 30% in German Credit; frequently 97% vs 3% in modern prime portfolios).
                    <br><br>
                    Without deliberate class-weight balancing (e.g. <code>class_weight='balanced'</code>) and threshold calibration, standard models default to minimizing aggregate error by over-predicting the majority class, masking severe default exposure.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="nb-card">
                <div class="nb-form-header">🏛️ 3. Historical Data Bias & Protected Attributes</div>
                <div style="font-size: 0.9rem; font-weight: 500; line-height: 1.5;">
                    Historical loan records reflect human underwriting decisions from past decades. If past credit granting was marred by systemic exclusion, algorithmic pipelines trained on that data risk encoding and perpetuating those exact biases.
                    <br><br>
                    <b>Regulatory Imperative (ECOA, FCRA, EU AI Act):</b>
                    Protected demographic characteristics—including gender, age, marital status, race, religion, and national origin—<b>must never be used to automatically reject applicants</b> or justify disparate adverse credit impact.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="nb-card">
                <div class="nb-form-header">🛡️ 4. Model Drift & Macroeconomic Sensitivity</div>
                <div style="font-size: 0.9rem; font-weight: 500; line-height: 1.5;">
                    Credit default relationships fluctuate across macroeconomic cycles (interest rate spikes, inflation, recessions). 
                    A static machine-learning model trained on benign economic conditions will underestimate default risk during macro downturns without recurring recalibration.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Intended Use vs Out-of-Scope Table
    st.markdown("<h3 style='margin-top: 18px; margin-bottom: 12px;'>System Scope & Operational Boundaries</h3>", unsafe_allow_html=True)
    scope_c1, scope_c2 = st.columns(2)

    with scope_c1:
        st.markdown(
            """
            <div class="nb-card" style="border-left: 8px solid #B6FF4A; background: #FFFFFF;">
                <div style="font-family: 'Space Grotesk', sans-serif; font-weight: 800; font-size: 1.05rem; color: #111111; margin-bottom: 10px;">
                    ✅ Approved Intended Use Cases
                </div>
                <ul style="font-size: 0.88rem; font-weight: 600; line-height: 1.6; margin: 0; padding-left: 18px;">
                    <li>Educational demonstration of explainable credit risk modeling and feature attribution.</li>
                    <li>Decision-support companion providing quantitative probability estimates to accredited human underwriters.</li>
                    <li>Sensitivity stress-testing: understanding how modifying repayment duration or principal changes estimated risk.</li>
                    <li>Benchmarking interpretable linear baselines against non-linear tree ensembles.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with scope_c2:
        st.markdown(
            """
            <div class="nb-card" style="border-left: 8px solid #FF6B5E; background: #FFFFFF;">
                <div style="font-family: 'Space Grotesk', sans-serif; font-weight: 800; font-size: 1.05rem; color: #111111; margin-bottom: 10px;">
                    ❌ Prohibited & Out-of-Scope Uses
                </div>
                <ul style="font-size: 0.88rem; font-weight: 600; line-height: 1.6; margin: 0; padding-left: 18px;">
                    <li>Autonomous, zero-human-in-the-loop loan approval or denial pipelines.</li>
                    <li>Generating legally binding Adverse Action Notices without accredited underwriter confirmation.</li>
                    <li>Employment screening, tenant evaluation, or general consumer profiling.</li>
                    <li>Production deployment without ongoing independent disparate impact audits.</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Final Legal Notice
    st.markdown(
        """
        <div class="nb-disclaimer" style="background: #FFFDF8; border: 3px solid #111111; margin-top: 20px;">
            <div style="font-family: 'Space Grotesk', sans-serif; font-weight: 800; font-size: 0.95rem; text-transform: uppercase;">
                Legal & Financial Disclaimer
            </div>
            <div style="font-size: 0.88rem; font-weight: 500; margin-top: 6px; line-height: 1.45;">
                This software application is provided solely for educational, research, and technical demonstration purposes. 
                It does not constitute financial advice, credit advisory services, or legal counsel. 
                Neither the developers nor affiliated entities assume liability for lending decisions, capital losses, or regulatory non-compliance arising from the use of this software.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Main Application Router
# -----------------------------------------------------------------------------
def main() -> None:
    artifacts = load_all_artifacts()
    df_clean = load_dataset_for_exploration()

    page = render_sidebar(artifacts)

    if artifacts is None:
        render_empty_state()
        return

    if page == "🏠 Dashboard / Home":
        render_home_page(artifacts, df_clean)
    elif page == "⚡ Risk Prediction":
        render_risk_prediction_page(artifacts)
    elif page == "📊 Model Performance":
        render_model_performance_page(artifacts)
    elif page == "🔍 Feature Insights":
        render_feature_insights_page(artifacts, df_clean)
    elif page == "⚖️ Responsible AI & Limitations":
        render_responsible_ai_page()


if __name__ == "__main__":
    main()
