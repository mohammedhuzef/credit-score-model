"""Utility and Helper Functions for CreditGuard.

Includes applicant presets, neo-brutalist Plotly styling, risk categorizers,
and plain-language explanation generators.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Neo-brutalist design palette
PALETTE = {
    "bg_cream": "#F7F3E8",
    "pure_white": "#FFFFFF",
    "black": "#111111",
    "border": "#111111",
    "blue": "#3B82F6",
    "purple": "#8B5CF6",
    "neon_green": "#B6FF4A",
    "yellow": "#FFD84D",
    "coral_red": "#FF6B5E",
    "gray_light": "#E5E1D5",
    "gray_dark": "#4A4A4A",
}

# Persona presets for immediate 1-click testing
PERSONA_PRESETS = {
    "Prime Borrower (Low Risk)": {
        "checking_status": ">= 200 DM (Healthy)",
        "duration_months": 12,
        "credit_history": "Critical account / other credits elsewhere",
        "purpose": "Car (used)",
        "credit_amount": 1500,
        "savings_status": ">= 1000 DM",
        "employment_duration": ">= 7 years",
        "installment_rate": 2,
        "personal_status_sex": "Male: Single",
        "other_debtors": "None",
        "residence_duration": 4,
        "property": "Real Estate",
        "age_years": 48,
        "other_installment_plans": "None",
        "housing": "Own",
        "existing_credits": 2,
        "job_type": "Management / Self-employed / Highly Qualified",
        "num_dependents": 1,
        "telephone": "Yes (Registered)",
        "foreign_worker": "No",
    },
    "Young Professional (Moderate Risk)": {
        "checking_status": "0 - 200 DM (Moderate)",
        "duration_months": 24,
        "credit_history": "Existing credits paid duly till now",
        "purpose": "Radio / Television",
        "credit_amount": 3200,
        "savings_status": "100 - 500 DM",
        "employment_duration": "1 - 4 years",
        "installment_rate": 3,
        "personal_status_sex": "Female: Single",
        "other_debtors": "None",
        "residence_duration": 2,
        "property": "Car or Other Assets",
        "age_years": 26,
        "other_installment_plans": "None",
        "housing": "Rent",
        "existing_credits": 1,
        "job_type": "Skilled Employee / Official",
        "num_dependents": 1,
        "telephone": "None",
        "foreign_worker": "No",
    },
    "Stretched Borrower (High Risk)": {
        "checking_status": "< 0 DM (Overdrawn)",
        "duration_months": 48,
        "credit_history": "Delay in paying off in the past",
        "purpose": "Business",
        "credit_amount": 9500,
        "savings_status": "< 100 DM",
        "employment_duration": "< 1 year",
        "installment_rate": 4,
        "personal_status_sex": "Male: Divorced / Separated",
        "other_debtors": "None",
        "residence_duration": 1,
        "property": "Unknown / No Property",
        "age_years": 23,
        "other_installment_plans": "Bank",
        "housing": "Rent",
        "existing_credits": 3,
        "job_type": "Unskilled Resident",
        "num_dependents": 2,
        "telephone": "None",
        "foreign_worker": "Yes",
    },
}


def get_risk_tier(probability: float, threshold: float = 0.50) -> Dict[str, Any]:
    """Determines risk classification tier, styling, and icon based on probability and threshold.
    
    Tiers:
    - Low Risk: prob < 0.35 (or < threshold - 0.15)
    - Moderate Risk: 0.35 <= prob < threshold (or borderline)
    - High Risk: prob >= threshold
    """
    pct = probability * 100
    
    if probability < 0.35:
        return {
            "tier": "LOW RISK",
            "badge_color": PALETTE["neon_green"],
            "text_color": "#0B3D0B",
            "icon": "🟢",
            "action_text": "Standard approval process recommended.",
            "summary": "Applicant profile shows strong historical indicators of creditworthiness with substantial liquidity buffers.",
        }
    elif probability < threshold:
        return {
            "tier": "MODERATE RISK",
            "badge_color": PALETTE["yellow"],
            "text_color": "#593C00",
            "icon": "🟡",
            "action_text": "Enhanced verification recommended.",
            "summary": "Applicant shows moderate default probability. Review debt-to-income, collateral, and secondary income sources before finalizing.",
        }
    else:
        return {
            "tier": "HIGH RISK",
            "badge_color": PALETTE["coral_red"],
            "text_color": "#FFFFFF",
            "icon": "🔴",
            "action_text": "Manual underwriter review required.",
            "summary": "Model flags significant default risk factors based on past default patterns. Consider requiring a guarantor, lower principal, or structured repayment.",
        }


def generate_applicant_explanation(
    data: Dict[str, Any],
    probability: float,
    threshold: float,
) -> List[Dict[str, str]]:
    """Generates plain-language bullet points explaining specific applicant risk factors."""
    reasons = []

    # Checking account
    chk = data.get("checking_status", "")
    if "< 0 DM" in chk:
        reasons.append({
            "type": "risk",
            "title": "Overdrawn Checking Balance",
            "desc": "An overdrawn or negative checking account is statistically one of the strongest historical default indicators in the German Credit dataset.",
        })
    elif ">= 200 DM" in chk:
        reasons.append({
            "type": "positive",
            "title": "Healthy Cash Reserves",
            "desc": "Checking balance above 200 DM indicates immediate liquidity and steady cash flow buffer.",
        })
    elif "0 - 200 DM" in chk:
        reasons.append({
            "type": "positive",
            "title": "Positive Operating Cash Balance",
            "desc": "Applicant maintains positive active checking balances and ongoing account activity.",
        })
    elif "No Checking" in chk:
        reasons.append({
            "type": "risk" if probability >= threshold else "positive",
            "title": "No Documented Bank Checking Account",
            "desc": "Lack of local transactional bank records limits direct visibility into routine cash flows.",
        })

    # Loan duration & amount
    dur = data.get("duration_months", 12)
    amt = data.get("credit_amount", 1000)
    if dur >= 36:
        reasons.append({
            "type": "risk",
            "title": f"Extended Loan Term ({dur} months)",
            "desc": "Longer repayment durations create longer exposure horizons where macro or life disruptions can trigger repayment failure.",
        })
    if amt >= 6000:
        reasons.append({
            "type": "risk",
            "title": f"High Principal Request ({amt:,.0f} DM)",
            "desc": "Large credit obligations relative to typical retail averages elevate default risk in historical cohorts.",
        })
    elif amt <= 2000 and dur <= 18:
        reasons.append({
            "type": "positive",
            "title": "Compact Loan Structure",
            "desc": f"Modest principal ({amt} DM) over a short tenor ({dur} months) minimizes credit exposure.",
        })

    # Savings
    sav = data.get("savings_status", "")
    if ">= 1000 DM" in sav:
        reasons.append({
            "type": "positive",
            "title": "Robust Emergency Savings Buffer",
            "desc": "High documented savings balance demonstrates financial resilience during income interruptions.",
        })
    elif "500 - 1000 DM" in sav or "100 - 500 DM" in sav:
        reasons.append({
            "type": "positive",
            "title": f"Documented Savings Buffer ({sav})",
            "desc": "Applicant maintains positive liquid deposits that serve as a debt-service cushion.",
        })
    elif "< 100 DM" in sav:
        reasons.append({
            "type": "risk",
            "title": "Minimal Savings Buffer (< 100 DM)",
            "desc": "Thin financial cushion leaves the borrower vulnerable to unexpected economic shocks.",
        })
    elif "Unknown" in sav:
        reasons.append({
            "type": "risk",
            "title": "Unverified Savings Status",
            "desc": "Lack of verified liquid savings reserves reduces confidence in emergency contingency funds.",
        })

    # Employment
    emp = data.get("employment_duration", "")
    if ">= 7 years" in emp or "4 - 7 years" in emp:
        reasons.append({
            "type": "positive",
            "title": f"Stable Employment Tenure ({emp})",
            "desc": "Long continuous employment tenure is strongly associated with predictable earnings.",
        })
    elif "1 - 4 years" in emp:
        reasons.append({
            "type": "positive",
            "title": f"Established Employment Tenure ({emp})",
            "desc": "Continuous medium-term vocational experience demonstrates stable earnings continuity.",
        })
    elif "Unemployed" in emp or "< 1 year" in emp:
        reasons.append({
            "type": "risk",
            "title": f"Employment Tenure ({emp})",
            "desc": "Short or interrupted employment histories correlate with higher income volatility.",
        })

    # Credit History
    hist = data.get("credit_history", "")
    if "Delay" in hist:
        reasons.append({
            "type": "risk",
            "title": "Historical Payment Delinquency",
            "desc": "Past delinquency delays are a primary signal of potential recurring repayment friction.",
        })
    elif "Existing credits paid duly" in hist or "all paid duly" in hist:
        reasons.append({
            "type": "positive",
            "title": "Clean Repayment History",
            "desc": "Existing credit obligations have consistently been serviced on time without defaults.",
        })
    elif "Critical" in hist or "All credits" in hist:
        reasons.append({
            "type": "positive",
            "title": "Active Credit Track Record",
            "desc": "Extensive borrowing history where past obligations have been serviced demonstrates established borrower discipline.",
        })

    # Housing & Property
    prop = data.get("property", "")
    if "Real Estate" in prop:
        reasons.append({
            "type": "positive",
            "title": "Collateral Backing (Real Estate)",
            "desc": "Property ownership offers asset security and roots the applicant in the local community.",
        })
    elif "Car or Other" in prop or "Building society" in prop:
        reasons.append({
            "type": "positive",
            "title": f"Asset Holdings ({prop})",
            "desc": "Documented tangible property holdings provide secondary economic recourse.",
        })
    elif "Unknown / No Property" in prop:
        reasons.append({
            "type": "risk",
            "title": "Unsecured Asset Profile",
            "desc": "Absence of unencumbered physical property increases uncollateralized credit loss given default.",
        })

    # Housing stability
    housing = data.get("housing", "")
    if housing == "Own":
        reasons.append({
            "type": "positive",
            "title": "Homeowner Status",
            "desc": "Home ownership indicates residential stability and reduces recurring rental payment overhead.",
        })

    # Installment rate
    rate = data.get("installment_rate", 2)
    if rate >= 4:
        reasons.append({
            "type": "risk",
            "title": f"High Installment Burden ({rate}% of income)",
            "desc": "Monthly credit commitments consume a heavy portion of disposable income.",
        })
    elif rate <= 1:
        reasons.append({
            "type": "positive",
            "title": f"Low Installment Commitment ({rate}% of income)",
            "desc": "Minimal monthly debt-service footprint relative to disposable income.",
        })

    # Guaranteed fallback
    if not reasons:
        reasons.append({
            "type": "positive" if probability < threshold else "risk",
            "title": "Baseline Underwriting Profile",
            "desc": f"Applicant attributes align with portfolio cohort averages with an estimated default risk of {probability*100:.1f}%.",
        })

    return reasons


def apply_neobrutalist_layout(
    fig: go.Figure,
    title: str = "",
    xaxis_title: str = "",
    yaxis_title: str = "",
    height: int = 440,
) -> go.Figure:
    """Applies high-contrast Neo-brutalist styling to Plotly figures.
    
    Features:
    - Cream background
    - Thick solid black axis lines
    - High-contrast typography (Space Grotesk / Arial Black)
    - Bold borders and gridlines
    """
    fig.update_layout(
        title={
            "text": f"<b>{title.upper()}</b>",
            "font": {"size": 17, "color": PALETTE["black"], "family": "Space Grotesk, Arial Black, sans-serif"},
            "x": 0.02,
            "y": 0.94,
        },
        paper_bgcolor=PALETTE["bg_cream"],
        plot_bgcolor="#FFFFFF",
        font={"family": "Space Grotesk, Inter, sans-serif", "color": PALETTE["black"], "size": 12},
        height=height,
        margin=dict(l=55, r=35, t=55, b=50),
        xaxis=dict(
            title=f"<b>{xaxis_title}</b>",
            showline=True,
            linewidth=2.5,
            linecolor=PALETTE["black"],
            gridcolor="#E5E1D5",
            gridwidth=1,
            zeroline=False,
            tickfont=dict(color=PALETTE["black"], size=11, family="monospace"),
        ),
        yaxis=dict(
            title=f"<b>{yaxis_title}</b>",
            showline=True,
            linewidth=2.5,
            linecolor=PALETTE["black"],
            gridcolor="#E5E1D5",
            gridwidth=1,
            zeroline=False,
            tickfont=dict(color=PALETTE["black"], size=11, family="monospace"),
        ),
        legend=dict(
            bgcolor="#FFFFFF",
            bordercolor=PALETTE["black"],
            borderwidth=2,
            font=dict(family="Space Grotesk, sans-serif", size=11, color=PALETTE["black"]),
        ),
    )
    return fig
