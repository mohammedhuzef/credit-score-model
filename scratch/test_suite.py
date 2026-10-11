import sys
from pathlib import Path
import traceback

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import pandas as pd
import joblib

def test_data_pipeline():
    print("Testing data preprocessing...")
    from src.data_preprocessing import load_raw_data, clean_data, CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COLUMN, RAW_DATA_PATH
    assert RAW_DATA_PATH.exists(), f"Raw data not found at {RAW_DATA_PATH}"
    df_raw = load_raw_data()
    print(f"Raw data shape: {df_raw.shape}")
    assert len(df_raw) == 1000, f"Expected 1000 rows, got {len(df_raw)}"
    
    df_clean = clean_data(df_raw)
    print(f"Clean data shape: {df_clean.shape}")
    assert TARGET_COLUMN in df_clean.columns
    assert df_clean[TARGET_COLUMN].isin([0, 1]).all()
    
    for col in CATEGORICAL_FEATURES + NUMERIC_FEATURES:
        assert col in df_clean.columns, f"Missing feature {col}"
    print("[PASS] Data pipeline test passed.")

def test_models():
    print("Testing model loading and inference...")
    from src.train_model import MODELS_DIR
    models_to_test = [
        "best_model_pipeline.joblib",
        "logistic_pipeline.joblib",
        "random_forest_pipeline.joblib",
        "gradient_boosting_pipeline.joblib"
    ]
    from src.data_preprocessing import load_raw_data, clean_data, CATEGORICAL_FEATURES, NUMERIC_FEATURES
    df_clean = clean_data(load_raw_data())
    X = df_clean[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    
    for m_name in models_to_test:
        m_path = MODELS_DIR / m_name
        assert m_path.exists(), f"Model file {m_path} does not exist"
        pipe = joblib.load(m_path)
        probs = pipe.predict_proba(X.iloc[:5])
        assert probs.shape == (5, 2), f"Expected shape (5, 2), got {probs.shape}"
        assert (probs >= 0.0).all() and (probs <= 1.0).all()
        print(f"  Model {m_name} predict_proba OK.")

    # Check metrics and feature importances
    comp_metrics = joblib.load(MODELS_DIR / "model_comparison_metrics.joblib")
    assert isinstance(comp_metrics, dict)
    print("  Comparison metrics loaded successfully.")
    
    feat_imp = joblib.load(MODELS_DIR / "feature_importances.joblib")
    assert isinstance(feat_imp, dict)
    print("  Feature importances loaded successfully.")
    print("[PASS] Model test passed.")

def test_persona_presets():
    print("Testing persona presets scoring...")
    from src.utils import PERSONA_PRESETS, get_risk_tier, generate_applicant_explanation
    from src.train_model import MODELS_DIR
    best_pipe = joblib.load(MODELS_DIR / "best_model_pipeline.joblib")
    feat_imp = joblib.load(MODELS_DIR / "feature_importances.joblib")
    
    for p_name, preset in PERSONA_PRESETS.items():
        sample_df = pd.DataFrame([preset])
        prob = float(best_pipe.predict_proba(sample_df)[0][1])
        tier = get_risk_tier(prob, threshold=0.50)
        assert "tier" in tier and "badge_color" in tier
        exp = generate_applicant_explanation(preset, prob, threshold=0.50)
        assert isinstance(exp, list) and len(exp) > 0
        print(f"  Persona '{p_name}': prob={prob:.3f}, tier={tier['tier']}")
    print("[PASS] Persona presets test passed.")

def test_evaluation_functions():
    print("Testing evaluation module functions...")
    from src.evaluate_model import (
        compute_metrics_at_threshold,
        compute_confusion_matrix_data,
        compute_curves_data
    )
    y_true = np.array([0, 1, 0, 1, 0, 0, 1, 1, 0, 0])
    y_prob = np.array([0.1, 0.8, 0.2, 0.65, 0.3, 0.4, 0.7, 0.9, 0.15, 0.35])
    
    metrics = compute_metrics_at_threshold(y_true, y_prob, 0.5)
    assert "Accuracy" in metrics and "F1-Score" in metrics and "ROC-AUC" in metrics
    
    cm = compute_confusion_matrix_data(y_true, y_prob, 0.5)
    assert cm["TP"] >= 0 and cm["FP"] >= 0
    
    curves = compute_curves_data(y_true, y_prob)
    assert "fpr" in curves and "precisions" in curves and "roc_auc" in curves
    print("[PASS] Evaluation module test passed.")

def test_app_imports_and_loaders():
    print("Testing app module and artifact loading...")
    import app
    artifacts = app.load_all_artifacts()
    assert artifacts is not None
    assert "best_pipeline" in artifacts
    assert "metrics" in artifacts
    assert "features" in artifacts
    print("[PASS] App artifact loader test passed.")

if __name__ == "__main__":
    try:
        test_data_pipeline()
        test_models()
        test_persona_presets()
        test_evaluation_functions()
        test_app_imports_and_loaders()
        print("\nALL TEST SUITE CHECKS COMPLETED SUCCESSFULLY!")
    except Exception as e:
        traceback.print_exc()
        sys.exit(1)
