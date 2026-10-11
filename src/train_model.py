"""Model Training and Comparison Module for CreditGuard.

Trains Logistic Regression, Random Forest, and Gradient Boosting models,
evaluates all three, extracts feature importances and coefficients,
and saves pipelines and artifacts to disk.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data_preprocessing import (
    BASE_DIR,
    PROCESSED_TEST_PATH,
    PROCESSED_TRAIN_PATH,
    TARGET_COLUMN,
    build_preprocessor,
    clean_data,
    load_raw_data,
    split_and_save_data,
)
from src.evaluate_model import evaluate_pipeline

MODELS_DIR = BASE_DIR / "models"


def format_feature_name(name: str) -> str:
    """Formats one-hot encoded or numeric feature names into human-readable labels."""
    if name.startswith("num__"):
        clean = name[len("num__"):]
        mapping = {
            "duration_months": "Credit Duration (Months)",
            "credit_amount": "Credit Amount (DM)",
            "installment_rate": "Installment Rate (% of Income)",
            "residence_duration": "Residence Duration (Years)",
            "age_years": "Applicant Age (Years)",
            "existing_credits": "Existing Credits Count",
            "num_dependents": "Maintenance Dependents Count",
        }
        return mapping.get(clean, clean.replace("_", " ").title())
    elif name.startswith("cat__"):
        clean = name[len("cat__"):]
        # Split prefix column and category value
        for col in [
            "checking_status",
            "credit_history",
            "purpose",
            "savings_status",
            "employment_duration",
            "personal_status_sex",
            "other_debtors",
            "property",
            "other_installment_plans",
            "housing",
            "job_type",
            "telephone",
            "foreign_worker",
        ]:
            prefix = col + "_"
            if clean.startswith(prefix):
                val = clean[len(prefix):]
                col_name = col.replace("_", " ").title()
                return f"{col_name}: {val}"
        return clean.replace("_", " ").title()
    return name


def train_and_compare_models(
    models_dir: Path = MODELS_DIR,
) -> Dict[str, Any]:
    """Trains and compares Logistic Regression, Random Forest, and Gradient Boosting.
    
    Returns:
        Dictionary of trained pipelines, comparison metrics, feature importances, and metadata.
    """
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ensure train and test data exist
    if not PROCESSED_TRAIN_PATH.exists() or not PROCESSED_TEST_PATH.exists():
        print("Processed data not found. Running data preprocessing first...")
        raw_df = load_raw_data()
        cleaned_df = clean_data(raw_df)
        train_df, test_df = split_and_save_data(cleaned_df)
    else:
        train_df = pd.read_csv(PROCESSED_TRAIN_PATH)
        test_df = pd.read_csv(PROCESSED_TEST_PATH)

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]
    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    print(f"Training dataset: {X_train.shape[0]} rows, {X_train.shape[1]} features")
    print(f"Test dataset:     {X_test.shape[0]} rows, {X_test.shape[1]} features")
    print(f"Default rate in train: {y_train.mean():.1%}, in test: {y_test.mean():.1%}")

    # 2. Define model architectures
    classifiers = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            C=0.4,
            max_iter=1000,
            random_state=42,
            solver="lbfgs",
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=7,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=160,
            learning_rate=0.06,
            max_depth=3,
            subsample=0.85,
            random_state=42,
        ),
    }

    pipelines: Dict[str, Pipeline] = {}
    evaluations: Dict[str, Dict[str, Any]] = {}
    summary_rows: List[Dict[str, Any]] = []

    # 3. Train each model in an isolated Pipeline
    for name, clf in classifiers.items():
        print(f"\nTraining {name} pipeline...")
        preprocessor = build_preprocessor()
        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", clf),
            ]
        )
        pipe.fit(X_train, y_train)
        pipelines[name] = pipe

        # Evaluate on test set
        eval_result = evaluate_pipeline(pipe, X_test, y_test, threshold=0.5)
        evaluations[name] = eval_result
        metrics = eval_result["metrics"]

        summary_rows.append(
            {
                "Model": name,
                "Accuracy": f"{metrics['Accuracy']:.3f}",
                "Precision": f"{metrics['Precision']:.3f}",
                "Recall": f"{metrics['Recall']:.3f}",
                "F1-Score": f"{metrics['F1-Score']:.3f}",
                "ROC-AUC": f"{metrics['ROC-AUC']:.3f}",
                "_raw_auc": metrics["ROC-AUC"],
                "_raw_recall": metrics["Recall"],
                "_raw_f1": metrics["F1-Score"],
            }
        )
        print(
            f"  -> Accuracy: {metrics['Accuracy']:.3f} | "
            f"Precision: {metrics['Precision']:.3f} | "
            f"Recall: {metrics['Recall']:.3f} | "
            f"F1-Score: {metrics['F1-Score']:.3f} | "
            f"ROC-AUC: {metrics['ROC-AUC']:.3f}"
        )

    summary_df = pd.DataFrame(summary_rows)

    # 4. Model Selection based on ROC-AUC, Recall, F1, and business suitability
    # In credit default risk, false negatives (approving a defaulter) cause large financial losses.
    # Therefore, we weight ROC-AUC (discriminative ability) and Recall heavily.
    def business_score(row: pd.Series) -> float:
        return (
            0.45 * float(row["_raw_auc"])
            + 0.35 * float(row["_raw_recall"])
            + 0.20 * float(row["_raw_f1"])
        )

    summary_df["Business_Score"] = summary_df.apply(business_score, axis=1)
    best_row = summary_df.sort_values(by="Business_Score", ascending=False).iloc[0]
    best_model_name = str(best_row["Model"])
    best_pipeline = pipelines[best_model_name]
    print(f"\nSelected Best Model: '{best_model_name}' (Business Score: {best_row['Business_Score']:.3f})")

    # 5. Extract Feature Names and Importances
    preprocessor = best_pipeline.named_steps["preprocessor"]
    raw_feature_names = preprocessor.get_feature_names_out()
    human_feature_names = [format_feature_name(f) for f in raw_feature_names]

    # Random Forest Importances
    rf_pipe = pipelines["Random Forest"]
    rf_clf = rf_pipe.named_steps["classifier"]
    rf_importances = rf_clf.feature_importances_
    rf_feat_df = pd.DataFrame(
        {
            "raw_feature": raw_feature_names,
            "feature": human_feature_names,
            "importance": rf_importances,
        }
    ).sort_values(by="importance", ascending=False).reset_index(drop=True)

    # Logistic Regression Coefficients
    lr_pipe = pipelines["Logistic Regression"]
    lr_clf = lr_pipe.named_steps["classifier"]
    lr_coefs = lr_clf.coef_[0]
    lr_coef_df = pd.DataFrame(
        {
            "raw_feature": raw_feature_names,
            "feature": human_feature_names,
            "coefficient": lr_coefs,
            "abs_coefficient": np.abs(lr_coefs),
        }
    ).sort_values(by="abs_coefficient", ascending=False).reset_index(drop=True)

    feature_insights = {
        "random_forest_importance": rf_feat_df.to_dict(orient="records"),
        "logistic_coefficients": lr_coef_df.to_dict(orient="records"),
        "top_risk_drivers": lr_coef_df[lr_coef_df["coefficient"] > 0]
        .sort_values(by="coefficient", ascending=False)
        .head(10)
        .to_dict(orient="records"),
        "top_protective_factors": lr_coef_df[lr_coef_df["coefficient"] < 0]
        .sort_values(by="coefficient", ascending=True)
        .head(10)
        .to_dict(orient="records"),
        "feature_names": human_feature_names,
    }

    # Clean display summary table
    clean_summary_df = summary_df.drop(
        columns=["_raw_auc", "_raw_recall", "_raw_f1", "Business_Score"]
    )

    # 6. Save models and metadata
    joblib.dump(pipelines["Logistic Regression"], models_dir / "logistic_pipeline.joblib")
    joblib.dump(pipelines["Random Forest"], models_dir / "random_forest_pipeline.joblib")
    joblib.dump(pipelines["Gradient Boosting"], models_dir / "gradient_boosting_pipeline.joblib")
    joblib.dump(best_pipeline, models_dir / "best_model_pipeline.joblib")

    comparison_data = {
        "summary_table": clean_summary_df.to_dict(orient="records"),
        "evaluations": evaluations,
        "best_model_name": best_model_name,
    }
    joblib.dump(comparison_data, models_dir / "model_comparison_metrics.joblib")
    joblib.dump(feature_insights, models_dir / "feature_importances.joblib")

    metadata = {
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "total_features": len(human_feature_names),
        "target_column": TARGET_COLUMN,
        "best_model_name": best_model_name,
        "dataset_name": "UCI Statlog German Credit Data",
    }
    joblib.dump(metadata, models_dir / "training_metadata.joblib")

    print("\nAll pipelines and artifacts successfully saved to:", models_dir)
    return {
        "pipelines": pipelines,
        "best_model_name": best_model_name,
        "summary": clean_summary_df,
        "feature_insights": feature_insights,
    }


if __name__ == "__main__":
    train_and_compare_models()
