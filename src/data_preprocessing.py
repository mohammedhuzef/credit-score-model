"""Data Preprocessing Module for CreditGuard.

Handles data loading, cleaning, imputation, transformation, and train-test splitting.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os
import urllib.request
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Project root paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "german_credit.csv"
PROCESSED_TRAIN_PATH = BASE_DIR / "data" / "processed" / "train.csv"
PROCESSED_TEST_PATH = BASE_DIR / "data" / "processed" / "test.csv"

# UCI German Credit Data URL (official archive)
UCI_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

# Standard German Credit Column Names
RAW_COLUMN_NAMES: List[str] = [
    "checking_status",
    "duration_months",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_status",
    "employment_duration",
    "installment_rate",
    "personal_status_sex",
    "other_debtors",
    "residence_duration",
    "property",
    "age_years",
    "other_installment_plans",
    "housing",
    "existing_credits",
    "job_type",
    "num_dependents",
    "telephone",
    "foreign_worker",
    "default_risk",
]

# Attribute value decoders based on UCI German Credit documentation
CATEGORICAL_MAPPINGS: Dict[str, Dict[str, str]] = {
    "checking_status": {
        "A11": "< 0 DM (Overdrawn)",
        "A12": "0 - 200 DM (Moderate)",
        "A13": ">= 200 DM (Healthy)",
        "A14": "No Checking Account",
    },
    "credit_history": {
        "A30": "No credits / all paid duly",
        "A31": "All credits at this bank paid duly",
        "A32": "Existing credits paid duly till now",
        "A33": "Delay in paying off in the past",
        "A34": "Critical account / other credits elsewhere",
    },
    "purpose": {
        "A40": "Car (new)",
        "A41": "Car (used)",
        "A42": "Furniture / Equipment",
        "A43": "Radio / Television",
        "A44": "Domestic appliances",
        "A45": "Repairs",
        "A46": "Education",
        "A47": "Vacation",
        "A48": "Retraining",
        "A49": "Business",
        "A410": "Others",
    },
    "savings_status": {
        "A61": "< 100 DM",
        "A62": "100 - 500 DM",
        "A63": "500 - 1000 DM",
        "A64": ">= 1000 DM",
        "A65": "Unknown / No Savings",
    },
    "employment_duration": {
        "A71": "Unemployed",
        "A72": "< 1 year",
        "A73": "1 - 4 years",
        "A74": "4 - 7 years",
        "A75": ">= 7 years",
    },
    "personal_status_sex": {
        "A91": "Male: Divorced / Separated",
        "A92": "Female: Divorced / Married",
        "A93": "Male: Single",
        "A94": "Male: Married / Widowed",
        "A95": "Female: Single",
    },
    "other_debtors": {
        "A101": "None",
        "A102": "Co-applicant",
        "A103": "Guarantor",
    },
    "property": {
        "A121": "Real Estate",
        "A122": "Building Society / Life Insurance",
        "A123": "Car or Other Assets",
        "A124": "Unknown / No Property",
    },
    "other_installment_plans": {
        "A141": "Bank",
        "A142": "Stores",
        "A143": "None",
    },
    "housing": {
        "A151": "Rent",
        "A152": "Own",
        "A153": "For Free",
    },
    "job_type": {
        "A171": "Unemployed / Unskilled Non-resident",
        "A172": "Unskilled Resident",
        "A173": "Skilled Employee / Official",
        "A174": "Management / Self-employed / Highly Qualified",
    },
    "telephone": {
        "A191": "None",
        "A192": "Yes (Registered)",
    },
    "foreign_worker": {
        "A201": "Yes",
        "A202": "No",
    },
}

# Numeric and categorical feature separation
NUMERIC_FEATURES: List[str] = [
    "duration_months",
    "credit_amount",
    "installment_rate",
    "residence_duration",
    "age_years",
    "existing_credits",
    "num_dependents",
]

CATEGORICAL_FEATURES: List[str] = [
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
]

TARGET_COLUMN = "default_risk"


def download_uci_dataset(destination_path: Optional[Path] = None) -> Path:
    """Download the raw UCI German Credit dataset if not already present.
    
    Args:
        destination_path: Target path to save the dataset. Defaults to RAW_DATA_PATH.
        
    Returns:
        Path to the saved CSV file.
    """
    dest = destination_path or RAW_DATA_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    
    print(f"Downloading German Credit dataset from {UCI_DATA_URL}...")
    try:
        # Download raw space-separated file
        temp_file, _ = urllib.request.urlretrieve(UCI_DATA_URL)
        df_raw = pd.read_csv(temp_file, sep=" ", header=None, names=RAW_COLUMN_NAMES)
        
        # Apply human-readable mappings
        for col, mapping in CATEGORICAL_MAPPINGS.items():
            if col in df_raw.columns:
                df_raw[col] = df_raw[col].map(mapping).fillna(df_raw[col])
        
        # In original UCI dataset: 1 = Good credit (0: non-default), 2 = Bad credit (1: default)
        df_raw[TARGET_COLUMN] = df_raw[TARGET_COLUMN].map({1: 0, 2: 1})
        
        df_raw.to_csv(dest, index=False)
        print(f"Successfully saved raw dataset to {dest} ({len(df_raw)} records)")
        return dest
    except Exception as e:
        print(f"Failed to download online dataset: {e}")
        # If offline fallback is needed, check if file exists
        if dest.exists():
            return dest
        raise RuntimeError(
            f"Could not download dataset from {UCI_DATA_URL} and no local file found at {dest}. "
            f"Please place 'german_credit.csv' inside {dest.parent}."
        ) from e


def load_raw_data(data_path: Optional[Path] = None) -> pd.DataFrame:
    """Safely loads the dataset from disk or downloads it if absent.
    
    Args:
        data_path: Optional path to the CSV file.
        
    Returns:
        pd.DataFrame containing the raw data.
    """
    path = data_path or RAW_DATA_PATH
    if not path.exists():
        path = download_uci_dataset(path)
    
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans data by removing duplicates, handling invalid values, and imputing.
    
    - Drops duplicate rows.
    - Ensures target column is binary (0 or 1).
    - Fills numeric missing values with column median.
    - Fills categorical missing values with 'Unknown'.
    
    Args:
        df: Input DataFrame.
        
    Returns:
        Cleaned pd.DataFrame.
    """
    df_clean = df.copy()
    
    # Drop duplicates
    initial_count = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    dropped = initial_count - len(df_clean)
    if dropped > 0:
        print(f"Dropped {dropped} duplicate rows.")
        
    # Standardize target if needed
    if TARGET_COLUMN in df_clean.columns:
        # Check if values are 1/2 or other
        unique_targets = set(df_clean[TARGET_COLUMN].dropna().unique())
        if unique_targets == {1, 2}:
            df_clean[TARGET_COLUMN] = df_clean[TARGET_COLUMN].map({1: 0, 2: 1})
        elif not unique_targets.issubset({0, 1}):
            raise ValueError(f"Unexpected target values in {TARGET_COLUMN}: {unique_targets}")
            
    # Impute numeric missing values with median
    for col in NUMERIC_FEATURES:
        if col in df_clean.columns:
            # Coerce invalid to numeric
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val)
            
    # Impute categorical missing values with 'Unknown'
    for col in CATEGORICAL_FEATURES:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).replace({"nan": "Unknown", "None": "Unknown", "": "Unknown"}).fillna("Unknown")
            
    return df_clean


def build_preprocessor() -> ColumnTransformer:
    """Constructs a Scikit-Learn ColumnTransformer for numeric and categorical pipelines.
    
    - Numeric features: Median imputation followed by StandardScaler.
    - Categorical features: Constant imputation ('Unknown') followed by OneHotEncoder.
    
    Returns:
        ColumnTransformer instance.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def split_and_save_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
    processed_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits data into train and test sets with stratification and saves them.
    
    Args:
        df: Cleaned DataFrame.
        test_size: Proportion of test data (default 0.20).
        random_state: Seed for reproducibility (default 42).
        processed_dir: Output folder for train.csv and test.csv.
        
    Returns:
        Tuple of (train_df, test_df).
    """
    target = df[TARGET_COLUMN]
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    out_dir = processed_dir or (BASE_DIR / "data" / "processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = out_dir / "train.csv"
    test_path = out_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Train split saved to {train_path} ({len(train_df)} rows)")
    print(f"Test split saved to {test_path} ({len(test_df)} rows)")

    return train_df, test_df


if __name__ == "__main__":
    print("Executing data preprocessing pipeline...")
    raw_df = load_raw_data()
    cleaned_df = clean_data(raw_df)
    train_data, test_data = split_and_save_data(cleaned_df)
    print("Preprocessing completed successfully.")
