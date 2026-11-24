"""
Supervised Learning: Customer Churn Prediction
Predicting customer churn probability using Random Forest & Logistic Regression with threshold analysis.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay

sns.set_theme(style="whitegrid")


def generate_telecom_churn_data(n: int = 1500, random_state: int = 42) -> pd.DataFrame:
    """
    Menghasilkan dataset churn telekomunikasi dengan distribusi statistik realistis,
    termasuk kontrak, tagihan bulanan, masa berlangganan, dan keluhan layanan.
    """
    np.random.seed(random_state)
    
    tenure_months = np.random.exponential(scale=24, size=n).clip(1, 72).astype(int)
    contract_type = np.random.choice(["Month-to-Month", "One-Year", "Two-Year"], size=n, p=[0.55, 0.25, 0.20])
    monthly_charges = np.random.normal(loc=70, scale=25, size=n).clip(20, 130)
    customer_service_calls = np.random.poisson(lam=1.5, size=n).clip(0, 9)
    tech_support = np.random.choice([0, 1], size=n, p=[0.65, 0.35])
    
    # Perhitungan log-odds probabilitas churn berdasarkan faktor bisnis
    contract_weight = np.where(contract_type == "Month-to-Month", 1.2, np.where(contract_type == "One-Year", -0.4, -1.5))
    log_odds = (
        -0.8
        - 0.04 * tenure_months
        + 0.02 * monthly_charges
        + 0.45 * customer_service_calls
        - 0.6 * tech_support
        + contract_weight
        + np.random.normal(0, 0.4, n)
    )
    prob_churn = 1 / (1 + np.exp(-log_odds))
    churn = (np.random.rand(n) < prob_churn).astype(int)
    
    return pd.DataFrame({
        "tenure_months": tenure_months,
        "contract_type": contract_type,
        "monthly_charges": np.round(monthly_charges, 2),
        "customer_service_calls": customer_service_calls,
        "tech_support": tech_support,
        "churn": churn
    })


def run_pipeline(output_dir: str):
    df = generate_telecom_churn_data(n=2000)
    
    # Feature Engineering & Encoding
    df_encoded = pd.get_dummies(df, columns=["contract_type"], drop_first=True)
    
    X = df_encoded.drop(columns=["churn"])
    y = df_encoded["churn"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Training Model
    rf_model = RandomForestClassifier(n_estimators=200, max_depth=7, random_state=42, class_weight="balanced")
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_prob)
    print("=" * 50)
    print("EVALUASI MODEL CUSTOMER CHURN PREDICTION")
    print("=" * 50)
    print(f"ROC-AUC Score: {roc_auc:.4f}\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Retain (0)", "Churn (1)"]))
    
    # Visualisasi
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Retain", "Churn"]).plot(ax=axes[0], cmap="Reds")
    axes[0].set_title("Confusion Matrix (Customer Churn)")
    axes[0].grid(False)
    
    # Feature Importance
    importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=True)
    importances.plot(kind="barh", ax=axes[1], color="#e74c3c")
    axes[1].set_title("Key Drivers of Customer Churn")
    axes[1].set_xlabel("Relative Importance")
    
    plt.tight_layout()
    chart_path = os.path.join(output_dir, "churn_model_evaluation.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Visualisasi tersimpan di: {chart_path}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    run_pipeline(current_dir)
