"""
Credit Risk Assessment Pipeline & Visual Evaluation
End-to-end Machine Learning pipeline for credit default prediction.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_curve,
    average_precision_score,
)

# Set visual aesthetic
sns.set_theme(style="whitegrid")
plt.rcParams.update({"font.sans-serif": "DejaVu Sans", "font.size": 10})


def load_data(file_path: str) -> pd.DataFrame:
    """Membaca dataset multi-sheet dari file Excel dan menggabungkannya."""
    sheet_debitur = pd.read_excel(file_path, sheet_name="Data_Debitur")
    sheet_pinjaman = pd.read_excel(file_path, sheet_name="Riwayat_Pinjaman")
    df_merged = pd.merge(sheet_debitur, sheet_pinjaman, on="id_nasabah")
    return df_merged


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Data cleaning, handling missing values, and domain-specific feature engineering."""
    df_clean = df.copy()

    # Imputasi missing values berdasarkan distribusi data
    mode_pekerjaan = df_clean["status_pekerjaan"].mode()[0]
    df_clean["status_pekerjaan"] = df_clean["status_pekerjaan"].fillna(mode_pekerjaan)

    median_skor = df_clean["skor_kredit_internal"].median()
    df_clean["skor_kredit_internal"] = df_clean["skor_kredit_internal"].fillna(median_skor)

    # Domain Feature Engineering:
    # 1. Estimasi Angsuran Bulanan dengan bunga flat 10%
    df_clean["angsuran_estimasi"] = (df_clean["plafon_pinjaman"] / df_clean["tenor_bulan"]) * 1.10
    # 2. Payment-to-Income Ratio (PIR)
    df_clean["rasio_angsuran_pendapatan"] = np.round(
        df_clean["angsuran_estimasi"] / df_clean["pendapatan_bulanan"], 4
    )
    # 3. Debt Burden Index (kombinasi DTI dan jumlah keterlambatan masa lalu)
    df_clean["burden_index"] = df_clean["rasio_dti"] * (1 + df_clean["jumlah_keterlambatan"])

    # One-Hot Encoding
    df_encoded = pd.get_dummies(df_clean, columns=["status_pekerjaan"], drop_first=True)
    return df_encoded


def train_and_evaluate(df: pd.DataFrame, output_dir: str):
    """Melatih Random Forest Classifier, mengevaluasi performa, dan menyimpan visualisasi."""
    feature_cols = [
        "usia",
        "pendapatan_bulanan",
        "jumlah_tanggungan",
        "skor_kredit_internal",
        "plafon_pinjaman",
        "tenor_bulan",
        "rasio_dti",
        "jumlah_keterlambatan",
        "angsuran_estimasi",
        "rasio_angsuran_pendapatan",
        "burden_index",
        "status_pekerjaan_Karyawan Swasta",
        "status_pekerjaan_PNS",
        "status_pekerjaan_Wiraswasta",
    ]
    feature_cols = [col for col in feature_cols if col in df.columns]

    X = df[feature_cols]
    y = df["status_gagal_bayar"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Inisialisasi model dengan penanganan imbalanced class
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    print("=" * 50)
    print("HASIL EVALUASI MODEL CREDIT RISK")
    print("=" * 50)
    print(f"ROC-AUC Score : {roc_auc:.4f}")
    print(f"PR-AUC Score  : {pr_auc:.4f}\n")
    print("Laporan Klasifikasi:")
    print(classification_report(y_test, y_pred, target_names=["Lancar (0)", "Default (1)"]))

    # Visualisasi
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=["Lancar", "Default"]).plot(
        ax=axes[0], cmap="Blues", values_format="d"
    )
    axes[0].set_title("Confusion Matrix (Default Detection)")
    axes[0].grid(False)

    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[1].plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].legend(loc="lower right")

    # 3. Top Feature Importances
    importances = pd.Series(rf_model.feature_importances_, index=feature_cols).sort_values(
        ascending=True
    )
    importances.tail(7).plot(kind="barh", ax=axes[2], color="#2ca02c")
    axes[2].set_title("Top 7 Predictors of Credit Default")
    axes[2].set_xlabel("Importance Weight")

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "model_evaluation_metrics.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Visualisasi berhasil disimpan di: {chart_path}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "dataset_kredit.xlsx")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} tidak ditemukan.")

    df = load_data(file_path)
    df_processed = preprocess_data(df)
    train_and_evaluate(df_processed, current_dir)
