"""
Supervised Learning: Prediksi Harga Kendaraan Bekas (Regression)
Comparative analysis: Linear Regression, Ridge, and Random Forest Regressor with residual diagnostics.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

sns.set_theme(style="whitegrid")


def generate_vehicle_dataset(n: int = 1500, random_state: int = 42) -> pd.DataFrame:
    """Menghasilkan dataset harga kendaraan bekas dengan relasi non-linier dan depresiasi tahunan."""
    np.random.seed(random_state)
    
    year = np.random.randint(2010, 2024, n)
    age = 2024 - year
    mileage_km = np.random.exponential(scale=15000, size=n) * age.clip(1, 10) + np.random.normal(5000, 2000, n)
    mileage_km = mileage_km.clip(1000, 300000)
    engine_hp = np.random.choice([110, 130, 150, 180, 220, 280, 350], size=n, p=[0.2, 0.25, 0.2, 0.15, 0.1, 0.06, 0.04])
    brand_tier = np.random.choice(["Economy", "Mid-Range", "Premium"], size=n, p=[0.5, 0.35, 0.15])
    
    # Model harga depresiasi
    brand_multiplier = np.where(brand_tier == "Premium", 1.6, np.where(brand_tier == "Mid-Range", 1.15, 0.85))
    base_price = 25000 * brand_multiplier + (engine_hp * 95)
    depreciation = (age * 1800) + (mileage_km * 0.04)
    price = np.maximum(3000, base_price - depreciation + np.random.normal(0, 1500, n))
    
    return pd.DataFrame({
        "year": year,
        "age": age,
        "mileage_km": np.round(mileage_km, 0),
        "engine_hp": engine_hp,
        "brand_tier": brand_tier,
        "price_usd": np.round(price, 0)
    })


def run_pipeline(output_dir: str):
    df = generate_vehicle_dataset(n=2000)
    df_encoded = pd.get_dummies(df, columns=["brand_tier"], drop_first=True)
    
    X = df_encoded.drop(columns=["price_usd"])
    y = df_encoded["price_usd"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Bandingkan Ridge vs Random Forest
    models = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
    }
    
    results = {}
    best_pred = None
    
    print("=" * 50)
    print("HASIL EVALUASI MODEL REGRESI HARGA MOBIL")
    print("=" * 50)
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        results[name] = {"R2": r2, "MAE": mae, "RMSE": rmse}
        print(f"[{name}]")
        print(f"  R2 Score : {r2:.4f}")
        print(f"  MAE      : ${mae:,.2f}")
        print(f"  RMSE     : ${rmse:,.2f}\n")
        if name == "Random Forest Regressor":
            best_pred = preds
            
    # Visualisasi Prediksi vs Aktual & Residuals
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].scatter(y_test, best_pred, alpha=0.35, color="#2980b9", edgecolors="none")
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    axes[0].set_xlabel("Actual Price ($)")
    axes[0].set_ylabel("Predicted Price ($)")
    axes[0].set_title(f"Prediction vs Actual (R² = {results['Random Forest Regressor']['R2']:.3f})")
    
    residuals = y_test - best_pred
    sns.histplot(residuals, kde=True, ax=axes[1], color="#27ae60", bins=30)
    axes[1].set_xlabel("Residual Error ($)")
    axes[1].set_title("Residual Error Distribution")
    
    plt.tight_layout()
    chart_path = os.path.join(output_dir, "regression_evaluation_metrics.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Visualisasi tersimpan di: {chart_path}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    run_pipeline(current_dir)
