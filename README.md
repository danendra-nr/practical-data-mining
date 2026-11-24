# Applied Data Mining & Machine Learning Engineering Portfolio

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/deployment-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Repositori portofolio komprehensif yang menampilkan implementasi **Data Mining, Applied Machine Learning, dan Model Deployment**. Proyek ini dirancang tidak hanya berfokus pada pelatihan algoritma, tetapi juga pada **rekayasa fitur domain finansial/bisnis**, **diagnostik residu & matriks evaluasi imbalanced**, serta **penyajian model interaktif (*model serving*)**.

---

## Proyek Unggulan (Featured Case Studies)

### 1. [Banking Credit Risk Assessment (End-to-End)](./07-end-to-end-case-studies/credit-risk-assessment/)
* **Konteks Bisnis:** Memprediksi probabilitas gagal bayar (*default*) nasabah untuk meminimalisir rasio *Non-Performing Loan* (NPL) perbankan.
* **Metode:** Integrasi dataset Excel multi-sheet relasional, Domain Feature Engineering (PIR & Debt Burden Index), Stratified Split, dan Random Forest Classifier dengan `class_weight='balanced'`.
* **Performa:** **ROC-AUC: 0.7956**, PR-AUC: 0.5657.
* **Demo Interaktif:** Tersedia aplikasi web [Streamlit Demo](./demo/app.py) untuk simulasi persetujuan kredit instan.

### 2. [Customer Churn Prediction](./04-supervised-learning/classification/customer-churn-prediction/)
* **Konteks Bisnis:** Mengidentifikasi pelanggan berisiko tinggi berhenti berlangganan pada industri telekomunikasi untuk intervensi retensi proaktif.
* **Metode:** Pemodelan Random Forest, analisis log-odds, dan threshold probability risk tiers.
* **Performa:** **ROC-AUC: 0.8489**, F1-Score: 0.80.

### 3. [Automotive Price Valuation & Residual Diagnostics](./04-supervised-learning/regression/car-price-prediction/)
* **Konteks Bisnis:** Penaksiran harga pasar wajar (*fair valuation*) kendaraan bekas berbasis depresiasi non-linier.
* **Metode:** Studi komparasi Ridge Regression vs Random Forest Regressor dengan uji normalitas galat residu.
* **Performa:** **$R^2$: 0.9843**, MAE: $1,248.

---

## Kurikulum & Struktur Modul

| Modul | Topik Inti | Metode & Algoritma | Highlight Studi Kasus |
|---|---|---|---|
| **01** | [Data Preprocessing](./01-data-preprocessing/) | Outlier detection, missing value imputation | KNN Imputer, IQR method, SimpleImputer | Sintetis Finansial |
| **02** | [Exploratory Data Analysis](./02-exploratory-data-analysis/) | Univariate, bivariate & correlation analysis | Correlation Heatmap, skewness distribution | [House Price EDA](./02-exploratory-data-analysis/house-price-eda/) |
| **03** | [Feature Engineering](./03-feature-engineering/) | Scaling, categorical encoding, log transformation | RobustScaler, One-Hot, Log1p | Salary & Career Pipeline |
| **04** | [Supervised Learning](./04-supervised-learning/) | Regression, classification, residual analysis | Ridge, Random Forest, Logistic Regression | [Car Price](./04-supervised-learning/regression/car-price-prediction/) & [Churn](./04-supervised-learning/classification/customer-churn-prediction/) |
| **05** | [Unsupervised Learning](./05-unsupervised-learning/) | Customer behavioral clustering | K-Means Clustering, Elbow Method | [Mall Segmentation](./05-unsupervised-learning/customer-segmentation-kmeans/) |
| **06** | [Deep Learning Fundamentals](./06-deep-learning-fundamentals/) | Neural network classification | Multi-Layer Perceptron (MLP) | [MNIST Digits](./06-deep-learning-fundamentals/mnist-digit-recognition/) |
| **07** | [End-to-End Case Studies](./07-end-to-end-case-studies/) | Relational merge, domain engineering & imbalanced target | Random Forest with Balanced Weighting | [Credit Risk Assessment](./07-end-to-end-case-studies/credit-risk-assessment/) |
| **Demo** | [Interactive Web App](./demo/) | Real-time Machine Learning model serving | Streamlit, UI Inference Dashboard | [Credit Decision Engine](./demo/app.py) |

---

## Panduan Menjalankan Proyek

### 1. Kloning Repositori
```bash
git clone https://github.com/danendra-nr/datamining-lrn.git
cd datamining-lrn
```

### 2. Konfigurasi Lingkungan & Instalasi Dependensi
```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Menjalankan Dashboard Demo Interaktif
```bash
streamlit run demo/app.py
```

### 4. Menjalankan Jupyter Lab untuk Riset
```bash
jupyter lab
```

---

## Lisensi
Proyek ini didistribusikan di bawah lisensi terbuka untuk tujuan portofolio profesional dan riset mandiri.
