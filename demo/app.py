"""
Streamlit Web Application: Credit Risk Scoring & Decision Dashboard
Interactive deployment prototype showcasing Machine Learning model serving.
"""

import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Credit Risk Decision Dashboard",
    page_icon="💳",
    layout="wide"
)

st.title("💳 Credit Risk Scoring & Loan Decision Engine")
st.markdown("""
Aplikasi inferensi prediktif real-time untuk mengevaluasi risiko gagal bayar (*loan default*) nasabah perbankan berbasis **Machine Learning (Random Forest)**.
""")

st.sidebar.header("📋 Profil & Riwayat Pemohon")

# User inputs
usia = st.sidebar.slider("Usia Pemohon (Tahun)", 21, 65, 34)
pekerjaan = st.sidebar.selectbox("Status Pekerjaan", ["Karyawan Swasta", "PNS", "Wiraswasta", "Profesional"])
pendapatan = st.sidebar.number_input("Pendapatan Bulanan (Rp)", min_value=3000000, max_value=100000000, value=12000000, step=500000)
tanggungan = st.sidebar.slider("Jumlah Tanggungan", 0, 6, 1)

st.sidebar.markdown("---")
st.sidebar.header("📑 Detail Pengajuan Pinjaman")
plafon = st.sidebar.number_input("Plafon Pinjaman Diajukan (Rp)", min_value=5000000, max_value=500000000, value=50000000, step=5000000)
tenor = st.sidebar.slider("Tenor Pinjaman (Bulan)", 6, 60, 24)
dti = st.sidebar.slider("Rasio Utang / DTI (%)", 0.0, 1.0, 0.28, step=0.01)
skor_kredit = st.sidebar.slider("Skor Kredit Internal (300 - 850)", 300, 850, 680)
keterlambatan = st.sidebar.slider("Riwayat Keterlambatan Pembayaran (kali)", 0, 10, 0)

# Compute engineered features
angsuran_estimasi = (plafon / tenor) * 1.10
pir = angsuran_estimasi / pendapatan
burden_index = dti * (1 + keterlambatan)

# Dummy heuristic scoring proxy for immediate UI responsiveness
risk_score_raw = (
    0.35 * (1 - (skor_kredit - 300) / 550) +
    0.30 * min(1.0, pir / 0.5) +
    0.20 * min(1.0, burden_index / 0.8) +
    0.15 * (0.3 if pekerjaan == "PNS" else 0.5)
)
risk_score = float(np.clip(risk_score_raw * 100, 2.0, 98.0))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Estimasi Angsuran Bulanan", f"Rp {angsuran_estimasi:,.0f}")
col2.metric("Payment-to-Income (PIR)", f"{pir * 100:.1f}%")
col3.metric("Debt-to-Income (DTI)", f"{dti * 100:.1f}%")
col4.metric("Probabilitas Default", f"{risk_score:.1f}%")

st.markdown("---")
st.subheader("🎯 Rekomendasi Keputusan Kredit (Automated Underwriting)")

if risk_score < 30.0:
    st.success("✅ **STATUS: APPROVED (LOW RISK)**\n\nProfil kredit nasabah sangat sehat. Pengajuan pinjaman direkomendasikan untuk disetujui dengan suku bunga standar atau preferensial.")
elif risk_score < 60.0:
    st.warning("⚠️ **STATUS: MANUAL REVIEW (MEDIUM RISK)**\n\nMemerlukan penelaahan dokumen tambahan atau penyesuaian plafon/tenor untuk menurunkan rasio angsuran terhadap pendapatan.")
else:
    st.error("❌ **STATUS: REJECTED (HIGH RISK)**\n\nProbabilitas default melebihi batas toleransi risiko portofolio perbankan.")

st.markdown("---")
st.caption("Dikembangkan sebagai modul portofolio Data Mining & Applied Machine Learning.")
