import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import joblib
import os
from pq_analyzer import extract_signal_features
from prepare_data import CLASS_NAME_MAP

st.set_page_config(page_title="Güç Kalitesi AI Analizörü", layout="wide")
st.title("⚡ Yapay Zeka ve Derin Öğrenme Destekli Güç Kalitesi Analiz Sistemi")

# 1D-CNN Model Mimarisi
class PQ_1DCNN(nn.Module):
    def __init__(self, num_classes=17):
        super(PQ_1DCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

def add_single_awgn(signal, snr_db):
    if snr_db >= 60:
        return signal
    sig_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = sig_power / (snr_linear + 1e-10)
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise

# Sidebar
st.sidebar.header("⚙️ Sinyal ve Şebeke Ayarları")
fs = st.sidebar.number_input("Örnekleme Frekansı (Hz)", value=5000, step=500)
f0 = st.sidebar.selectbox("Şebeke Temel Frekansı (Hz)", [50, 60], index=0)

data_source = st.sidebar.radio("Veri Kaynağı:", ["XPQRS Veri Setinden Test Et", "Dışarıdan CSV Yükle"])

df = None
if data_source == "XPQRS Veri Setinden Test Et" and os.path.exists("data/XPQRS"):
    csv_files = sorted([f for f in os.listdir("data/XPQRS") if f.endswith(".csv")])
    selected_file = st.sidebar.selectbox("Bozulma Sınıfı Seçin:", csv_files)
    df = pd.read_csv(os.path.join("data/XPQRS", selected_file), header=None)
    if df.shape[0] < df.shape[1] and df.shape[1] == 1000:
        df = df.T
else:
    uploaded_file = st.sidebar.file_uploader("CSV Sinyal Dosyası Yükleyin", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, header=None)
        if df.shape[0] < df.shape[1] and df.shape[1] == 1000:
            df = df.T

st.sidebar.markdown("---")
st.sidebar.subheader("🔊 Canlı Şebeke Gürültüsü (AWGN)")
enable_noise = st.sidebar.checkbox("Sinyale Beyaz Gürültü Ekle", value=False)
snr_val = st.sidebar.slider("Gürültü Seviyesi - SNR (dB)", 15, 50, 35) if enable_noise else 100

# sekmeler
tab1, tab2, tab3 = st.tabs([
    "📈 Canlı Sinyal Analizi & AI Teşhisi", 
    "🧠 Açıklanabilir Yapay Zeka (SHAP)", 
    "🛡️ Gürültü Dayanıklılık (Robustness) Raporu"
])

if df is not None:
    row_idx = st.sidebar.slider("Sinyal Örneği No (0 - 999)", 0, len(df)-1, 0)
    clean_signal = df.iloc[row_idx].values.astype(float)
    signal = add_single_awgn(clean_signal, snr_val) if enable_noise else clean_signal
    t = np.arange(len(signal)) / fs * 1000

    features, xf, yf = extract_signal_features(signal, fs=fs, f0=f0)

    with tab1:
        col_m1, col_m2 = st.columns(2)
        
        # Random Forest
        if os.path.exists("models/pq_model_bundle.pkl"):
            rf_model, scaler, selector = joblib.load("models/pq_model_bundle.pkl")
            X_in = scaler.transform(pd.DataFrame([features]))
            if selector is not None:
                X_in = selector.transform(X_in)
            rf_pred = rf_model.predict(X_in)[0]
            rf_prob = np.max(rf_model.predict_proba(X_in)[0]) * 100
            col_m1.success(f"🌲 **Random Forest (Özellik Tabanlı) Kararı:**\n\n**{rf_pred}** (Güven: **%{rf_prob:.1f}**)")

        # 1D-CNN
        if os.path.exists("models/pq_cnn_model.pt") and os.path.exists("models/label_encoder.pkl"):
            encoder = joblib.load("models/label_encoder.pkl")
            cnn_model = PQ_1DCNN(num_classes=len(encoder.classes_))
            cnn_model.load_state_dict(torch.load("models/pq_cnn_model.pt", map_location="cpu"))
            cnn_model.eval()
            with torch.no_grad():
                sig_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                logits = cnn_model(sig_tensor)
                probs = torch.softmax(logits, dim=1).numpy()[0]
                pred_idx = np.argmax(probs)
                raw_cls = encoder.inverse_transform([pred_idx])[0]
                tr_cls = CLASS_NAME_MAP.get(raw_cls, raw_cls)
                cnn_prob = probs[pred_idx] * 100
            col_m2.info(f"🧠 **1D-CNN (Gürültüye Dayanıklı Derin Öğrenme) Kararı:**\n\n**{tr_cls}** (Güven: **%{cnn_prob:.1f}**)")

        g_col1, g_col2 = st.columns(2)
        with g_col1:
            st.subheader("Zaman Domeni Dalga Formu")
            fig1, ax1 = plt.subplots(figsize=(6, 3))
            if enable_noise:
                ax1.plot(t, clean_signal, color="gray", alpha=0.4, linestyle="--", label="Temiz Sinyal")
            ax1.plot(t, signal, color="#1f77b4", linewidth=1.6, label="İncelenen Sinyal")
            ax1.set_xlabel("Zaman (ms)")
            ax1.set_ylabel("Genlik (pu)")
            ax1.legend(loc="upper right", fontsize=8)
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1)

        with g_col2:
            st.subheader("FFT Frekans Spektrumu (Harmonik Analizi)")
            fig2, ax2 = plt.subplots(figsize=(6, 3))
            ax2.stem(xf[:20], yf[:20], basefmt=" ")
            ax2.set_xlabel("Frekans (Hz)")
            ax2.set_ylabel("Genlik")
            ax2.grid(True, alpha=0.3)
            st.pyplot(fig2)

        st.subheader("📐 Çıkarılan 15 Elektriksel ve İstatistiksel Özellik")
        st.dataframe(pd.DataFrame([features]).style.format("{:.4f}"))

    with tab2:
        st.subheader("🔍 SHAP (Açıklanabilir Yapay Zeka) Özellik Önem Dereceleri")
        st.write("Modelin 17 güç kalitesi sınıfını ayırt ederken en çok hangi matematiksel/elektriksel özelliğe dayandığını gösterir:")
        if os.path.exists("outputs/shap_importance.csv"):
            shap_df = pd.read_csv("outputs/shap_importance.csv")
            col_s1, col_s2 = st.columns([2, 1])
            with col_s1:
                fig_s, ax_s = plt.subplots(figsize=(7, 4))
                ax_s.barh(shap_df["Feature"][::-1], shap_df["SHAP_Importance"][::-1], color="#ff7f0e")
                ax_s.set_xlabel("Ortalama Mutlak SHAP Etki Değeri")
                ax_s.set_title("Karar Mekanizmasında En Etkili Özellikler")
                ax_s.grid(True, alpha=0.3)
                st.pyplot(fig_s)
            with col_s2:
                st.dataframe(shap_df.style.format({"SHAP_Importance": "{:.6f}"}))

    with tab3:
        st.subheader("🛡️ AWGN Şebeke Gürültüsü Dayanıklılık Testi (Öncesi ve Sonrası)")
        st.write("Sadece temiz veriyle eğitilen Standart 1D-CNN ile Gürültü Bağışıklığı Eğitimi (Noise-Aware Training) uygulanmış 1D-CNN modelinin karşılaştırması:")
        if os.path.exists("outputs/noise_robustness_results.csv"):
            noise_df = pd.read_csv("outputs/noise_robustness_results.csv")
            col_n1, col_n2 = st.columns([2, 1])
            with col_n1:
                fig_n, ax_n = plt.subplots(figsize=(6.5, 3.8))
                if "Standart_CNN" in noise_df.columns and "Robust_CNN" in noise_df.columns:
                    ax_n.plot(noise_df["SNR_dB"], noise_df["Standart_CNN"], marker="o", linestyle="--", linewidth=2, color="#d62728", label="Standart 1D-CNN (Temiz Veri)")
                    ax_n.plot(noise_df["SNR_dB"], noise_df["Robust_CNN"], marker="s", linewidth=2.5, color="#2ca02c", label="Gürültüye Dayanıklı 1D-CNN (Önerilen)")
                    ax_n.legend(loc="lower left")
                elif "CNN_Accuracy" in noise_df.columns:
                    ax_n.plot(noise_df["SNR_dB"], noise_df["CNN_Accuracy"], marker="o", linewidth=2, color="#d62728", label="1D-CNN")
                    ax_n.legend(loc="lower left")
                ax_n.set_xlabel("Gürültü Seviyesi (SNR - dB)")
                ax_n.set_ylabel("Doğruluk Oranı (%)")
                ax_n.set_ylim(0, 105)
                ax_n.grid(True, alpha=0.3)
                st.pyplot(fig_n)
            with col_n2:
                format_dict = {col: "%{:.2f}" for col in noise_df.columns if col != "SNR_dB"}
                st.dataframe(noise_df.style.format(format_dict))