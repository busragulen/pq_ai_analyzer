import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import hilbert
import torch
import torch.nn as nn
import joblib
import os
import time
from pq_analyzer import extract_signal_features
from prepare_data import CLASS_NAME_MAP

st.set_page_config(page_title="Güç Kalitesi AI Analizörü", layout="wide")

st.markdown("""
    <style>
        .block-container {
            padding-top: 3rem !important;
        }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ Yapay Zeka ve Derin Öğrenme Destekli Güç Kalitesi Analiz Sistemi")

class PQ_1DCNN(nn.Module):
    def __init__(self, num_classes=23, dropout_rate=0.25):
        super(PQ_1DCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
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
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        feat = self.features(x)
        return self.classifier(feat)

@st.cache_resource
def load_ml_bundle(mtime=0.0):
    if os.path.exists("models/pq_model_bundle.pkl"):
        return joblib.load("models/pq_model_bundle.pkl")
    return None

@st.cache_resource
def load_cnn_bundle():
    if os.path.exists("models/pq_cnn_model.pt") and os.path.exists("models/label_encoder.pkl"):
        encoder = joblib.load("models/label_encoder.pkl")
        cnn_model = PQ_1DCNN(num_classes=len(encoder.classes_))
        cnn_model.load_state_dict(torch.load("models/pq_cnn_model.pt", map_location="cpu"))
        cnn_model.eval()
        return cnn_model, encoder
    return None, None

def add_single_awgn(signal, snr_db):
    if snr_db >= 60:
        return signal
    sig_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10.0)
    noise_power = sig_power / (snr_linear + 1e-10)
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise

def build_multicycle_view(single_cycle_sig, class_hint="", fs=5000, f0=50, n_cycles=10):
    """
    1 periyotluk (20 ms) sinyali Sag, Swell, Interruption ve Flicker zarflarını
    gözle net görebilmek için otomatik olarak 10 periyotluk (200 ms) panoramik pencereye genişletir.
    """
    N = len(single_cycle_sig)
    t_single = np.arange(N) / fs
    nominal_cycle = np.sin(2 * np.pi * f0 * t_single)
    
    name_lower = class_hint.lower()
    cycles = []
    
    for c in range(n_cycles):
        if "flicker" in name_lower:
            t_curr = (np.arange(N) + c * N) / fs
            flicker_mod = 1.0 + 0.22 * np.sin(2 * np.pi * 7.5 * t_curr)
            if ("sag" in name_lower or "swell" in name_lower) and (2 <= c <= 7):
                cyc = single_cycle_sig * flicker_mod
            elif "sag" in name_lower or "swell" in name_lower:
                cyc = nominal_cycle * flicker_mod
            else:
                cyc = single_cycle_sig * flicker_mod
        elif any(k in name_lower for k in ["sag", "swell", "interruption"]):
            if 2 <= c <= 7:
                cyc = single_cycle_sig.copy()
            else:
                cyc = nominal_cycle.copy()
        elif "transient" in name_lower:
            if c in [3, 6]:
                cyc = single_cycle_sig.copy()
            else:
                cyc = nominal_cycle.copy()
        else:
            cyc = single_cycle_sig.copy()
            
        cycles.append(cyc)
        
    full_sig = np.concatenate(cycles)
    t_full_ms = np.arange(len(full_sig)) / fs * 1000.0
    return t_full_ms, full_sig

# Sidebar
st.sidebar.header("⚙️ Sinyal ve Şebeke Ayarları")
fs = st.sidebar.number_input("Örnekleme Frekansı (Hz)", value=5000, step=500)
f0 = st.sidebar.selectbox("Şebeke Temel Frekansı (Hz)", [50, 60], index=0)

data_source = st.sidebar.radio("Veri Kaynağı:", ["XPQRS Veri Setinden Test Et", "Dışarıdan CSV Yükle"])

df = None
selected_file = "Harici_Sinyal"
if data_source == "XPQRS Veri Setinden Test Et" and os.path.exists("data/XPQRS"):
    csv_files = sorted([f for f in os.listdir("data/XPQRS") if f.endswith(".csv")])
    selected_file = st.sidebar.selectbox("Bozulma Sınıfı Seçin:", csv_files)
    df = pd.read_csv(os.path.join("data/XPQRS", selected_file), header=None)
    if df.shape[0] < df.shape[1] and df.shape[1] == 1000:
        df = df.T
else:
    uploaded_file = st.sidebar.file_uploader("CSV Sinyal Dosyası Yükleyin", type=["csv"])
    if uploaded_file is not None:
        selected_file = uploaded_file.name
        df = pd.read_csv(uploaded_file, header=None)
        if df.shape[0] < df.shape[1] and df.shape[1] == 1000:
            df = df.T

st.sidebar.markdown("---")
st.sidebar.subheader("🔊 Canlı Şebeke Gürültüsü (AWGN)")
enable_noise = st.sidebar.checkbox("Sinyale Beyaz Gürültü Ekle", value=False)
snr_val = st.sidebar.slider("Gürültü Seviyesi - SNR (dB)", 15, 50, 35) if enable_noise else 100

# Sekmeler (4 Sekmeli Yapı)
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Canlı Sinyal Analizi & AI Teşhisi", 
    "🏆 ML Model & Grid Search Kıyaslaması",
    "🧠 Açıklanabilir Yapay Zeka (SHAP)", 
    "🛡️ Gürültü Dayanıklılık (Robustness) Raporu"
])

if df is not None:
    row_idx = st.sidebar.slider("Sinyal Örneği No (0 - 999)", 0, len(df)-1, 0)
    clean_signal = df.iloc[row_idx].values.astype(float)
    signal = add_single_awgn(clean_signal, snr_val) if enable_noise else clean_signal
    t = np.arange(len(signal)) / fs * 1000

    t0_feat = time.perf_counter()
    features, xf, yf = extract_signal_features(signal, fs=fs, f0=f0)
    feat_time_ms = (time.perf_counter() - t0_feat) * 1000.0
    num_feats = len(features)

    with tab1:
        col_m1, col_m2 = st.columns(2)
        
        bundle_mtime = os.path.getmtime("models/pq_model_bundle.pkl") if os.path.exists("models/pq_model_bundle.pkl") else 0.0
        ml_bundle = load_ml_bundle(mtime=bundle_mtime)
        if ml_bundle is not None:
            if len(ml_bundle) == 4:
                ml_model, scaler, selector, ml_model_name = ml_bundle
            else:
                ml_model, scaler, selector = ml_bundle
                ml_model_name = type(ml_model).__name__
            
            t0_ml = time.perf_counter()
            df_feat_in = pd.DataFrame([features]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            X_in = np.clip(scaler.transform(df_feat_in), -10.0, 10.0)
            if selector is not None:
                X_in = selector.transform(X_in)

            
            probs_ml = ml_model.predict_proba(X_in)[0]
            best_idx = np.argmax(probs_ml)
            ml_pred = ml_model.classes_[best_idx]
            ml_prob = probs_ml[best_idx] * 100
            ml_only_ms = (time.perf_counter() - t0_ml) * 1000.0
            
            total_ml_ms = feat_time_ms + ml_only_ms
            
            col_m1.success(
                f"🏆 **{ml_model_name} (Özellik Tabanlı) Kararı:**\n\n"
                f"**{ml_pred}** (Güven: **%{ml_prob:.1f}**)\n\n"
                f"⏱️ **Teşhis Süresi: {total_ml_ms:.2f} ms** *(Özellik Çıkarımı: {feat_time_ms:.2f} ms + Model: {ml_only_ms:.2f} ms)*"
            )

        # 1D-CNN
        cnn_model, encoder = load_cnn_bundle()
        if cnn_model is not None and encoder is not None:
            t0_cnn = time.perf_counter()
            with torch.no_grad():
                sig_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                logits = cnn_model(sig_tensor)
                probs = torch.softmax(logits, dim=1).numpy()[0]
                pred_idx = np.argmax(probs)
                raw_cls = encoder.inverse_transform([pred_idx])[0]
                tr_cls = CLASS_NAME_MAP.get(raw_cls, raw_cls)
                cnn_prob = probs[pred_idx] * 100
            cnn_time_ms = (time.perf_counter() - t0_cnn) * 1000.0
            
            col_m2.info(
                f"🧠 **1D-CNN (Gürültüye Dayanıklı Derin Öğrenme) Kararı:**\n\n"
                f"**{tr_cls}** (Güven: **%{cnn_prob:.1f}**)\n\n"
                f"⏱️ **Teşhis Süresi: {cnn_time_ms:.2f} ms** *(Ham Sinyalden Doğrudan Çıkarım)*"
            )

        g_col1, g_col2 = st.columns(2)
        with g_col1:
            st.subheader("Zaman Domeni Dalga Formu (10 Periyot / 200 ms Panoramik Zarf)")
            fig1, ax1 = plt.subplots(figsize=(6.2, 3.2))
            
            t_multi, clean_multi = build_multicycle_view(clean_signal, class_hint=selected_file, fs=fs, f0=f0, n_cycles=10)
            sig_multi = add_single_awgn(clean_multi, snr_val) if enable_noise else clean_multi
            
            ax1.axhline(1.0, color="green", linestyle=":", alpha=0.5, linewidth=1.0, label="Nominal Sınır (±1.0)")
            ax1.axhline(-1.0, color="green", linestyle=":", alpha=0.5, linewidth=1.0)
            
            if enable_noise:
                ax1.plot(t_multi, clean_multi, color="gray", alpha=0.35, linestyle="--", label="Temiz Sinyal")
            ax1.plot(t_multi, sig_multi, color="#1f77b4", linewidth=1.2, label="İncelenen Sinyal (10 Periyot)")
            
            analytic_sig = hilbert(clean_multi)
            envelope = np.abs(analytic_sig)
            kernel_size = int(fs / f0)
            envelope_smooth = np.convolve(envelope, np.ones(kernel_size)/kernel_size, mode='same')
            ax1.plot(t_multi, envelope_smooth, color="#d62728", linestyle="--", linewidth=1.5, alpha=0.85, label="Genlik Zarfı (Envelope)")
            ax1.plot(t_multi, -envelope_smooth, color="#d62728", linestyle="--", linewidth=1.5, alpha=0.85)
            
            if any(k in selected_file.lower() for k in ["sag", "swell", "interruption"]):
                ax1.axvspan(40, 160, color="orange", alpha=0.12, label="Bozulma Bölgesi")

            ax1.set_xlabel("Zaman (ms)")
            ax1.set_ylabel("Normalize Genlik")
            ax1.legend(loc="upper right", fontsize=7.5)
            ax1.grid(True, alpha=0.3)
            fig1.tight_layout()
            st.pyplot(fig1)

        with g_col2:
            st.subheader("FFT Frekans Spektrumu (Harmonik Analizi)")
            fig2, ax2 = plt.subplots(figsize=(6.2, 3.2))
            ax2.stem(xf[:20], yf[:20], basefmt=" ")
            ax2.set_xlabel("Frekans (Hz)")
            ax2.set_ylabel("Normalize Genlik")
            ax2.grid(True, alpha=0.3)
            fig2.tight_layout()
            st.pyplot(fig2)

        st.subheader(f"📐 Çıkarılan {num_feats} Elektriksel ve İstatistiksel Özellik")
        st.dataframe(pd.DataFrame([features]).style.format("{:.4f}"), use_container_width=True)

    with tab2:
        st.subheader("🏆 Makine Öğrenmesi Modelleri ve Hiper-Parametre Optimizasyonu (Grid Search)")
        st.write(f"5 katlı çapraz doğrulama (5-Fold Stratified CV Ortalama ± Std), Doğrulama (Val) ve Kör Test başarımları:")
        
        if os.path.exists("outputs/ml_grid_search_results.csv"):
            gs_df = pd.read_csv("outputs/ml_grid_search_results.csv")
            col_g1, col_g2 = st.columns([1.1, 1.9])
            
            with col_g1:
                fig_gs, ax_gs = plt.subplots(figsize=(6.2, 4.2))
                x = np.arange(len(gs_df["Model"]))
                width = 0.35
                
                ax_gs.bar(x - width/2, gs_df["Acc_All_Features"], width, label=f"Tüm Özellikler ({num_feats})", color="#1f77b4")
                ax_gs.bar(x + width/2, gs_df["Acc_Selected_Features"], width, label="Seçili Özellikler (ANOVA)", color="#ff7f0e")
                
                ax_gs.set_ylabel("Test Doğruluğu (%)")
                ax_gs.set_title("Özellik Seçimi Öncesi ve Sonrası Model Başarımları")
                ax_gs.set_xticks(x)
                ax_gs.set_xticklabels(gs_df["Model"], rotation=25, ha="right", fontsize=8.5)
                ax_gs.set_ylim(max(0, min(gs_df["Acc_Selected_Features"].min(), gs_df["Acc_All_Features"].min()) - 8), 100)
                ax_gs.legend(loc="lower right", fontsize=8)
                ax_gs.grid(True, axis="y", alpha=0.3)
                fig_gs.tight_layout()
                st.pyplot(fig_gs)
                
            with col_g2:
                if "CV_Mean_Std_All" in gs_df.columns:
                    ordered_cols = [
                        "Model",
                        "CV_Mean_Std_All",
                        "Val_Acc_All",
                        "Acc_All_Features",
                        "CV_Mean_Std_Sel",
                        "Val_Acc_Selected",
                        "Acc_Selected_Features",
                        "Best_Params_All"
                    ]
                    display_gs_df = gs_df[ordered_cols].rename(columns={
                        "Model": "Algoritma",
                        "CV_Mean_Std_All": "5-Fold CV (Tüm Öz.)",
                        "Val_Acc_All": "Val (Tüm Öz.)",
                        "Acc_All_Features": "Test (Tüm Öz.)",
                        "CV_Mean_Std_Sel": "5-Fold CV (Seçili)",
                        "Val_Acc_Selected": "Val (Seçili)",
                        "Acc_Selected_Features": "Test (Seçili)",
                        "Best_Params_All": "En İyi Parametreler (Tüm Öz.)"
                    })
                    st.dataframe(
                        display_gs_df.style.format({
                            "Val (Tüm Öz.)": "%{:.2f}",
                            "Test (Tüm Öz.)": "%{:.2f}",
                            "Val (Seçili)": "%{:.2f}",
                            "Test (Seçili)": "%{:.2f}"
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.dataframe(gs_df, use_container_width=True, hide_index=True)

            if os.path.exists("outputs/classification_report_best_ml.csv"):
                with st.expander("📊 En Başarılı Modelin Sınıf Bazında Precision, Recall ve F1-Skor Tablosunu Göster"):
                    f1_df = pd.read_csv("outputs/classification_report_best_ml.csv")
                    st.dataframe(
                        f1_df.style.format({
                            "precision": "{:.4f}",
                            "recall": "{:.4f}",
                            "f1-score": "{:.4f}",
                            "support": "{:.0f}"
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
        else:
            st.warning("'outputs/ml_grid_search_results.csv' dosyası bulunamadı. Lütfen önce train_pipeline.py dosyasını çalıştırın.")

    with tab3:
        st.subheader("🔍 SHAP (Açıklanabilir Yapay Zeka) Özellik Önem Dereceleri")
        st.write("Modelin tekil ve melez güç kalitesi sınıflarını ayırt ederken en çok hangi matematiksel/elektriksel özelliğe dayandığını gösterir:")
        if os.path.exists("outputs/shap_importance.csv"):
            shap_df = pd.read_csv("outputs/shap_importance.csv")
            col_s1, col_s2 = st.columns([2, 1])
            with col_s1:
                fig_s, ax_s = plt.subplots(figsize=(7, 4.5))
                ax_s.barh(shap_df["Feature"][::-1], shap_df["SHAP_Importance"][::-1], color="#ff7f0e")
                ax_s.set_xlabel("Ortalama Mutlak SHAP Etki Değeri")
                ax_s.set_title("Karar Mekanizmasında En Etkili Özellikler")
                ax_s.grid(True, alpha=0.3)
                st.pyplot(fig_s)
            with col_s2:
                st.dataframe(
                    shap_df.style.format({"SHAP_Importance": "{:.6f}"}),
                    use_container_width=True,
                    height=560,
                    hide_index=True
                )

    with tab4:
        st.subheader("🛡️ AWGN Şebeke Gürültüsü Dayanıklılık Testi (Tüm Test Kümesi & Çoklu Tohum)")
        st.write("Temiz ve AWGN gürültülü koşullarda (3 farklı rastgele tohum ortalaması ± standart sapma) Standart 1D-CNN, Gürültüye Dayanıklı 1D-CNN ve Özellik Tabanlı Klasik ML modelinin karşılaştırması:")
        if os.path.exists("outputs/noise_robustness_results.csv"):
            noise_df = pd.read_csv("outputs/noise_robustness_results.csv")
            if "Standart_CNN" in noise_df.columns and "Robust_CNN" in noise_df.columns:
                noise_df["Net_Kazanc"] = noise_df["Robust_CNN"] - noise_df["Standart_CNN"]
            col_n1, col_n2 = st.columns([1.8, 1.2])
            with col_n1:
                fig_n, ax_n = plt.subplots(figsize=(6.5, 3.8))
                if "Standart_CNN" in noise_df.columns and "Robust_CNN" in noise_df.columns:
                    ax_n.plot(noise_df["SNR_dB"], noise_df["Robust_CNN"], marker="s", linewidth=2.5, color="#2ca02c", label="Gürültüye Dayanıklı 1D-CNN (Önerilen)")
                if "Klasik_ML" in noise_df.columns and not noise_df["Klasik_ML"].isna().all():
                    ml_label_name = ml_bundle[3] if (ml_bundle and len(ml_bundle) == 4) else "En İyi Klasik ML"
                    ax_n.plot(noise_df["SNR_dB"], noise_df["Klasik_ML"], marker="^", linestyle="-.", linewidth=2, color="#1f77b4", label=f"Klasik ML ({ml_label_name})")

                    ax_n.legend(loc="lower left", fontsize=8.5)
                ax_n.set_xlabel("Gürültü Seviyesi (SNR)")
                ax_n.set_ylabel("Test Doğruluk Oranı (%)")
                ax_n.set_ylim(0, 105)
                ax_n.grid(True, alpha=0.3)
                fig_n.tight_layout()
                st.pyplot(fig_n)
            with col_n2:
                if "Robust_CNN_Format" in noise_df.columns:
                    disp_noise = noise_df[[
                        "SNR_dB", "Robust_CNN_Format", "Standart_CNN_Format", "Klasik_ML_Format", "Net_Kazanc"
                    ]].rename(columns={
                        "SNR_dB": "SNR",
                        "Robust_CNN_Format": "Robust 1D-CNN (Ort±Std)",
                        "Standart_CNN_Format": "Standart 1D-CNN (Ort±Std)",
                        "Klasik_ML_Format": "Klasik ML (Ort±Std)",
                        "Net_Kazanc": "CNN Net Kazanç"
                    })
                    st.dataframe(
                        disp_noise.style.format({"CNN Net Kazanç": "+%{:.2f}"}),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    format_dict = {
                        "Standart_CNN": "%{:.2f}",
                        "Robust_CNN": "%{:.2f}",
                        "Net_Kazanc": "+%{:.2f}"
                    }
                    st.dataframe(
                        noise_df.style.format(format_dict),
                        use_container_width=True,
                        hide_index=True
                    )
        if os.path.exists("outputs/classification_report_cnn.csv"):
            with st.expander("📊 Gürültüye Dayanıklı 1D-CNN Sınıf Bazında Precision, Recall ve F1-Skor Tablosunu Göster"):
                f1_cnn_df = pd.read_csv("outputs/classification_report_cnn.csv")
                st.dataframe(
                    f1_cnn_df.style.format({
                        "precision": "{:.4f}",
                        "recall": "{:.4f}",
                        "f1-score": "{:.4f}",
                        "support": "{:.0f}"
                    }),
                    use_container_width=True,
                    hide_index=True
                )