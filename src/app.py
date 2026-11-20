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
from pq_analyzer import (
    extract_signal_features,
    synthesize_harmonic_signal,
    decompose_signal_harmonics,
)
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


def get_harmonic_preset_df(preset_name, base_f0):
    """Harmonik Sentezleyici sekmesi için hazır şebeke senaryoları."""
    if preset_name == "6-Darbeli Doğrultucu (5., 7., 11., 13.)":
        rows = [
            {"Aktif": True, "Frekans (Hz)": 5.0 * base_f0, "Genlik": 0.20, "Faz (°)": 0.0},
            {"Aktif": True, "Frekans (Hz)": 7.0 * base_f0, "Genlik": 0.14, "Faz (°)": 0.0},
            {"Aktif": True, "Frekans (Hz)": 11.0 * base_f0, "Genlik": 0.09, "Faz (°)": 0.0},
            {"Aktif": True, "Frekans (Hz)": 13.0 * base_f0, "Genlik": 0.07, "Faz (°)": 0.0},
        ]
    elif preset_name == "Tek Fazlı Doğrusal Olmayan Yük / SMPS (3., 5., 7., 9.)":
        rows = [
            {"Aktif": True, "Frekans (Hz)": 3.0 * base_f0, "Genlik": 0.30, "Faz (°)": 180.0},
            {"Aktif": True, "Frekans (Hz)": 5.0 * base_f0, "Genlik": 0.18, "Faz (°)": 0.0},
            {"Aktif": True, "Frekans (Hz)": 7.0 * base_f0, "Genlik": 0.10, "Faz (°)": 180.0},
            {"Aktif": True, "Frekans (Hz)": 9.0 * base_f0, "Genlik": 0.05, "Faz (°)": 0.0},
        ]
    elif preset_name == "Kare Dalga Yaklaşımı (Tek Harmonikler 1/h)":
        rows = [
            {"Aktif": True, "Frekans (Hz)": float(h * base_f0), "Genlik": round(1.0 / h, 4), "Faz (°)": 0.0}
            for h in [3, 5, 7, 9, 11]
        ]
    elif preset_name == "Ara Harmonikli (Interharmonic) Senaryo":
        rows = [
            {"Aktif": True, "Frekans (Hz)": 3.0 * base_f0, "Genlik": 0.22, "Faz (°)": 0.0},
            {"Aktif": True, "Frekans (Hz)": 5.0 * base_f0, "Genlik": 0.15, "Faz (°)": 30.0},
            {"Aktif": True, "Frekans (Hz)": 3.5 * base_f0, "Genlik": 0.12, "Faz (°)": 0.0},
            {"Aktif": True, "Frekans (Hz)": 6.25 * base_f0, "Genlik": 0.08, "Faz (°)": -45.0},
        ]
    else:
        rows = [
            {"Aktif": True, "Frekans (Hz)": 3.0 * base_f0, "Genlik": 0.25, "Faz (°)": 0.0},
            {"Aktif": True, "Frekans (Hz)": 5.0 * base_f0, "Genlik": 0.15, "Faz (°)": 30.0},
            {"Aktif": True, "Frekans (Hz)": 7.0 * base_f0, "Genlik": 0.10, "Faz (°)": -20.0},
        ]
    return pd.DataFrame(rows)


def render_isolated_components(
    t_ms,
    composite_sig,
    components,
    fft_max_freq,
    fft_scale,
    share_y_axis,
    view_mode,
    composite_label="Bileşke (Toplam)",
):
    """Ayrık harmonik bileşenleri teker teker (Zaman + Bireysel FFT) veya üst üste çizer."""
    color_palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#393b79"
    ]
    global_max_amp = max([float(np.max(np.abs(c["signal"]))) for c in components] + [0.1]) * 1.15

    if view_mode == "Her Bileşen İçin Ayrı Zaman + FFT Satırı":
        for idx_c, comp in enumerate(components):
            color = "#555555" if comp.get("kind") == "residual" else color_palette[idx_c % len(color_palette)]
            phase_txt = "-" if pd.isna(comp["phase_deg"]) else f"{comp['phase_deg']:.1f}°"
            st.markdown(
                f"**{idx_c + 1}. {comp['name']}** — "
                f"Genlik: `{comp['amp']:.4f}` | RMS: `{comp['rms']:.4f}` | "
                f"Faz: `{phase_txt}` | Temele Oran (HD): `%{comp['hd_percent']:.2f}` | "
                f"Enerji Payı: `%{comp['power_share_pct']:.2f}`"
            )
            c_left, c_right = st.columns(2)

            with c_left:
                fig_t, ax_t = plt.subplots(figsize=(6.2, 2.35))
                ax_t.plot(t_ms, comp["signal"], color=color, linewidth=1.4)
                ax_t.axhline(0.0, color="black", linewidth=0.6, alpha=0.4)
                if share_y_axis:
                    ax_t.set_ylim(-global_max_amp, global_max_amp)
                ax_t.set_xlabel("Zaman (ms)", fontsize=8.5)
                ax_t.set_ylabel("Genlik", fontsize=8.5)
                ax_t.set_title(f"Zaman Domeni — {comp['name']}", fontsize=9.5)
                ax_t.grid(True, alpha=0.3)
                fig_t.tight_layout()
                st.pyplot(fig_t)
                plt.close(fig_t)

            with c_right:
                fig_f, ax_f = plt.subplots(figsize=(6.2, 2.35))
                c_mask = comp["xf"] <= fft_max_freq
                cxf = comp["xf"][c_mask]
                cyf = comp["yf"][c_mask]
                if fft_scale == "dB Ölçek":
                    cyf_db = 20.0 * np.log10(np.maximum(cyf, 1e-6))
                    ax_f.plot(cxf, cyf_db, color=color, linewidth=1.3)
                    ax_f.set_ylabel("Genlik (dB)", fontsize=8.5)
                else:
                    ax_f.stem(cxf, cyf, linefmt=color, markerfmt="o", basefmt=" ")
                    if share_y_axis:
                        ax_f.set_ylim(0, global_max_amp)
                    ax_f.set_ylabel("Genlik", fontsize=8.5)
                ax_f.set_xlim(0, fft_max_freq)
                ax_f.set_xlabel("Frekans (Hz)", fontsize=8.5)
                fft_title = "Bireysel FFT — Artık Spektrum" if comp.get("kind") == "residual" else f"Bireysel FFT — {comp['freq']:.1f} Hz"
                ax_f.set_title(fft_title, fontsize=9.5)
                ax_f.grid(True, alpha=0.3)
                fig_f.tight_layout()
                st.pyplot(fig_f)
                plt.close(fig_f)
    else:
        fig_ov, ax_ov = plt.subplots(figsize=(12, 4.2))
        for idx_c, comp in enumerate(components):
            color = "#555555" if comp.get("kind") == "residual" else color_palette[idx_c % len(color_palette)]
            ax_ov.plot(t_ms, comp["signal"], color=color, linewidth=1.35, label=comp["name"])
        ax_ov.plot(t_ms, composite_sig, color="black", linewidth=2.0, linestyle="--", alpha=0.8, label=composite_label)
        ax_ov.set_xlabel("Zaman (ms)")
        ax_ov.set_ylabel("Genlik")
        ax_ov.set_title("Ayrık Bileşenler ve Toplam Sinyal Karşılaştırması")
        ax_ov.legend(loc="upper right", fontsize=8, ncol=2)
        ax_ov.grid(True, alpha=0.3)
        fig_ov.tight_layout()
        st.pyplot(fig_ov)
        plt.close(fig_ov)


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

# Sekmeler
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Canlı Sinyal Analizi & AI Teşhisi",
    "🏆 ML Model & Grid Search Kıyaslaması",
    "🧠 Açıklanabilir Yapay Zeka (SHAP)",
    "🛡️ Gürültü Dayanıklılık (Robustness) Raporu",
    "🎛️ Harmonik Sentezleyici & Ayrık Bileşen Analizi"
])

bundle_mtime = os.path.getmtime("models/pq_model_bundle.pkl") if os.path.exists("models/pq_model_bundle.pkl") else 0.0
ml_bundle = load_ml_bundle(mtime=bundle_mtime)
cnn_model, encoder = load_cnn_bundle()

clean_signal = None
signal = None
row_idx = 0

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
        st.write("5 katlı çapraz doğrulama (5-Fold Stratified CV Ortalama ± Std), Doğrulama (Val) ve Kör Test başarımları:")

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

# -------------------------------------------------------------------------
# SEKME 5: HARMONİK SENTEZLEYİCİ & AYRIK BİLEŞEN ANALİZİ
# (Hem parametrik sentez hem de XPQRS / CSV sinyal ayrıştırma destekli)
# -------------------------------------------------------------------------
with tab5:
    st.subheader("🎛️ Harmonik Sentezleyici, FFT ve Ayrık Bileşen Analizi")

    harm_mode = st.radio(
        "Analiz Modu Seçin:",
        [
            "🧪 Parametrik Harmonik Sentezleyici (Manuel Giriş)",
            "📂 XPQRS / Yüklenen CSV Sinyalini Harmoniklerine Ayrıştır",
        ],
        horizontal=True,
        key="harm_tab_mode",
    )

    # PARAMETRİK HARMONİK SENTEZLEYİCİ

    if harm_mode == "🧪 Parametrik Harmonik Sentezleyici (Manuel Giriş)":
        st.caption(
            "Temel bileşen ve dilediğiniz harmonik/ara-harmonik bileşenlerin frekans, genlik ve faz parametrelerini girerek "
            "bileşke dalga formunu, FFT frekans spektrumunu ve her bir bileşenin ayrık zaman/FFT grafiklerini inceleyin."
        )

        # 1. Temel Sinyal ve Örnekleme Parametreleri
        with st.expander("⚙️ Temel Dalga ve Analiz Parametreleri", expanded=True):
            p_col1, p_col2, p_col3, p_col4 = st.columns(4)
            h_f0 = p_col1.number_input("Temel Frekans f₀ (Hz)", min_value=1.0, max_value=1000.0, value=float(f0), step=5.0, key="harm_f0")
            h_fs = p_col2.number_input("Örnekleme Frekansı fs (Hz)", min_value=500.0, max_value=50000.0, value=float(fs), step=500.0, key="harm_fs")
            h_n_cycles = p_col3.slider("Periyot Sayısı", min_value=1, max_value=15, value=3, key="harm_cycles")
            h_window_label = p_col4.selectbox(
                "FFT Pencereleme",
                ["Dikdörtgen (Penceresiz)", "Hann Penceresi"],
                index=0,
                key="harm_window"
            )
            h_window = "hann" if "Hann" in h_window_label else "rect"

            p_col5, p_col6, p_col7, p_col8 = st.columns(4)
            h_fund_amp = p_col5.number_input("Temel Bileşen Genliği (A₁)", min_value=0.01, max_value=1000.0, value=1.0, step=0.1, key="harm_a1")
            h_fund_phase = p_col6.number_input("Temel Faz Açısı φ₁ (°)", min_value=-360.0, max_value=360.0, value=0.0, step=5.0, key="harm_p1")
            h_dc = p_col7.number_input("DC Ofset (A₀)", min_value=-500.0, max_value=500.0, value=0.0, step=0.05, key="harm_dc")
            h_add_noise = p_col8.checkbox("Sentez Sinyale AWGN Ekle", value=False, key="harm_noise_chk")
            h_snr = p_col8.slider("SNR (dB)", 10, 50, 30, key="harm_snr") if h_add_noise else None

        # 2. Harmonik Bileşen Tablosu & Hazır Senaryolar
        preset_options = [
            "Varsayılan (3., 5., 7. Harmonik)",
            "6-Darbeli Doğrultucu (5., 7., 11., 13.)",
            "Tek Fazlı Doğrusal Olmayan Yük / SMPS (3., 5., 7., 9.)",
            "Kare Dalga Yaklaşımı (Tek Harmonikler 1/h)",
            "Ara Harmonikli (Interharmonic) Senaryo",
        ]
        pre_col1, pre_col2 = st.columns([2, 1])
        selected_preset = pre_col1.selectbox("Hazır Harmonik Senaryosu Yükle:", preset_options, key="harm_preset_select")
        load_preset_btn = pre_col2.button("Senaryoyu Tabloya Uygula", use_container_width=True, key="harm_apply_preset")

        if "harm_table_df" not in st.session_state or load_preset_btn:
            st.session_state["harm_table_df"] = get_harmonic_preset_df(selected_preset, h_f0)

        with st.container():
            st.markdown("**Harmonik / Frekans Bileşenleri Tablosu** *(Satır ekleyip silebilir, Frekans, Genlik ve Faz değerlerini doğrudan düzenleyebilirsiniz)*")
            edited_harm_df = st.data_editor(
                st.session_state["harm_table_df"],
                num_rows="dynamic",
                use_container_width=True,
                key="harm_data_editor",
                column_config={
                    "Aktif": st.column_config.CheckboxColumn("Aktif", default=True),
                    "Frekans (Hz)": st.column_config.NumberColumn("Frekans (Hz)", min_value=0.1, max_value=25000.0, step=10.0, format="%.2f", required=True),
                    "Genlik": st.column_config.NumberColumn("Genlik (Tepe)", min_value=0.0, max_value=1000.0, step=0.05, format="%.4f", required=True),
                    "Faz (°)": st.column_config.NumberColumn("Faz (°)", min_value=-360.0, max_value=360.0, step=5.0, format="%.1f", required=True),
                },
            )

        harmonics_list = []
        for _, row in edited_harm_df.iterrows():
            if pd.isna(row.get("Frekans (Hz)")) or pd.isna(row.get("Genlik")):
                continue
            harmonics_list.append({
                "active": True if pd.isna(row.get("Aktif")) else bool(row.get("Aktif")),
                "freq": float(row["Frekans (Hz)"]),
                "amp": float(row["Genlik"]),
                "phase_deg": 0.0 if pd.isna(row.get("Faz (°)")) else float(row["Faz (°)"]),
            })

        synth = synthesize_harmonic_signal(
            f0=h_f0,
            fs=h_fs,
            n_cycles=h_n_cycles,
            fundamental_amp=h_fund_amp,
            fundamental_phase_deg=h_fund_phase,
            harmonics=harmonics_list,
            dc_offset=h_dc,
            snr_db=h_snr,
            window=h_window,
        )
        h_metrics = synth["metrics"]
        h_components = synth["components"]

        if h_metrics["Nyquist_Warnings"]:
            warn_freqs = ", ".join(f"{f:.1f} Hz" for f in h_metrics["Nyquist_Warnings"])
            st.warning(
                f"⚠️ Nyquist Uyarısı: {warn_freqs} bileşen(ler)i örnekleme frekansının yarısına "
                f"({h_fs / 2.0:.1f} Hz) eşit veya büyük. Örtüşme (aliasing) oluşmaması için fs değerini artırın."
            )

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Teorik THD (%)", f"%{h_metrics['THD_Theoretical_Percent']:.2f}")
        k2.metric("FFT THD (%)", f"%{h_metrics['THD_FFT_Percent']:.2f}")
        k3.metric("Bileşke RMS", f"{h_metrics['Composite_RMS']:.4f}")
        k4.metric("Tepe / Crest Factor", f"{h_metrics['Composite_Peak']:.3f} / {h_metrics['Crest_Factor']:.3f}")
        k5.metric("Aktif Bileşen / Δf", f"{h_metrics['Num_Components']} adet | {h_metrics['Freq_Resolution_Hz']:.2f} Hz")

        max_comp_freq = max([c["freq"] for c in h_components] + [h_f0 * 10.0])
        default_fft_max = min(h_fs / 2.0, max(h_f0 * 15.0, max_comp_freq * 1.25))
        vc1, vc2, vc3 = st.columns([2, 1, 1])
        fft_max_freq = vc1.slider(
            "FFT Maksimum Gösterim Frekansı (Hz)",
            min_value=float(max(h_f0 * 2.0, 100.0)),
            max_value=float(h_fs / 2.0),
            value=float(default_fft_max),
            step=float(max(10.0, h_f0)),
            key="harm_fft_max"
        )
        fft_scale = vc2.radio("FFT Y-Ekseni Ölçeği", ["Lineer Genlik", "dB Ölçek"], horizontal=True, key="harm_fft_scale")
        share_y_axis = vc3.checkbox("Ayrık Grafiklerde Ortak Y-Ekseni", value=True, key="harm_share_y")

        fft_mask = synth["xf"] <= fft_max_freq

        st.markdown("### 📊 Bileşke (Toplam) Sinyal ve FFT Spektrumu")
        comp_col1, comp_col2 = st.columns(2)

        with comp_col1:
            fig_comp, ax_comp = plt.subplots(figsize=(6.4, 3.4))
            ax_comp.plot(
                synth["t_ms"], synth["fundamental"],
                color="gray", linestyle="--", linewidth=1.1, alpha=0.65,
                label=f"Temel Sinüs ({h_f0:.1f} Hz)"
            )
            ax_comp.plot(
                synth["t_ms"], synth["composite"],
                color="#1f77b4", linewidth=1.6,
                label="Bileşke Sinyal (Toplam)"
            )
            ax_comp.set_title("Bileşke Dalga Formu (Zaman Domeni)")
            ax_comp.set_xlabel("Zaman (ms)")
            ax_comp.set_ylabel("Genlik")
            ax_comp.legend(loc="upper right", fontsize=8)
            ax_comp.grid(True, alpha=0.3)
            fig_comp.tight_layout()
            st.pyplot(fig_comp)
            plt.close(fig_comp)

        with comp_col2:
            fig_cfft, ax_cfft = plt.subplots(figsize=(6.4, 3.4))
            xf_plot = synth["xf"][fft_mask]
            yf_plot = synth["yf"][fft_mask]
            if fft_scale == "dB Ölçek":
                yf_db = 20.0 * np.log10(np.maximum(yf_plot, 1e-6))
                ax_cfft.plot(xf_plot, yf_db, color="#d62728", linewidth=1.3)
                ax_cfft.set_ylabel("Genlik (dB)")
                ax_cfft.set_ylim(max(-100.0, float(np.min(yf_db)) - 5.0), float(np.max(yf_db)) + 8.0)
            else:
                ax_cfft.stem(xf_plot, yf_plot, linefmt="#d62728", markerfmt="ro", basefmt=" ")
                ax_cfft.set_ylabel("Genlik")
                peak_thresh = max(0.02 * float(np.max(yf_plot)), 1e-4)
                for c in h_components:
                    if c["freq"] <= fft_max_freq and abs(c["amp"]) >= peak_thresh:
                        idx_p = int(np.argmin(np.abs(xf_plot - c["freq"])))
                        ax_cfft.annotate(
                            f"{c['freq']:.0f}Hz\n({yf_plot[idx_p]:.2f})",
                            xy=(xf_plot[idx_p], yf_plot[idx_p]),
                            xytext=(0, 6),
                            textcoords="offset points",
                            ha="center",
                            fontsize=7.2,
                            color="#222222",
                        )
            ax_cfft.set_title("Bileşke FFT Frekans Spektrumu")
            ax_cfft.set_xlabel("Frekans (Hz)")
            ax_cfft.set_xlim(0, fft_max_freq)
            ax_cfft.grid(True, alpha=0.3)
            fig_cfft.tight_layout()
            st.pyplot(fig_cfft)
            plt.close(fig_cfft)

        st.markdown("### 🧩 Bileşenlerin Teker Teker Ayrık Görünümü (Zaman Domeni + Bireysel FFT)")
        view_mode = st.radio(
            "Görünüm Biçimi:",
            ["Her Bileşen İçin Ayrı Zaman + FFT Satırı", "Tüm Ayrık Bileşenler Üst Üste (Karşılaştırmalı)"],
            horizontal=True,
            key="harm_view_mode"
        )
        render_isolated_components(
            t_ms=synth["t_ms"],
            composite_sig=synth["composite"],
            components=h_components,
            fft_max_freq=fft_max_freq,
            fft_scale=fft_scale,
            share_y_axis=share_y_axis,
            view_mode=view_mode,
            composite_label="Bileşke (Toplam)",
        )

        st.markdown("### 📋 Ayrık Bileşenler Sayısal Özeti")
        summary_rows = [
            {
                "Bileşen": c["name"],
                "Frekans (Hz)": c["freq"],
                "Harmonik Derecesi (h)": c["order"],
                "Genlik (Tepe)": c["amp"],
                "RMS": c["rms"],
                "Faz (°)": c["phase_deg"],
                "Temele Oran - HD (%)": c["hd_percent"],
                "Güç / Enerji Payı (%)": c["power_share_pct"],
            }
            for c in h_components
        ]
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(
            summary_df.style.format({
                "Frekans (Hz)": "{:.2f}",
                "Harmonik Derecesi (h)": "{:.2f}",
                "Genlik (Tepe)": "{:.4f}",
                "RMS": "{:.4f}",
                "Faz (°)": "{:.1f}",
                "Temele Oran - HD (%)": "%{:.2f}",
                "Güç / Enerji Payı (%)": "%{:.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        act_col1, act_col2 = st.columns([1.4, 1])
        with act_col1:
            if st.button("🤖 Sentezlenen Sinyali AI Modelleriyle (ML & 1D-CNN) Teşhis Et", use_container_width=True, key="harm_run_ai"):
                one_cycle_len = max(16, int(round(h_fs / h_f0)))
                ai_sig = synth["composite"][:one_cycle_len] if len(synth["composite"]) >= one_cycle_len else synth["composite"]
                h_feats, _, _ = extract_signal_features(ai_sig, fs=h_fs, f0=h_f0)

                r_col1, r_col2 = st.columns(2)
                if ml_bundle is not None:
                    if len(ml_bundle) == 4:
                        ml_m, ml_sc, ml_sel, ml_nm = ml_bundle
                    else:
                        ml_m, ml_sc, ml_sel = ml_bundle
                        ml_nm = type(ml_m).__name__
                    df_h_in = pd.DataFrame([h_feats]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                    X_h = np.clip(ml_sc.transform(df_h_in), -10.0, 10.0)
                    if ml_sel is not None:
                        X_h = ml_sel.transform(X_h)
                    p_ml = ml_m.predict_proba(X_h)[0]
                    b_i = int(np.argmax(p_ml))
                    r_col1.success(f"🏆 **{ml_nm} Tahmini:** **{ml_m.classes_[b_i]}** (Güven: **%{p_ml[b_i]*100:.1f}**)")
                else:
                    r_col1.info("Klasik ML modeli (`models/pq_model_bundle.pkl`) bulunamadı.")

                if cnn_model is not None and encoder is not None:
                    with torch.no_grad():
                        sig_t = torch.tensor(ai_sig, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                        p_cnn = torch.softmax(cnn_model(sig_t), dim=1).numpy()[0]
                        c_i = int(np.argmax(p_cnn))
                        raw_c = encoder.inverse_transform([c_i])[0]
                        tr_c = CLASS_NAME_MAP.get(raw_c, raw_c)
                    r_col2.info(f"🧠 **1D-CNN Tahmini:** **{tr_c}** (Güven: **%{p_cnn[c_i]*100:.1f}**)")
                else:
                    r_col2.info("1D-CNN modeli (`models/pq_cnn_model.pt`) bulunamadı.")

        with act_col2:
            csv_bytes = pd.DataFrame([synth["composite"]]).to_csv(index=False, header=False).encode("utf-8")
            st.download_button(
                label="⬇️ Sentezlenen Sinyali CSV Olarak İndir",
                data=csv_bytes,
                file_name=f"sentez_harmonik_{int(h_f0)}Hz_THD{h_metrics['THD_Theoretical_Percent']:.1f}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    else:
        st.caption(
            "Sol menüde seçili XPQRS sınıfını veya dışarıdan yüklediğiniz CSV sinyalini FFT tabanlı spektral ayrıştırma "
            "(rfft / irfft) ile temel bileşen, tekil harmonikler ve artık (gürültü/diğer olaylar) bileşenlerine ayırarak inceleyin."
        )

        if signal is None:
            st.info(
                "Ayrıştırılacak bir sinyal bulunamadı. Lütfen sol menüden XPQRS veri setinden bir sınıf seçin "
                "veya 'Dışarıdan CSV Yükle' seçeneğiyle bir CSV dosyası yükleyin."
            )
        else:
            noise_badge = f" + AWGN ({snr_val} dB)" if enable_noise else " (Temiz)"
            st.info(
                f"📌 **İncelenen Sinyal:** `{selected_file}` | **Örnek No:** `{row_idx}` | "
                f"**Örnek Sayısı:** `{len(signal)}` | **fs / f₀:** `{fs} Hz / {f0} Hz`{noise_badge}"
            )

            with st.expander("⚙️ Ayrıştırma Ayarları", expanded=True):
                d_col1, d_col2, d_col3, d_col4 = st.columns(4)
                max_nyq_h = max(3, min(40, int((fs / 2.0) // f0)))
                d_max_order = d_col1.slider(
                    "Maksimum Harmonik Derecesi (h)",
                    min_value=3,
                    max_value=max_nyq_h,
                    value=min(15, max_nyq_h),
                    key="dec_max_h",
                )
                d_min_hd = d_col2.slider(
                    "Minimum Gösterim Eşiği — HD (%)",
                    min_value=0.0,
                    max_value=15.0,
                    value=1.0,
                    step=0.5,
                    key="dec_min_hd",
                )
                d_win_label = d_col3.selectbox(
                    "FFT Görünüm Penceresi",
                    ["Dikdörtgen (Penceresiz)", "Hann Penceresi"],
                    index=0,
                    key="dec_window",
                )
                d_window = "hann" if "Hann" in d_win_label else "rect"
                d_odd_only = d_col4.checkbox("Sadece Tek Harmonikleri Göster", value=False, key="dec_odd_only")
                d_include_res = d_col4.checkbox("Artık (Residual) Bileşeni Göster", value=True, key="dec_inc_res")

            decomp = decompose_signal_harmonics(
                signal=signal,
                fs=fs,
                f0=f0,
                max_order=d_max_order,
                min_hd_percent=d_min_hd,
                odd_only=d_odd_only,
                include_dc=True,
                include_residual=d_include_res,
                window=d_window,
            )
            dm = decomp["metrics"]
            d_components = decomp["components"]

            dk1, dk2, dk3, dk4, dk5 = st.columns(5)
            dk1.metric("FFT THD (%)", f"%{dm['THD_FFT_Percent']:.2f}")
            dk2.metric("Orijinal RMS", f"{dm['Original_RMS']:.4f}")
            dk3.metric("Temel Bileşen (A₁ / RMS)", f"{dm['Fundamental_Amp']:.3f} / {dm['Fundamental_RMS']:.3f}")
            dk4.metric("Eşik Üstü Harmonik", f"{dm['Detected_Harmonics']} adet")
            dk5.metric("Artık (Residual) Enerji", f"%{dm['Residual_Energy_Pct']:.2f}")

            d_default_max_f = min(float(fs) / 2.0, float((d_max_order + 2) * f0))
            dvc1, dvc2, dvc3 = st.columns([2, 1, 1])
            d_fft_max = dvc1.slider(
                "FFT Maksimum Gösterim Frekansı (Hz)",
                min_value=float(max(f0 * 2.0, 100.0)),
                max_value=float(fs / 2.0),
                value=float(d_default_max_f),
                step=float(max(10.0, f0)),
                key="dec_fft_max",
            )
            d_fft_scale = dvc2.radio("FFT Y-Ekseni Ölçeği", ["Lineer Genlik", "dB Ölçek"], horizontal=True, key="dec_fft_scale")
            d_share_y = dvc3.checkbox("Ayrık Grafiklerde Ortak Y-Ekseni", value=True, key="dec_share_y")

            st.markdown("### 📊 Orijinal Sinyal, Yeniden Oluşturulan Toplam ve FFT Spektrumu")
            dg_col1, dg_col2 = st.columns(2)

            with dg_col1:
                fig_do, ax_do = plt.subplots(figsize=(6.4, 3.4))
                ax_do.plot(
                    decomp["t_ms"], decomp["original"],
                    color="#1f77b4", linewidth=1.6, label="Orijinal Sinyal"
                )
                ax_do.plot(
                    decomp["t_ms"], decomp["reconstructed"],
                    color="#d62728", linestyle="--", linewidth=1.3, alpha=0.85,
                    label="Seçili Harmoniklerin Toplamı"
                )
                ax_do.plot(
                    decomp["t_ms"], decomp["fundamental"],
                    color="gray", linestyle=":", linewidth=1.1, alpha=0.75,
                    label=f"Temel Bileşen ({f0} Hz)"
                )
                ax_do.set_title("Orijinal Sinyal ve Harmonik Yeniden Oluşturma")
                ax_do.set_xlabel("Zaman (ms)")
                ax_do.set_ylabel("Genlik")
                ax_do.legend(loc="upper right", fontsize=7.8)
                ax_do.grid(True, alpha=0.3)
                fig_do.tight_layout()
                st.pyplot(fig_do)
                plt.close(fig_do)

            with dg_col2:
                fig_df, ax_df = plt.subplots(figsize=(6.4, 3.4))
                d_mask = decomp["xf"] <= d_fft_max
                dxf = decomp["xf"][d_mask]
                dyf = decomp["yf"][d_mask]
                if d_fft_scale == "dB Ölçek":
                    dyf_db = 20.0 * np.log10(np.maximum(dyf, 1e-6))
                    ax_df.plot(dxf, dyf_db, color="#d62728", linewidth=1.3)
                    ax_df.set_ylabel("Genlik (dB)")
                else:
                    ax_df.stem(dxf, dyf, linefmt="#d62728", markerfmt="ro", basefmt=" ")
                    ax_df.set_ylabel("Genlik")
                    for c in d_components:
                        if c["kind"] in ("fundamental", "harmonic") and c["freq"] <= d_fft_max:
                            idx_p = int(np.argmin(np.abs(dxf - c["freq"])))
                            ax_df.annotate(
                                f"h{int(round(c['order']))}\n({dyf[idx_p]:.2f})",
                                xy=(dxf[idx_p], dyf[idx_p]),
                                xytext=(0, 6),
                                textcoords="offset points",
                                ha="center",
                                fontsize=7.2,
                                color="#222222",
                            )
                ax_df.set_title("Sinyal FFT Frekans Spektrumu")
                ax_df.set_xlabel("Frekans (Hz)")
                ax_df.set_xlim(0, d_fft_max)
                ax_df.grid(True, alpha=0.3)
                fig_df.tight_layout()
                st.pyplot(fig_df)
                plt.close(fig_df)

            st.markdown("### 🧩 Ayrıştırılan Bileşenlerin Teker Teker Görünümü (Zaman Domeni + Bireysel FFT)")
            d_view_mode = st.radio(
                "Görünüm Biçimi:",
                ["Her Bileşen İçin Ayrı Zaman + FFT Satırı", "Tüm Ayrık Bileşenler Üst Üste (Karşılaştırmalı)"],
                horizontal=True,
                key="dec_view_mode",
            )
            render_isolated_components(
                t_ms=decomp["t_ms"],
                composite_sig=decomp["original"],
                components=d_components,
                fft_max_freq=d_fft_max,
                fft_scale=d_fft_scale,
                share_y_axis=d_share_y,
                view_mode=d_view_mode,
                composite_label="Orijinal Sinyal",
            )

            st.markdown("### 📋 Ayrıştırılan Harmonik Bileşenler Tablosu")
            dec_rows = [
                {
                    "Bileşen": c["name"],
                    "Frekans (Hz)": c["freq"] if c["kind"] != "residual" else np.nan,
                    "Harmonik Derecesi (h)": c["order"],
                    "Genlik (Tepe)": c["amp"],
                    "RMS": c["rms"],
                    "Faz (°)": c["phase_deg"],
                    "Temele Oran - HD (%)": c["hd_percent"],
                    "Enerji Payı (%)": c["power_share_pct"],
                }
                for c in d_components
            ]
            dec_df = pd.DataFrame(dec_rows)
            st.dataframe(
                dec_df.style.format(
                    {
                        "Frekans (Hz)": "{:.2f}",
                        "Harmonik Derecesi (h)": "{:.0f}",
                        "Genlik (Tepe)": "{:.4f}",
                        "RMS": "{:.4f}",
                        "Faz (°)": "{:.1f}",
                        "Temele Oran - HD (%)": "%{:.2f}",
                        "Enerji Payı (%)": "%{:.2f}",
                    },
                    na_rep="-",
                ),
                use_container_width=True,
                hide_index=True,
            )

            extracted_harms = [c for c in d_components if c["kind"] == "harmonic"]
            if extracted_harms:
                if st.button(
                    "🔁 Ayrıştırılan Harmonikleri Parametrik Sentezleyici Tablosuna Aktar",
                    use_container_width=True,
                    key="dec_send_to_synth",
                ):
                    st.session_state["harm_table_df"] = pd.DataFrame([
                        {
                            "Aktif": True,
                            "Frekans (Hz)": round(c["freq"], 2),
                            "Genlik": round(c["amp"], 4),
                            "Faz (°)": round(c["phase_deg"], 1),
                        }
                        for c in extracted_harms
                    ])
                    st.success(
                        f"{len(extracted_harms)} harmonik bileşen Parametrik Harmonik Sentezleyici tablosuna aktarıldı. "
                        "Yukarıdan 'Parametrik Harmonik Sentezleyici (Manuel Giriş)' moduna geçerek düzenleyebilirsiniz."
                    )