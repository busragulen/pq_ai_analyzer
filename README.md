# Yapay Zeka ve Derin Öğrenme Destekli Güç Kalitesi Analiz Sistemi

Dual-Pipeline (Çift Kollu) yapay zeka mimarisiyle tekil ve melez güç kalitesi bozulmalarının (23 benzersiz sınıf) tespiti, SHAP tabanlı açıklanabilir yapay zeka analizleri, AWGN gürültü dayanıklılık testleri, harmonik sentezleme/ayrıştırma ve etkileşimli Streamlit karar destek arayüzü.

## Veri Seti ve Öznitelik Özeti

- **Temel Veri Seti:** [SEED_PQD_v1 (XPQRS)](https://zenodo.org/records/11843312) — 50 Hz temel frekans, 5 kHz örnekleme frekansı, 1 periyot (20 ms / 100 örnek), 17 ham bozulma sınıfı (sınıf başına 1.000 sinyal).
- **Veri Seti Lisansı:** Creative Commons Attribution 4.0 International (**CC BY 4.0**) — Khan, M. U., Aziz, S., & Usman, A. (2023), *XPQRS: Expert power quality recognition system for sensitive load applications*, *Measurement*, 216, 112889 ([DOI](https://doi.org/10.1016/j.measurement.2023.112889)).
- **Eş Sınıf Birleştirmesi ve Melez Zenginleştirme:** `Sag_with_Harmonics` / `Harmonics_with_Sag` ve `Swell_with_Harmonics` / `Harmonics_with_Swell` eş sınıf çiftleri birleştirilerek **15 temel sınıf** elde edilir; `src/augment_data.py` ile faz-uyumlu spektral ayrıştırma üzerinden **8 yeni melez sınıf** (2 ikili, 6 üçlü melez) sentezlenerek toplam **23 benzersiz sınıf** (25.000 sinyal) oluşturulur.
- **Öznitelik Seti (`src/pq_analyzer.py`):** Her sinyalden zaman domeni, FFT harmonik analizi ve Daubechies-4 (`db4`, 3 seviye) Dalgacık Dönüşümü (DWT) tabanlı **20 elektriksel ve istatistiksel öznitelik** çıkarılır (`RMS`, `Peak`, `Crest_Factor`, `Std_Dev`, `Kurtosis`, `Skewness`, `Energy`, `Entropy`, `RMS_Min_Ratio`, `RMS_Max_Ratio`, `RMS_Env_Std`, `THD_Percent`, `H3_Ratio`, `H5_Ratio`, `H7_Ratio`, `Wavelet_E_D1`, `Wavelet_E_D2`, `Wavelet_E_A3`, `Wavelet_Max_D1`, `Wavelet_Max_D2`).

## Proje Klasör Yapısı

- `src/`: Python kaynak kodları
  - `pq_analyzer.py`: 20 öznitelik çıkarımı, tek taraflı FFT, parametrik harmonik sentezleyici ve harmonik ayrıştırıcı
  - `augment_data.py`: Faz-uyumlu spektral ayrıştırma ile 8 sentetik melez sınıf üretimi
  - `prepare_data.py`: `data/XPQRS/` ham sinyallerinden proje kök dizininde `xpqrs_features.csv` üretimi
  - `train_pipeline.py`: 8 klasik ML algoritması için 5-katlı CV + Grid Search, ANOVA Top-16 seçimi, Doğrulama/Test Doğruluk ve Makro F1 hesaplaması, **SHAP (TreeExplainer) + ANOVA** karşılaştırması
  - `train_cnn.py`: 1D-CNN hiper-parametre taraması, Standart ve AWGN-dayanıklı (Robust) 1D-CNN eğitimi, 40 dB hata analizi ve çoklu tohumlu AWGN gürültü dayanıklılık kıyaslaması
  - `evaluate_15classes.py`: Sadece 15 birleştirilmiş orijinal sınıf (17.000 sinyal) üzerinde baz senaryo kıyaslaması
  - `app.py`: 5 sekmeli etkileşimli Streamlit arayüzü
  - `/sizinti_kontrolu.py`: Sentetik melez sınıflar için satır-indeksi grup bazlı bölme vs standart katmanlı bölme veri sızıntısı denetimi
- `data/XPQRS/`: Orijinal ve sentetik güç kalitesi CSV sinyal dosyaları
- `xpqrs_features.csv`: Çıkarılmış 20 öznitelikli veri tablosu (proje kök dizininde)
- `models/`: Eğitilmiş model paketleri
  - `pq_model_bundle.pkl` (en iyi klasik ML modeli, ölçekleyici ve seçici)
  - `all_ml_models_bundle.pkl` (tüm 8 klasik ML modeli)
  - `pq_std_cnn_model.pt` (Standart 1D-CNN ağırlıkları)
  - `pq_cnn_model.pt` (Robust 1D-CNN ağırlıkları)
  - `label_encoder.pkl` (1D-CNN sınıf etiketleyici)
- `outputs/`: Tüm değerlendirme, SHAP, gürültü ve sızıntı denetimi CSV çıktıları
  - `ml_grid_search_results.csv`, `classification_report_best_ml.csv`, `confusion_matrix_best_ml.csv`, `shap_importance.csv`
  - `classification_report_std_cnn_clean.csv`, `classification_report_std_cnn_40dB.csv`, `confusion_matrix_std_cnn_40dB.csv`
  - `classification_report_cnn.csv`, `confusion_matrix_cnn.csv`
  - `noise_robustness_results.csv`, `noise_robustness_all_models.csv`
  - `xpqrs_15_classes_benchmark.csv`
  - `sizinti_kontrolu_sonuclari.csv`, `sizinti_kontrolu_ozet.csv`, `sizinti_kopya_ozeti.csv`

## Çalıştırma Adımları

> **Not:** `data/XPQRS/` sinyal dosyaları ve `xpqrs_features.csv` repoda hazır olduğundan, yeniden üretim için doğrudan 2. adımdan (model eğitimi) başlayabilirsiniz.

### 1. Repoyu Klonlayın ve Bağımlılıkları Yükleyin

```bash
git clone https://github.com/busragulen/pq_ai_analyzer.git
cd pq_ai_analyzer
pip install -r requirements.txt
```

### (İsteğe Bağlı) Sıfırdan Melez Veri ve Öznitelik Üretimi

Sadece ham sinyallerden `xpqrs_features.csv` dosyasını baştan oluşturmak isterseniz proje kök dizininden çalıştırın:

```bash
python3 src/augment_data.py
python3 src/prepare_data.py
```

### 2. Klasik Makine Öğrenmesi Modelleri ve SHAP Analizi

8 klasik ML algoritmasını 5-katlı çapraz doğrulama + Grid Search ile eğitir, Doğruluk ve Makro F1 metriklerini kaydeder ve `outputs/shap_importance.csv` SHAP + ANOVA analizini üretir:

```bash
python3 src/train_pipeline.py
```

### 3. Derin Öğrenme (1D-CNN) Eğitimi ve AWGN Gürültü Stres Testi

Standart ve Robust 1D-CNN modellerini eğitir, 40 dB hata analizini yapar ve tüm modellerin 40/30/25/20 dB AWGN dayanıklılığını raporlar:

```bash
python3 src/train_cnn.py
```

### 4. 15 Sınıflı Orijinal XPQRS Baz Senaryo Kıyaslaması

```bash
python3 src/evaluate_15classes.py
```

### 5. Eğitim / Test Veri Sızıntısı Denetimi

Sentetik melez sınıflarda satır-indeksi grup bölmesi ile standart katmanlı bölme sonuçlarını karşılaştırır:

```bash
python3 rapor/kontrol/sizinti_kontrolu.py
```

### 6. Streamlit Karar Destek Arayüzünün Başlatılması

```bash
python3 -m streamlit run src/app.py
```

Terminalde görünen `Local URL` (genellikle `http://localhost:8501`) üzerinden arayüzü açabilirsiniz.

## Windows ve macOS Kullanıcıları İçin Notlar

1. **Donanım İvmesi (MPS / CUDA / CPU):** PyTorch cihaz seçimi sırasıyla Apple Silicon (`mps`), NVIDIA (`cuda`) ve `cpu` şeklinde otomatiktir. Windows sistemlerde NVIDIA GPU kullanmak için PyTorch'un CUDA uyumlu sürümünü kurabilirsiniz.
2. **Sanal Ortam (Virtual Environment):**

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate
pip install -r requirements.txt
```
