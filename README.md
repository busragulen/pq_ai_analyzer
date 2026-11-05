# Yapay Zeka ve Derin Öğrenme Destekli Güç Kalitesi Analiz Sistemi

Dual-Pipeline (Çift Kollu) yapay zeka mimarisiyle güç kalitesi bozulmalarının tespiti, SHAP tabanlı açıklanabilir yapay zeka analizleri ve etkileşimli karar destek arayüzü.

---

## Proje Klasör Yapısı

Proje, kodların ve çıktıların düzenli yönetilebilmesi için modüler bir mimaride yapılandırılmıştır:
* `src/`: Tüm Python kaynak kodları (`app.py`, `pq_analyzer.py`, `prepare_data.py`, `train_cnn.py`, `train_pipeline.py`)
* `models/`: Eğitilmiş model dosyaları ve encoder'lar (`pq_model_bundle.pkl`, `pq_cnn_model.pt`, `label_encoder.pkl`)
* `outputs/`: Test ve analiz raporu çıktıları (`noise_robustness_results.csv`, `shap_importance.csv`)
* `data/XPQRS/`: 17 sınıflı güç kalitesi sinyal veri seti

---

## Çalıştırma Adımları

Projeyi kendi bilgisayarınızda sıfırdan çalıştırmak için terminal üzerinden şu adımları takip edebilirsiniz:

### 1. Repoyu Klonlayın ve Bağımlılıkları Yükleyin
```bash
git clone [https://github.com/busragulen/pq_ai_analyzer.git]
cd pq_ai_analyzer
pip install -r requirements.txt

```

### 2. Veri Ön İşleme ve Öznitelik Çıkarımı

*(17 sınıflık ham sinyallerden 15 elektriksel/istatistiksel özniteliğin çıkarılması)*

```bash
python3 src/prepare_data.py

```

### 3. Klasik Makine Öğrenmesi Modellerinin Eğitimi

*(Modellerin eğitilmesi ve `pq_model_bundle.pkl` çıktısının `models/` altına üretilmesi)*

```bash
python3 src/train_pipeline.py

```

### 4. Derin Öğrenme Modelinin Eğitimi

*(1D-CNN modelinin AWGN gürültü zenginleştirmesiyle eğitilmesi ve robust stres testi)*

```bash
python3 src/train_cnn.py

```

### 5. Streamlit Arayüzünün Başlatılması

```bash
python3 -m streamlit run src/app.py

```

*(Komut sonrasında terminalde beliren `Local URL` (genellikle `http://localhost:8501`) bağlantısına tıklayarak arayüze tarayıcınız üzerinden ulaşabilir; farklı arıza sınıflarını seçerek sistemin yapay zeka kararlarını canlı olarak test edebilirsiniz.)*

---

## Windows Kullanıcıları İçin Notlar

Bu proje macOS ortamında Apple Silicon işlemci altyapısıyla geliştirilmiştir. Windows işletim sistemi kullanan geliştiricilerin projeyi klonlayıp çalıştırırken dikkate alması gereken hususlar:

1. **Donanım İvmesi (GPU / CPU):**
Kod içerisindeki PyTorch cihaz tanımlamaları Apple Silicon (`mps`) tabanlı tasarlandığından, Windows cihazlarda (NVIDIA ekran kartı kullanan sistemler için) PyTorch'un CUDA uyumlu sürümünün kurulu olduğundan emin olun. Aksi takdirde modeller otomatik olarak CPU üzerinden çalışacaktır.
2. **Sanal Ortam (Virtual Environment):**
Projeyi çalıştırmadan önce Windows terminalinde (CMD veya PowerShell) yeni bir sanal ortam oluşturmanız önerilir:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

```



```

```