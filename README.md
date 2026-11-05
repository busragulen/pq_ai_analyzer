# pq_ai_analyzer
Dual-Pipeline yapay zeka mimarisiyle güç kalitesi bozulmalarının tespiti ve etkileşimli karar destek arayüzü.

# 1. 17 sınıflık ham sinyallerden 15 özniteliğin çıkarılması
python3 prepare_data.py

# 2. Klasik Makine Öğrenmesi modellerinin eğitilmesi ve pq_model_bundle.pkl üretimi
python3 train_pipeline.py

# 3. 1D-CNN modelinin AWGN gürültü zenginleştirmesiyle eğitilmesi ve stres testi
python3 train_cnn.py

----------------------
streamlit run app.py
----------------------
## Windows Kullanıcıları İçin Notlar

Bu proje macOS ortamında Apple Silicon M4 işlemci ile geliştirilmiştir. Windows işletim sistemi kullanan geliştiricilerin projeyi klonlayıp çalıştırırken dikkate alması gereken hususlar var lütfen sorun yaşamamak için bunlara dikkat ediniz:

1. **Donanım İvmesi (GPU / CPU):** 
   Kod içerisindeki PyTorch cihaz tanımlamaları Apple Silicon (`mps`) tabanlı tasarlandığından, Windows cihazlarda (NVIDIA ekran kartı kullanan sistemler için) PyTorch'un CUDA uyumlu sürümünün kurulu olduğundan emin olun. Aksi takdirde modeller CPU üzerinden çalışacaktır.
2. **Sanal Ortam (Virtual Environment):**
   Projeyi çalıştırmadan önce Windows terminalinde (CMD veya PowerShell) yeni bir sanal ortam oluşturmanız önerilir:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
