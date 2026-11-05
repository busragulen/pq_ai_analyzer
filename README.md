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
