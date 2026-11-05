import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from pq_analyzer import extract_signal_features

# XPQRS karşılıkları
CLASS_NAME_MAP = {
    "Pure_Sinusoidal": "Normal Sinüs",
    "Sag": "Gerilim Çökmesi (Sag)",
    "Swell": "Gerilim Yükselmesi (Swell)",
    "Interruption": "Gerilim Kesintisi (Interruption)",
    "Harmonics": "Harmonik Bozulma",
    "Flicker": "Kırpışma (Flicker)",
    "Notch": "Çentiklenme (Notch)",
    "Transient": "Darbe Tipi Geçici Olay (Transient)",
    "Oscillatory_Transient": "Salınımlı Geçici Olay",
    "Flicker_with_Sag": "Kırpışma + Gerilim Çökmesi",
    "Flicker_with_Swell": "Kırpışma + Gerilim Yükselmesi",
    "Harmonics_with_Sag": "Harmonik + Gerilim Çökmesi",
    "Sag_with_Harmonics": "Gerilim Çökmesi + Harmonik",
    "Harmonics_with_Swell": "Harmonik + Gerilim Yükselmesi",
    "Swell_with_Harmonics": "Gerilim Yükselmesi + Harmonik",
    "Sag_with_Oscillatory_Transient": "Gerilim Çökmesi + Salınımlı Geçici",
    "Swell_with_Oscillatory_Transient": "Gerilim Yükselmesi + Salınımlı Geçici"
}

def load_xpqrs_dataset(dataset_folder="XPQRS", fs=5000, f0=50, merge_duplicate_pairs=False):
    csv_files = sorted(glob.glob(os.path.join(dataset_folder, "*.csv")))
    
    if not csv_files:
        raise FileNotFoundError(f"'{dataset_folder}' klasöründe CSV bulunamadı!")
        
    print(f"XPQRS Veri Setinde {len(csv_files)} adet CSV sınıfı bulundu.\n")
    
    all_features = []
    all_labels = []
    all_raw_names = []
    
    for file_path in csv_files:
        raw_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Harmonics_with_Sag ile Sag_with_Harmonics'i birleştir
        if merge_duplicate_pairs:
            if raw_name == "Sag_with_Harmonics":
                raw_name = "Harmonics_with_Sag"
            elif raw_name == "Swell_with_Harmonics":
                raw_name = "Harmonics_with_Swell"
                
        tr_label = CLASS_NAME_MAP.get(raw_name, raw_name)
        
        # CSV dosyasını başlık olmadan oku
        df = pd.read_csv(file_path, header=None)
        
        # Eğer ilk hücre metin ise tekrar başlıklı oku
        if isinstance(df.iloc[0, 0], str):
            df = pd.read_csv(file_path)
            
        data = df.values.astype(float)
        
        # matris (100, 1000) ise transpozunu al (1000, 100)
        if data.shape[0] < data.shape[1] and data.shape[1] == 1000:
            data = data.T
            
        print(f"İşleniyor: {raw_name:32s} | Boyut: {data.shape} -> {tr_label}")
        
        for i in range(data.shape[0]):
            signal = data[i, :]
            feats, _, _ = extract_signal_features(signal, fs=fs, f0=f0)
            all_features.append(feats)
            all_labels.append(tr_label)
            all_raw_names.append(raw_name)
            
    features_df = pd.DataFrame(all_features)
    features_df["Class_Code"] = all_raw_names
    features_df["Label"] = all_labels
    
    output_csv = "xpqrs_features.csv"
    features_df.to_csv(output_csv, index=False)
    print(f"\n✅ Toplam {len(features_df)} adet sinyal analiz edildi ve '{output_csv}' dosyasına kaydedildi!")
    return features_df

if __name__ == "__main__":
    load_xpqrs_dataset(dataset_folder="XPQRS", fs=5000, f0=50, merge_duplicate_pairs=False)