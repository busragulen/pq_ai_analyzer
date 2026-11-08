import os
import numpy as np
import pandas as pd

def load_csv_matrix(file_path):
    df = pd.read_csv(file_path, header=None)
    if isinstance(df.iloc[0, 0], str):
        df = pd.read_csv(file_path)
    data = df.values.astype(np.float32)
    # Her zaman (1000 örnek, N zaman adımı) formatına zorla
    if data.shape[0] < data.shape[1] and data.shape[1] == 1000:
        data = data.T
    return data

def extract_pure_disturbance(signal_matrix, fs=5000, f0=50):
    """
    Faz kayması (phase mismatch) hatasını önlemek için 'harm - pure' yapmak yerine,
    her sinyalin kendi 50 Hz temel bileşenini (ve DC ofsetini) FFT ile süzerek 
    SADECE saf bozulma bileşenini (harmonik, darbe, çentik vb.) çıkarır.
    """
    N = signal_matrix.shape[1]
    freqs = np.fft.rfftfreq(N, d=1.0/fs)
    fft_coeffs = np.fft.rfft(signal_matrix, axis=1)
    
    fund_mask = (freqs <= (f0 + 20))
    fft_coeffs[:, fund_mask] = 0
    
    pure_disturbance = np.fft.irfft(fft_coeffs, n=N, axis=1)
    return pure_disturbance.astype(np.float32)

def apply_random_shift(residual_matrix, max_shift=15):
    shifted = np.zeros_like(residual_matrix)
    for i in range(residual_matrix.shape[0]):
        shift = np.random.randint(-max_shift, max_shift + 1)
        shifted[i] = np.roll(residual_matrix[i], shift)
    return shifted

def generate_hybrid_disturbances(dataset_folder="data/XPQRS", fs=5000, f0=50):
    print("Faz-Uyumlu Spektral Ayrıştırma ile 8 Yeni Melez Sınıf Türetiliyor...\n")
    
    required_files = [
        "Pure_Sinusoidal.csv", "Sag.csv", "Swell.csv", "Harmonics.csv", 
        "Flicker.csv", "Notch.csv", "Transient.csv", "Oscillatory_Transient.csv", 
        "Flicker_with_Sag.csv", "Flicker_with_Swell.csv"
    ]
    
    for req in required_files:
        req_path = os.path.join(dataset_folder, req)
        if not os.path.exists(req_path):
            raise FileNotFoundError(f"Gerekli temel sınıf bulunamadı: {req_path}")

    pure = load_csv_matrix(os.path.join(dataset_folder, "Pure_Sinusoidal.csv"))
    sag = load_csv_matrix(os.path.join(dataset_folder, "Sag.csv"))
    swell = load_csv_matrix(os.path.join(dataset_folder, "Swell.csv"))
    harm = load_csv_matrix(os.path.join(dataset_folder, "Harmonics.csv"))
    flicker = load_csv_matrix(os.path.join(dataset_folder, "Flicker.csv"))
    notch = load_csv_matrix(os.path.join(dataset_folder, "Notch.csv"))
    trans = load_csv_matrix(os.path.join(dataset_folder, "Transient.csv"))
    osc_trans = load_csv_matrix(os.path.join(dataset_folder, "Oscillatory_Transient.csv"))
    flicker_sag = load_csv_matrix(os.path.join(dataset_folder, "Flicker_with_Sag.csv"))
    flicker_swell = load_csv_matrix(os.path.join(dataset_folder, "Flicker_with_Swell.csv"))

    num_samples = pure.shape[0]
    np.random.seed(42)

    harm_res  = extract_pure_disturbance(harm, fs=fs, f0=f0)
    notch_res = extract_pure_disturbance(notch, fs=fs, f0=f0)
    trans_res = extract_pure_disturbance(trans, fs=fs, f0=f0)
    osc_res   = extract_pure_disturbance(osc_trans, fs=fs, f0=f0)

    w_harm_1 = np.random.uniform(0.75, 1.05, size=(num_samples, 1)).astype(np.float32)
    w_harm_2 = np.random.uniform(0.70, 1.00, size=(num_samples, 1)).astype(np.float32)
    w_notch  = np.random.uniform(0.85, 1.15, size=(num_samples, 1)).astype(np.float32)
    w_trans  = np.random.uniform(0.90, 1.25, size=(num_samples, 1)).astype(np.float32)
    w_osc    = np.random.uniform(0.85, 1.15, size=(num_samples, 1)).astype(np.float32)

    # 1. Kırpışma + Harmonik
    h_flicker_harm = flicker + (w_harm_1 * harm_res)

    # 2. Harmonik + Çentiklenme
    h_harm_notch = pure + (w_harm_1 * harm_res) + (w_notch * apply_random_shift(notch_res, 8))

    # 3. Gerilim Çökmesi + Harmonik + Darbe Geçicisi
    h_sag_harm_trans = sag + (w_harm_2 * harm_res) + (w_trans * apply_random_shift(trans_res, 12))

    # 4. Gerilim Yükselmesi + Harmonik + Darbe Geçicisi
    h_swell_harm_trans = swell + (w_harm_2 * harm_res) + (w_trans * apply_random_shift(trans_res, 12))

    # 5. Gerilim Çökmesi + Harmonik + Salınımlı Geçici
    h_sag_harm_osc = sag + (w_harm_2 * harm_res) + (w_osc * apply_random_shift(osc_res, 10))

    # 6. Gerilim Yükselmesi + Harmonik + Salınımlı Geçici
    h_swell_harm_osc = swell + (w_harm_2 * harm_res) + (w_osc * apply_random_shift(osc_res, 10))

    # 7. Kırpışma + Gerilim Çökmesi + Harmonik
    h_flicker_sag_harm = flicker_sag + (w_harm_2 * harm_res)

    # 8. Kırpışma + Gerilim Yükselmesi + Harmonik
    h_flicker_swell_harm = flicker_swell + (w_harm_2 * harm_res)

    hybrids = {
        "Flicker_with_Harmonics.csv": h_flicker_harm,
        "Harmonics_with_Notch.csv": h_harm_notch,
        "Sag_with_Harmonics_and_Transient.csv": h_sag_harm_trans,
        "Swell_with_Harmonics_and_Transient.csv": h_swell_harm_trans,
        "Sag_with_Harmonics_and_Oscillatory_Transient.csv": h_sag_harm_osc,
        "Swell_with_Harmonics_and_Oscillatory_Transient.csv": h_swell_harm_osc,
        "Flicker_with_Sag_and_Harmonics.csv": h_flicker_sag_harm,
        "Flicker_with_Swell_and_Harmonics.csv": h_flicker_swell_harm
    }

    for filename, matrix in hybrids.items():
        out_path = os.path.join(dataset_folder, filename)
        pd.DataFrame(matrix).to_csv(out_path, index=False, header=False)
        print(f"Sentezlendi: {filename:50s} | Boyut: {matrix.shape}")

    print("\nToplam 8 yeni Melez Bozulma sınıfı kusursuz faz uyumuyla üretildi!")

if __name__ == "__main__":
    generate_hybrid_disturbances()