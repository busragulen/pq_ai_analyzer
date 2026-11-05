import numpy as np
from scipy.fft import fft, fftfreq
from scipy.stats import kurtosis, skew, entropy
import pywt

def extract_signal_features(signal, fs=5000, f0=50):
    N = len(signal)
    
    # 1. İstatistiksel Özellikler
    rms = np.sqrt(np.mean(signal**2))
    peak = np.max(np.abs(signal))
    crest_factor = peak / (rms + 1e-10)
    std_dev = np.std(signal)
    sig_kurtosis = kurtosis(signal)
    sig_skewness = skew(signal)
    energy = np.sum(signal**2)
    
    hist, _ = np.histogram(signal, bins=20, density=True)
    hist = hist[hist > 0]
    sig_entropy = entropy(hist)
    
    # 2. FFT ve Harmonik Analizi
    yf = np.abs(fft(signal))[:N // 2] * (2.0 / N)
    xf = fftfreq(N, 1 / fs)[:N // 2]
    
    def get_harmonic_amp(h_order):
        target_freq = h_order * f0
        idx = np.argmin(np.abs(xf - target_freq))
        return yf[idx]
    
    v1 = get_harmonic_amp(1)
    v3 = get_harmonic_amp(3)
    v5 = get_harmonic_amp(5)
    v7 = get_harmonic_amp(7)
    
    harmonics_sq_sum = sum(get_harmonic_amp(h)**2 for h in range(2, 16))
    thd = (np.sqrt(harmonics_sq_sum) / (v1 + 1e-10)) * 100
    
    # 3. Ayrık Dalgacık Dönüşümü (Wavelet - db4)
    coeffs = pywt.wavedec(signal, 'db4', level=3)
    wavelet_energy_d1 = np.sum(coeffs[-1]**2)
    wavelet_energy_d2 = np.sum(coeffs[-2]**2)
    wavelet_energy_a3 = np.sum(coeffs[0]**2)
    
    features = {
        "RMS": rms,
        "Peak": peak,
        "Crest_Factor": crest_factor,
        "Std_Dev": std_dev,
        "Kurtosis": sig_kurtosis,
        "Skewness": sig_skewness,
        "Energy": energy,
        "Entropy": sig_entropy,
        "THD_Percent": thd,
        "H3_Ratio": (v3 / (v1 + 1e-10)) * 100,
        "H5_Ratio": (v5 / (v1 + 1e-10)) * 100,
        "H7_Ratio": (v7 / (v1 + 1e-10)) * 100,
        "Wavelet_E_D1": wavelet_energy_d1,
        "Wavelet_E_D2": wavelet_energy_d2,
        "Wavelet_E_A3": wavelet_energy_a3
    }
    return features, xf, yf