import numpy as np
from scipy.fft import fft, fftfreq
from scipy.stats import kurtosis, skew, entropy
import pywt


def extract_signal_features(signal, fs=5000, f0=50):
    N = len(signal)

    rms = np.sqrt(np.mean(signal**2))
    peak = np.max(np.abs(signal))
    crest_factor = peak / (rms + 1e-10)
    std_dev = np.std(signal)

    sig_kurtosis = kurtosis(signal, fisher=False)
    sig_skewness = skew(signal)
    energy = np.sum(signal**2)

    hist, _ = np.histogram(signal, bins=20, density=True)
    hist = hist[hist > 0]
    sig_entropy = entropy(hist, base=2)

    segments = np.array_split(signal, 5)
    seg_rms = np.array([np.sqrt(np.mean(seg**2)) for seg in segments])
    rms_min_ratio = np.min(seg_rms) / (rms + 1e-10)
    rms_max_ratio = np.max(seg_rms) / (rms + 1e-10)
    rms_env_std = np.std(seg_rms)

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

    coeffs = pywt.wavedec(signal, 'db4', level=3)
    wavelet_energy_d1 = np.sum(coeffs[-1]**2)
    wavelet_energy_d2 = np.sum(coeffs[-2]**2)
    wavelet_energy_a3 = np.sum(coeffs[0]**2)

    wavelet_max_d1 = np.max(np.abs(coeffs[-1]))
    wavelet_max_d2 = np.max(np.abs(coeffs[-2]))

    features = {
        "RMS": rms,
        "Peak": peak,
        "Crest_Factor": crest_factor,
        "Std_Dev": std_dev,
        "Kurtosis": sig_kurtosis,
        "Skewness": sig_skewness,
        "Energy": energy,
        "Entropy": sig_entropy,
        "RMS_Min_Ratio": rms_min_ratio,
        "RMS_Max_Ratio": rms_max_ratio,
        "RMS_Env_Std": rms_env_std,
        "THD_Percent": thd,
        "H3_Ratio": (v3 / (v1 + 1e-10)) * 100,
        "H5_Ratio": (v5 / (v1 + 1e-10)) * 100,
        "H7_Ratio": (v7 / (v1 + 1e-10)) * 100,
        "Wavelet_E_D1": wavelet_energy_d1,
        "Wavelet_E_D2": wavelet_energy_d2,
        "Wavelet_E_A3": wavelet_energy_a3,
        "Wavelet_Max_D1": wavelet_max_d1,
        "Wavelet_Max_D2": wavelet_max_d2
    }
    return features, xf, yf


def compute_single_sided_fft(signal, fs=5000, window="rect"):
    """
    Tek taraflı (single-sided) genlik ve faz FFT spektrumunu hesaplar.
    window: 'rect' (penceresiz, koherent örnekleme için) veya 'hann' (spektral sızıntıyı azaltmak için).
    """
    sig = np.asarray(signal, dtype=float)
    N = len(sig)
    if N < 2:
        return np.array([0.0]), np.array([0.0]), np.array([0.0])

    if window == "hann":
        win = np.hanning(N)
        coherent_gain = np.mean(win)
        sig_win = sig * win
    else:
        coherent_gain = 1.0
        sig_win = sig

    spec = np.fft.rfft(sig_win)
    xf = np.fft.rfftfreq(N, d=1.0 / fs)

    mag = np.abs(spec) / (N * coherent_gain)
    if N % 2 == 0:
        mag[1:-1] *= 2.0
    else:
        mag[1:] *= 2.0

    phase_deg = np.degrees(np.angle(spec))
    return xf, mag, phase_deg


def synthesize_harmonic_signal(
    f0=50.0,
    fs=5000.0,
    n_cycles=3,
    fundamental_amp=1.0,
    fundamental_phase_deg=0.0,
    harmonics=None,
    dc_offset=0.0,
    snr_db=None,
    window="rect",
    seed=42,
):
    """
    Kullanıcı tarafından girilen temel bileşen ve harmonik/ara-harmonik parametrelerinden
    bileşke sinyali, her bir ayrık bileşeni, tekil/bileşke FFT spektrumlarını ve
    elektriksel bozulma metriklerini (THD, RMS, Tepe Faktörü vb.) üretir.

    harmonics: list of dicts, örn:
        [{"freq": 150.0, "amp": 0.25, "phase_deg": 0.0, "active": True}, ...]
    """
    f0 = float(f0)
    fs = float(fs)
    n_cycles = max(1, int(n_cycles))
    N = max(16, int(round(n_cycles * fs / f0)))

    t = np.arange(N, dtype=float) / fs
    t_ms = t * 1000.0

    components = []
    clean_composite = np.zeros(N, dtype=float)

    if abs(float(dc_offset)) > 1e-12:
        dc_val = float(dc_offset)
        dc_sig = np.full(N, dc_val, dtype=float)
        xf_c, yf_c, _ = compute_single_sided_fft(dc_sig, fs=fs, window=window)
        components.append({
            "name": "DC Ofset (0 Hz)",
            "kind": "dc",
            "freq": 0.0,
            "order": 0.0,
            "amp": dc_val,
            "rms": abs(dc_val),
            "phase_deg": 0.0 if dc_val >= 0 else 180.0,
            "hd_percent": (abs(dc_val) / (abs(fundamental_amp) + 1e-10)) * 100.0,
            "signal": dc_sig,
            "xf": xf_c,
            "yf": yf_c,
        })
        clean_composite += dc_sig

    fund_amp = float(fundamental_amp)
    fund_phase = float(fundamental_phase_deg)
    fund_sig = fund_amp * np.sin(2.0 * np.pi * f0 * t + np.deg2rad(fund_phase))
    xf_f, yf_f, _ = compute_single_sided_fft(fund_sig, fs=fs, window=window)
    components.append({
        "name": f"Temel Bileşen (1. Harmonik - {f0:.1f} Hz)",
        "kind": "fundamental",
        "freq": f0,
        "order": 1.0,
        "amp": fund_amp,
        "rms": abs(fund_amp) / np.sqrt(2.0),
        "phase_deg": fund_phase,
        "hd_percent": 100.0,
        "signal": fund_sig,
        "xf": xf_f,
        "yf": yf_f,
    })
    clean_composite += fund_sig

    harmonic_sq_sum = 0.0
    nyquist_warnings = []

    if harmonics:
        for item in harmonics:
            if not item.get("active", True):
                continue
            freq = float(item.get("freq", 0.0))
            amp = float(item.get("amp", 0.0))
            phase = float(item.get("phase_deg", 0.0))
            if freq <= 0.0 or abs(amp) <= 1e-12:
                continue

            if freq >= fs / 2.0:
                nyquist_warnings.append(freq)

            order = freq / f0
            is_int_harmonic = abs(order - round(order)) < 1e-3 and round(order) >= 2
            if is_int_harmonic:
                h_int = int(round(order))
                label = f"{h_int}. Harmonik ({freq:.1f} Hz)"
                kind = "harmonic"
            else:
                label = f"Ara/Özel Bileşen ({freq:.1f} Hz | h={order:.2f})"
                kind = "interharmonic"

            comp_sig = amp * np.sin(2.0 * np.pi * freq * t + np.deg2rad(phase))
            xf_c, yf_c, _ = compute_single_sided_fft(comp_sig, fs=fs, window=window)

            components.append({
                "name": label,
                "kind": kind,
                "freq": freq,
                "order": order,
                "amp": amp,
                "rms": abs(amp) / np.sqrt(2.0),
                "phase_deg": phase,
                "hd_percent": (abs(amp) / (abs(fund_amp) + 1e-10)) * 100.0,
                "signal": comp_sig,
                "xf": xf_c,
                "yf": yf_c,
            })
            clean_composite += comp_sig
            harmonic_sq_sum += amp**2

    composite = clean_composite.copy()
    if snr_db is not None and snr_db < 60:
        rng = np.random.default_rng(seed)
        sig_power = np.mean(clean_composite**2)
        snr_linear = 10.0 ** (float(snr_db) / 10.0)
        noise_power = sig_power / (snr_linear + 1e-10)
        composite = clean_composite + rng.normal(0.0, np.sqrt(noise_power), size=N)

    xf, yf, phase_deg = compute_single_sided_fft(composite, fs=fs, window=window)

    comp_rms = float(np.sqrt(np.mean(composite**2)))
    comp_peak = float(np.max(np.abs(composite)))
    crest_factor = comp_peak / (comp_rms + 1e-10)
    thd_theoretical = (np.sqrt(harmonic_sq_sum) / (abs(fund_amp) + 1e-10)) * 100.0

    idx_f0 = int(np.argmin(np.abs(xf - f0)))
    v1_fft = float(yf[idx_f0])
    max_h = min(40, int((fs / 2.0) // f0))
    fft_harm_sq = 0.0
    for h in range(2, max_h + 1):
        idx_h = int(np.argmin(np.abs(xf - h * f0)))
        fft_harm_sq += float(yf[idx_h]) ** 2
    thd_fft = (np.sqrt(fft_harm_sq) / (v1_fft + 1e-10)) * 100.0

    total_rms_sq = sum(c["rms"] ** 2 for c in components) + 1e-12
    for c in components:
        c["power_share_pct"] = (c["rms"] ** 2 / total_rms_sq) * 100.0

    metrics = {
        "THD_Theoretical_Percent": thd_theoretical,
        "THD_FFT_Percent": thd_fft,
        "Composite_RMS": comp_rms,
        "Composite_Peak": comp_peak,
        "Crest_Factor": crest_factor,
        "Fundamental_RMS": abs(fund_amp) / np.sqrt(2.0),
        "Harmonic_RMS": np.sqrt(harmonic_sq_sum) / np.sqrt(2.0),
        "Freq_Resolution_Hz": fs / N,
        "Num_Samples": N,
        "Num_Components": len(components),
        "Nyquist_Warnings": nyquist_warnings,
    }

    return {
        "t": t,
        "t_ms": t_ms,
        "clean_composite": clean_composite,
        "composite": composite,
        "fundamental": fund_sig,
        "xf": xf,
        "yf": yf,
        "phase_deg": phase_deg,
        "components": components,
        "metrics": metrics,
    }


def _wrap_phase_deg(deg):
    """Faz açısını [-180, 180] aralığına normalize eder."""
    return float(((deg + 180.0) % 360.0) - 180.0)


def decompose_signal_harmonics(
    signal,
    fs=5000.0,
    f0=50.0,
    max_order=15,
    min_hd_percent=1.0,
    odd_only=False,
    include_dc=True,
    include_residual=True,
    window="rect",
):

    sig = np.asarray(signal, dtype=float)
    N = len(sig)
    fs = float(fs)
    f0 = float(f0)

    t = np.arange(N, dtype=float) / fs
    t_ms = t * 1000.0

    full_spec = np.fft.rfft(sig)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    xf_vis, yf_vis, _ = compute_single_sided_fft(sig, fs=fs, window=window)

    idx_f1 = int(np.argmin(np.abs(freqs - f0)))
    spec_f1 = np.zeros_like(full_spec)
    spec_f1[idx_f1] = full_spec[idx_f1]
    fund_sig = np.fft.irfft(spec_f1, n=N)

    scale_f1 = 2.0 / N if (0 < idx_f1 < len(full_spec) - 1 or N % 2 == 1) else 1.0 / N
    v1_amp = float(np.abs(full_spec[idx_f1]) * scale_f1)
    v1_phase = _wrap_phase_deg(np.degrees(np.angle(full_spec[idx_f1])) + 90.0)
    xf_f1, yf_f1, _ = compute_single_sided_fft(fund_sig, fs=fs, window=window)

    components = []
    used_bins = set()
    reconstructed = np.zeros(N, dtype=float)

    dc_val = float(full_spec[0].real / N)
    dc_hd = (abs(dc_val) / (v1_amp + 1e-10)) * 100.0
    if include_dc and dc_hd >= max(float(min_hd_percent), 0.5):
        spec_dc = np.zeros_like(full_spec)
        spec_dc[0] = full_spec[0]
        dc_sig = np.fft.irfft(spec_dc, n=N)
        xf_dc, yf_dc, _ = compute_single_sided_fft(dc_sig, fs=fs, window=window)
        components.append({
            "name": "DC Ofset (0 Hz)",
            "kind": "dc",
            "freq": 0.0,
            "order": 0.0,
            "amp": dc_val,
            "rms": abs(dc_val),
            "phase_deg": 0.0 if dc_val >= 0 else 180.0,
            "hd_percent": dc_hd,
            "signal": dc_sig,
            "xf": xf_dc,
            "yf": yf_dc,
        })
        used_bins.add(0)
        reconstructed += dc_sig

    components.append({
        "name": f"Temel Bileşen (1. Harmonik - {freqs[idx_f1]:.1f} Hz)",
        "kind": "fundamental",
        "freq": float(freqs[idx_f1]),
        "order": 1.0,
        "amp": v1_amp,
        "rms": float(np.sqrt(np.mean(fund_sig**2))),
        "phase_deg": v1_phase,
        "hd_percent": 100.0,
        "signal": fund_sig,
        "xf": xf_f1,
        "yf": yf_f1,
    })
    used_bins.add(idx_f1)
    reconstructed += fund_sig

    nyquist_max_h = int((fs / 2.0) // f0)
    effective_max_h = min(int(max_order), max(2, nyquist_max_h))
    all_harm_sq = 0.0
    detected_harmonics_count = 0

    for h in range(2, effective_max_h + 1):
        target_f = h * f0
        if target_f >= fs / 2.0:
            break
        idx_h = int(np.argmin(np.abs(freqs - target_f)))
        if idx_h in used_bins:
            continue

        scale_h = 2.0 / N if (0 < idx_h < len(full_spec) - 1 or N % 2 == 1) else 1.0 / N
        amp_h = float(np.abs(full_spec[idx_h]) * scale_h)
        all_harm_sq += amp_h ** 2
        hd_pct = (amp_h / (v1_amp + 1e-10)) * 100.0

        if odd_only and (h % 2 == 0):
            continue
        if hd_pct < float(min_hd_percent):
            continue

        spec_h = np.zeros_like(full_spec)
        spec_h[idx_h] = full_spec[idx_h]
        sig_h = np.fft.irfft(spec_h, n=N)
        phase_h = _wrap_phase_deg(np.degrees(np.angle(full_spec[idx_h])) + 90.0)
        xf_h, yf_h, _ = compute_single_sided_fft(sig_h, fs=fs, window=window)

        components.append({
            "name": f"{h}. Harmonik ({freqs[idx_h]:.1f} Hz)",
            "kind": "harmonic",
            "freq": float(freqs[idx_h]),
            "order": float(h),
            "amp": amp_h,
            "rms": float(np.sqrt(np.mean(sig_h**2))),
            "phase_deg": phase_h,
            "hd_percent": hd_pct,
            "signal": sig_h,
            "xf": xf_h,
            "yf": yf_h,
        })
        used_bins.add(idx_h)
        reconstructed += sig_h
        detected_harmonics_count += 1

    residual_sig = sig - reconstructed
    residual_rms = float(np.sqrt(np.mean(residual_sig**2)))
    orig_rms = float(np.sqrt(np.mean(sig**2)))
    residual_energy_pct = ((residual_rms**2) / (orig_rms**2 + 1e-12)) * 100.0

    if include_residual and residual_rms > 1e-6:
        xf_res, yf_res, _ = compute_single_sided_fft(residual_sig, fs=fs, window=window)
        components.append({
            "name": "Artık Bileşen (Gürültü / Diğer Frekanslar)",
            "kind": "residual",
            "freq": 0.0,
            "order": np.nan,
            "amp": float(np.max(np.abs(residual_sig))),
            "rms": residual_rms,
            "phase_deg": np.nan,
            "hd_percent": (residual_rms / (abs(v1_amp) / np.sqrt(2.0) + 1e-10)) * 100.0,
            "signal": residual_sig,
            "xf": xf_res,
            "yf": yf_res,
        })

    total_energy_sq = (orig_rms**2) + 1e-12
    for c in components:
        c["power_share_pct"] = (c["rms"] ** 2 / total_energy_sq) * 100.0

    thd_fft = (np.sqrt(all_harm_sq) / (v1_amp + 1e-10)) * 100.0
    orig_peak = float(np.max(np.abs(sig)))

    metrics = {
        "THD_FFT_Percent": thd_fft,
        "Original_RMS": orig_rms,
        "Original_Peak": orig_peak,
        "Crest_Factor": orig_peak / (orig_rms + 1e-10),
        "Fundamental_Amp": v1_amp,
        "Fundamental_RMS": float(np.sqrt(np.mean(fund_sig**2))),
        "Fundamental_Phase_Deg": v1_phase,
        "Detected_Harmonics": detected_harmonics_count,
        "Residual_RMS": residual_rms,
        "Residual_Energy_Pct": residual_energy_pct,
        "Freq_Resolution_Hz": fs / N,
        "Num_Samples": N,
        "DC_Offset": dc_val,
    }

    return {
        "t": t,
        "t_ms": t_ms,
        "original": sig,
        "reconstructed": reconstructed,
        "fundamental": fund_sig,
        "residual": residual_sig,
        "xf": xf_vis,
        "yf": yf_vis,
        "components": components,
        "metrics": metrics,
    }