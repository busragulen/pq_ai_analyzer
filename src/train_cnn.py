import os
import glob
import copy
import itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from concurrent.futures import ProcessPoolExecutor
import joblib

from pq_analyzer import extract_signal_features
from prepare_data import CLASS_NAME_MAP

def _extract_feat_worker(sig):
    feats, _, _ = extract_signal_features(sig, fs=5000, f0=50)
    return feats

def extract_features_batch(signals_matrix):
    """Gürültülü test sinyalleri için çok çekirdekli (multiprocessing) hızlı özellik çıkarımı."""
    with ProcessPoolExecutor() as executor:
        feat_list = list(executor.map(_extract_feat_worker, signals_matrix))
    return pd.DataFrame(feat_list).replace([np.inf, -np.inf], np.nan).fillna(0.0)

def load_raw_xpqrs_data(dataset_folder="data/XPQRS", merge_duplicate_pairs=True):
    csv_files = sorted(glob.glob(os.path.join(dataset_folder, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"'{dataset_folder}' içinde CSV bulunamadı!")
    X_list, y_list = [], []
    print(f"XPQRS klasöründeki {len(csv_files)} dosya okunuyor...")
    for file_path in csv_files:
        raw_name = os.path.splitext(os.path.basename(file_path))[0]
        
        if merge_duplicate_pairs:
            if raw_name == "Sag_with_Harmonics":
                raw_name = "Harmonics_with_Sag"
            elif raw_name == "Swell_with_Harmonics":
                raw_name = "Harmonics_with_Swell"
                
        tr_label = CLASS_NAME_MAP.get(raw_name, raw_name)
        
        df = pd.read_csv(file_path, header=None)
        if isinstance(df.iloc[0, 0], str):
            df = pd.read_csv(file_path)
        data = df.values.astype(np.float32)
        if data.shape[0] < data.shape[1] and data.shape[1] == 1000:
            data = data.T
        X_list.append(data)
        y_list.extend([tr_label] * data.shape[0])
        
    X = np.vstack(X_list)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_list)
    os.makedirs("models", exist_ok=True)
    joblib.dump(encoder, "models/label_encoder.pkl")
    print(f"Toplam {len(X)} sinyal ve {len(encoder.classes_)} benzersiz sınıf yüklendi.")
    return X, y, encoder

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

def add_awgn_noise(signals, snr_db, seed=None):
    rng = np.random.default_rng(seed)
    noisy_signals = np.zeros_like(signals)
    snr_linear = 10 ** (snr_db / 10.0)
    for i in range(len(signals)):
        sig = signals[i]
        sig_power = np.mean(sig ** 2)
        noise_power = sig_power / (snr_linear + 1e-10)
        noise = rng.normal(0, np.sqrt(noise_power), sig.shape)
        noisy_signals[i] = sig + noise
    return noisy_signals.astype(np.float32)

def evaluate_loader(model, loader, device, return_preds=False):
    model.eval()
    correct, total = 0, 0
    all_preds = []
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            preds = model(bx).argmax(dim=1)
            correct += (preds == by).sum().item()
            total += by.size(0)
            if return_preds:
                all_preds.append(preds.cpu().numpy())
    acc = (correct / total) * 100.0
    if return_preds:
        return acc, np.concatenate(all_preds)
    return acc

def run_cnn_grid_search(X_train, y_train, X_val, y_val, num_classes, device, search_epochs=6):
    print("\n1D-CNN HIZLI HİPER-PARAMETRE OPTİMİZASYONU (GRID SEARCH)")
    
    val_ds = TensorDataset(torch.tensor(X_val).unsqueeze(1), torch.tensor(y_val, dtype=torch.long))
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    
    param_grid = {
        "lr": [0.001, 0.002, 0.003],
        "batch_size": [32, 64],
        "dropout": [0.2, 0.3],
        "weight_decay": [0.0, 1e-4]
    }
    
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    print(f"Toplam {len(combinations)} farklı hiper-parametre kombinasyonu Doğrulama (Val) kümesinde taranacak...\n")
    
    best_val_acc = 0.0
    best_cfg = None
    
    for idx, cfg in enumerate(combinations, 1):
        torch.manual_seed(42)
        tr_ds = TensorDataset(torch.tensor(X_train).unsqueeze(1), torch.tensor(y_train, dtype=torch.long))
        tr_loader = DataLoader(tr_ds, batch_size=cfg["batch_size"], shuffle=True)
        
        model = PQ_1DCNN(num_classes=num_classes, dropout_rate=cfg["dropout"]).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
        )
        
        for _ in range(search_epochs):
            model.train()
            for bx, by in tr_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()
                
        val_acc = evaluate_loader(model, val_loader, device)
        
        print(f"[{idx:02d}/{len(combinations)}] {cfg} --> Doğrulama (Val) Başarımı: %{val_acc:.2f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_cfg = cfg
            
    print(f"\nEn İyi CNN Parametreleri (Val Kümesine Göre): {best_cfg} (Val Acc: %{best_val_acc:.2f})")
    return best_cfg

def train_model(model, train_loader, val_loader, device, lr=0.002, weight_decay=0.0, epochs=15, name="Model"):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    best_val_acc = 0.0
    best_epoch = 1
    best_weights = copy.deepcopy(model.state_dict())
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        val_acc = evaluate_loader(model, val_loader, device)
        scheduler.step(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_weights = copy.deepcopy(model.state_dict())
            
        print(f"[{name}] Epoch {epoch+1:02d}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Val Doğruluğu: %{val_acc:.2f} (En İyi Val: %{best_val_acc:.2f} @ Epoch {best_epoch})")
        
    print(f"--> [{name}] Doğrulama kümesine göre en iyi ağırlıklar (Epoch {best_epoch} - %{best_val_acc:.2f}) yüklendi.")
    model.load_state_dict(best_weights)
    return model

def analyze_std_cnn_40db_drop(std_model, X_test, y_test, y_test_tr, encoder, device):
    """Hocanın istediği: Standart CNN'in 40 dB'de düşmesinin nedenini sınıf bazlı karışmalarla raporlar."""
    X_40 = add_awgn_noise(X_test, snr_db=40, seed=42)
    ds_40 = TensorDataset(torch.tensor(X_40).unsqueeze(1), torch.tensor(y_test, dtype=torch.long))
    loader_40 = DataLoader(ds_40, batch_size=256, shuffle=False)
    
    acc_40, preds_40 = evaluate_loader(std_model, loader_40, device, return_preds=True)
    preds_40_tr = encoder.inverse_transform(preds_40)
    
    # 40 dB Sınıflandırma Raporu ve Karmaşıklık Matrisi
    rep_40 = classification_report(y_test_tr, preds_40_tr, output_dict=True)
    pd.DataFrame(rep_40).transpose().reset_index().rename(columns={"index": "Sınıf / Metrik"}).to_csv(
        "outputs/classification_report_std_cnn_40dB.csv", index=False
    )
    labels_sorted = sorted(np.unique(y_test_tr))
    cm_40 = confusion_matrix(y_test_tr, preds_40_tr, labels=labels_sorted)
    pd.DataFrame(cm_40, index=labels_sorted, columns=labels_sorted).to_csv("outputs/confusion_matrix_std_cnn_40dB.csv")
    
    cm_off_diag = cm_40.copy()
    np.fill_diagonal(cm_off_diag, 0)
    flat_indices = np.argsort(cm_off_diag.ravel())[::-1][:5]
    print(f"\n[ANALİZ] Standart-CNN 40 dB SNR Hata İncelemesi (Doğruluk: %{acc_40:.2f}):")
    for idx in flat_indices:
        r, c = divmod(idx, len(labels_sorted))
        if cm_off_diag[r, c] > 0:
            print(f"   -> Gerçek: '{labels_sorted[r]}' ==> Tahmin: '{labels_sorted[c]}' ({cm_off_diag[r, c]} örnek karıştı)")
    print("40 dB detaylı analiz tabloları 'outputs/classification_report_std_cnn_40dB.csv' ve 'outputs/confusion_matrix_std_cnn_40dB.csv' olarak kaydedildi.")

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Çalışma Ortamı: {device}")
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    X, y, encoder = load_raw_xpqrs_data(dataset_folder="data/XPQRS", merge_duplicate_pairs=True)
    num_classes = len(encoder.classes_)
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    print(f"Veri Bölünmesi -> Eğitim: {len(X_train)} (%70) | Doğrulama (Val): {len(X_val)} (%15) | Test: {len(X_test)} (%15)")
    
    best_cfg = run_cnn_grid_search(X_train, y_train, X_val, y_val, num_classes, device, search_epochs=6)
    
    val_dataset = TensorDataset(torch.tensor(X_val).unsqueeze(1), torch.tensor(y_val, dtype=torch.long))
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    
    test_dataset = TensorDataset(torch.tensor(X_test).unsqueeze(1), torch.tensor(y_test, dtype=torch.long))
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    print("\nSTANDART 1D-CNN EĞİTİMİ (SADECE TEMİZ VERİ - EPOCH SEÇİMİ VAL KÜMESİYLE)")
    clean_train_ds = TensorDataset(torch.tensor(X_train).unsqueeze(1), torch.tensor(y_train, dtype=torch.long))
    clean_train_loader = DataLoader(clean_train_ds, batch_size=best_cfg["batch_size"], shuffle=True)
    
    std_model = PQ_1DCNN(num_classes=num_classes, dropout_rate=best_cfg["dropout"]).to(device)
    std_model = train_model(
        std_model, clean_train_loader, val_loader, device,
        lr=best_cfg["lr"], weight_decay=best_cfg["weight_decay"], epochs=14, name="Standart-CNN"
    )
    std_clean_acc, std_clean_preds = evaluate_loader(std_model, test_loader, device, return_preds=True)
    print(f"★ [Standart-CNN] Nihai Kör Test (Temiz Test Set) Doğruluğu: %{std_clean_acc:.2f}")
    torch.save(std_model.state_dict(), "models/pq_std_cnn_model.pt")
    
    y_test_tr = encoder.inverse_transform(y_test)
    std_preds_tr = encoder.inverse_transform(std_clean_preds)
    pd.DataFrame(classification_report(y_test_tr, std_preds_tr, output_dict=True)).transpose().reset_index().rename(
        columns={"index": "Sınıf / Metrik"}
    ).to_csv("outputs/classification_report_std_cnn_clean.csv", index=False)
    
    # S-CNN 40 dB düşüş analizi
    analyze_std_cnn_40db_drop(std_model, X_test, y_test, y_test_tr, encoder, device)
    
    print("\nGÜRÜLTÜYE DAYANIKLI 1D-CNN EĞİTİMİ (TEMİZ + AWGN GÜRÜLTÜLÜ VERİ - EPOCH SEÇİMİ VAL KÜMESİYLE)")
    X_noisy_30 = add_awgn_noise(X_train, snr_db=30, seed=42)
    X_noisy_20 = add_awgn_noise(X_train, snr_db=20, seed=43)
    
    X_aug = np.vstack([X_train, X_noisy_30, X_noisy_20])
    y_aug = np.concatenate([y_train, y_train, y_train])
    
    X_val_noisy_30 = add_awgn_noise(X_val, snr_db=30, seed=42)
    X_val_noisy_20 = add_awgn_noise(X_val, snr_db=20, seed=43)
    X_val_aug = np.vstack([X_val, X_val_noisy_30, X_val_noisy_20])
    y_val_aug = np.concatenate([y_val, y_val, y_val])
    
    aug_train_ds = TensorDataset(torch.tensor(X_aug).unsqueeze(1), torch.tensor(y_aug, dtype=torch.long))
    aug_train_loader = DataLoader(aug_train_ds, batch_size=best_cfg["batch_size"], shuffle=True)
    
    aug_val_ds = TensorDataset(torch.tensor(X_val_aug).unsqueeze(1), torch.tensor(y_val_aug, dtype=torch.long))
    aug_val_loader = DataLoader(aug_val_ds, batch_size=128, shuffle=False)
    
    robust_model = PQ_1DCNN(num_classes=num_classes, dropout_rate=best_cfg["dropout"]).to(device)
    robust_model = train_model(
        robust_model, aug_train_loader, aug_val_loader, device,
        lr=best_cfg["lr"], weight_decay=best_cfg["weight_decay"], epochs=18, name="Robust-CNN"
    )
    rob_clean_acc, rob_clean_preds = evaluate_loader(robust_model, test_loader, device, return_preds=True)
    print(f"★ [Robust-CNN] Nihai Kör Test (Temiz Test Set) Doğruluğu: %{rob_clean_acc:.2f}")
    
    torch.save(robust_model.state_dict(), "models/pq_cnn_model.pt")
    print("Gürültüye Dayanıklı 1D-CNN Modeli 'models/pq_cnn_model.pt' olarak kaydedildi!")
    
    rob_preds_tr = encoder.inverse_transform(rob_clean_preds)
    
    cnn_report_dict = classification_report(y_test_tr, rob_preds_tr, output_dict=True)
    pd.DataFrame(cnn_report_dict).transpose().reset_index().rename(columns={"index": "Sınıf / Metrik"}).to_csv(
        "outputs/classification_report_cnn.csv", index=False
    )
    labels_sorted = sorted(np.unique(y_test_tr))
    cm_cnn = confusion_matrix(y_test_tr, rob_preds_tr, labels=labels_sorted)
    pd.DataFrame(cm_cnn, index=labels_sorted, columns=labels_sorted).to_csv("outputs/confusion_matrix_cnn.csv")
    print("1D-CNN F1 raporu ('outputs/classification_report_cnn.csv') ve Karmaşıklık Matrisi ('outputs/confusion_matrix_cnn.csv') kaydedildi.")

    ml_model, ml_scaler, ml_selector, ml_name = None, None, None, "Klasik ML"
    if os.path.exists("models/pq_model_bundle.pkl"):
        bundle = joblib.load("models/pq_model_bundle.pkl")
        if len(bundle) == 4:
            ml_model, ml_scaler, ml_selector, ml_name = bundle
        else:
            ml_model, ml_scaler, ml_selector = bundle
            ml_name = type(ml_model).__name__

    all_ml_models, all_scaler, all_selector = {}, None, None
    if os.path.exists("models/all_ml_models_bundle.pkl"):
        all_ml_models, all_scaler, all_selector = joblib.load("models/all_ml_models_bundle.pkl")

    print(f"\nAWGN GÜRÜLTÜ DAYANIKLILIK TESTİ (TÜM TEST KÜMESİ: {len(X_test)} ÖRNEK & 3 FARKLI TOHUM)")
    noise_seeds = [42, 100, 2026]
    snr_levels = [40, 30, 25, 20]
    noise_results = []
    all_models_noise_rows = []
    
    ml_clean_acc = np.nan
    clean_row_all = {
        "SNR_dB": "Temiz",
        "Robust 1D-CNN": f"%{rob_clean_acc:.2f} ± %0.00",
        "Standart 1D-CNN": f"%{std_clean_acc:.2f} ± %0.00"
    }
    
    if ml_model is not None or all_ml_models:
        print("Temiz test kümesi için klasik model özellikleri çıkarılıyor...")
        df_clean_feats = extract_features_batch(X_test)
        
        if ml_model is not None:
            X_clean_in = np.clip(ml_scaler.transform(df_clean_feats), -10.0, 10.0)
            if ml_selector is not None:
                X_clean_in = ml_selector.transform(X_clean_in)
            ml_clean_preds = ml_model.predict(X_clean_in)
            ml_clean_acc = accuracy_score(y_test_tr, ml_clean_preds) * 100.0
            
        if all_ml_models:
            X_clean_all = np.clip(all_scaler.transform(df_clean_feats), -10.0, 10.0)
            X_clean_sel = all_selector.transform(X_clean_all)
            for m_name, m_info in all_ml_models.items():
                X_in_m = X_clean_sel if m_info["use_selector"] else X_clean_all
                acc_m = accuracy_score(y_test_tr, m_info["estimator"].predict(X_in_m)) * 100.0
                clean_row_all[m_name] = f"%{acc_m:.2f} ± %0.00"

    all_models_noise_rows.append(clean_row_all)

    print(f"SNR = Temiz (∞ dB) | Standart CNN: %{std_clean_acc:.2f} | Robust CNN: %{rob_clean_acc:.2f} | {ml_name}: %{ml_clean_acc:.2f}")
    noise_results.append({
        "SNR_dB": "Temiz",
        "Standart_CNN": std_clean_acc,
        "Standart_CNN_Std": 0.0,
        "Standart_CNN_Format": f"%{std_clean_acc:.2f} ± %0.00",
        "Robust_CNN": rob_clean_acc,
        "Robust_CNN_Std": 0.0,
        "Robust_CNN_Format": f"%{rob_clean_acc:.2f} ± %0.00",
        "Klasik_ML": ml_clean_acc,
        "Klasik_ML_Std": 0.0,
        "Klasik_ML_Format": f"%{ml_clean_acc:.2f} ± %0.00"
    })
    
    for snr in snr_levels:
        std_accs, rob_accs, ml_accs = [], [], []
        per_model_accs = {m_name: [] for m_name in all_ml_models.keys()}
        
        for s in noise_seeds:
            X_noisy = add_awgn_noise(X_test, snr_db=snr, seed=s)
            noisy_ds = TensorDataset(torch.tensor(X_noisy).unsqueeze(1), torch.tensor(y_test, dtype=torch.long))
            noisy_loader = DataLoader(noisy_ds, batch_size=256, shuffle=False)
            
            s_acc = evaluate_loader(std_model, noisy_loader, device)
            r_acc = evaluate_loader(robust_model, noisy_loader, device)
            std_accs.append(s_acc)
            rob_accs.append(r_acc)
            
            if ml_model is not None or all_ml_models:
                df_noisy_feats = extract_features_batch(X_noisy)
                
                if ml_model is not None:
                    X_noisy_in = np.clip(ml_scaler.transform(df_noisy_feats), -10.0, 10.0)
                    if ml_selector is not None:
                        X_noisy_in = ml_selector.transform(X_noisy_in)
                    ml_preds = ml_model.predict(X_noisy_in)
                    m_acc = accuracy_score(y_test_tr, ml_preds) * 100.0
                    ml_accs.append(m_acc)
                    
                if all_ml_models:
                    X_noisy_all = np.clip(all_scaler.transform(df_noisy_feats), -10.0, 10.0)
                    X_noisy_sel = all_selector.transform(X_noisy_all)
                    for m_name, m_info in all_ml_models.items():
                        X_in_m = X_noisy_sel if m_info["use_selector"] else X_noisy_all
                        acc_m = accuracy_score(y_test_tr, m_info["estimator"].predict(X_in_m)) * 100.0
                        per_model_accs[m_name].append(acc_m)
                
        std_m, std_s = np.mean(std_accs), np.std(std_accs)
        rob_m, rob_s = np.mean(rob_accs), np.std(rob_accs)
        ml_m, ml_s = (np.mean(ml_accs), np.std(ml_accs)) if ml_accs else (np.nan, np.nan)
        
        print(
            f"SNR = {snr} dB | Standart CNN: %{std_m:.2f} ± %{std_s:.2f} | "
            f"Robust CNN: %{rob_m:.2f} ± %{rob_s:.2f} | "
            f"{ml_name}: %{ml_m:.2f} ± %{ml_s:.2f}"
        )
        noise_results.append({
            "SNR_dB": f"{snr} dB",
            "Standart_CNN": std_m,
            "Standart_CNN_Std": std_s,
            "Standart_CNN_Format": f"%{std_m:.2f} ± %{std_s:.2f}",
            "Robust_CNN": rob_m,
            "Robust_CNN_Std": rob_s,
            "Robust_CNN_Format": f"%{rob_m:.2f} ± %{rob_s:.2f}",
            "Klasik_ML": ml_m,
            "Klasik_ML_Std": ml_s,
            "Klasik_ML_Format": f"%{ml_m:.2f} ± %{ml_s:.2f}"
        })
        
        row_all = {
            "SNR_dB": f"{snr} dB",
            "Robust 1D-CNN": f"%{rob_m:.2f} ± %{rob_s:.2f}",
            "Standart 1D-CNN": f"%{std_m:.2f} ± %{std_s:.2f}"
        }
        for m_name, acc_list in per_model_accs.items():
            row_all[m_name] = f"%{np.mean(acc_list):.2f} ± %{np.std(acc_list):.2f}"
        all_models_noise_rows.append(row_all)
        
    pd.DataFrame(noise_results).to_csv("outputs/noise_robustness_results.csv", index=False)
    pd.DataFrame(all_models_noise_rows).to_csv("outputs/noise_robustness_all_models.csv", index=False)
    print("Kapsamlı gürültü dayanıklılık sonuçları 'outputs/noise_robustness_results.csv' ve 'outputs/noise_robustness_all_models.csv' olarak kaydedildi!")
