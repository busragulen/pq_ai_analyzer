import os
import glob
import copy
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from train_cnn import PQ_1DCNN, evaluate_loader

warnings.filterwarnings("ignore")

# Sentezlenen 8  sınıf
SYNTHETIC_HYBRID_CLASSES = {
    "Flicker_with_Harmonics",
    "Harmonics_with_Notch",
    "Sag_with_Harmonics_and_Transient",
    "Swell_with_Harmonics_and_Transient",
    "Sag_with_Harmonics_and_Oscillatory_Transient",
    "Swell_with_Harmonics_and_Oscillatory_Transient",
    "Flicker_with_Sag_and_Harmonics",
    "Flicker_with_Swell_and_Harmonics"
}

def run_15_class_benchmark():
    print("=" * 90)
    print("ORİJİNAL XPQRS (EŞ SINIF ÇİFTLERİ BİRLEŞTİRİLMİŞ - 15 SINIF / 17.000 SİNYAL) ANALİZİ")
    print("=" * 90)

    df = pd.read_csv("xpqrs_features.csv")
    df_15 = df[~df["Class_Code"].isin(SYNTHETIC_HYBRID_CLASSES)].copy()

    X = df_15.drop(columns=["Class_Code", "Label"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df_15["Label"].values

    print(f"Klasik ML Veri Seti : {len(df_15)} sinyal | Benzersiz Sınıf Sayısı: {len(np.unique(y))}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train_sc = np.clip(scaler.fit_transform(X_train), -10.0, 10.0)
    X_val_sc = np.clip(scaler.transform(X_val), -10.0, 10.0)
    X_test_sc = np.clip(scaler.transform(X_test), -10.0, 10.0)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    ml_models = {
        "Kübik SVM (DVM)": SVC(kernel="poly", degree=3, C=50, coef0=1.0, gamma="auto", random_state=42),
        "RBF SVM (DVM)": SVC(kernel="rbf", C=100, gamma="scale", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=5, criterion="entropy", random_state=42)
    }

    results = []
    for name, model in ml_models.items():
        cv_scores = cross_val_score(model, X_train_sc, y_train, cv=cv, scoring="accuracy", n_jobs=-1) * 100
        model.fit(X_train_sc, y_train)
        val_acc = accuracy_score(y_val, model.predict(X_val_sc)) * 100
        test_acc = accuracy_score(y_test, model.predict(X_test_sc)) * 100
        print(f"[{name:18s}] 5-Fold CV: %{cv_scores.mean():.2f} ± %{cv_scores.std():.2f} | Val: %{val_acc:.2f} | Test: %{test_acc:.2f}")
        results.append({
            "Model": name,
            "Sınıf_Sayısı": 15,
            "CV_Mean_Std": f"%{cv_scores.mean():.2f} ± %{cv_scores.std():.2f}",
            "Val_Acc": val_acc,
            "Test_Acc": test_acc
        })

    csv_files = sorted(glob.glob(os.path.join("data/XPQRS", "*.csv")))
    X_raw_list, y_raw_list = [], []
    for fp in csv_files:
        raw_name = os.path.splitext(os.path.basename(fp))[0]
        if raw_name in SYNTHETIC_HYBRID_CLASSES:
            continue
        if raw_name == "Sag_with_Harmonics":
            raw_name = "Harmonics_with_Sag"
        elif raw_name == "Swell_with_Harmonics":
            raw_name = "Harmonics_with_Swell"

        raw_df = pd.read_csv(fp, header=None)
        if isinstance(raw_df.iloc[0, 0], str):
            raw_df = pd.read_csv(fp)
        data = raw_df.values.astype(np.float32)
        if data.shape[0] < data.shape[1] and data.shape[1] == 1000:
            data = data.T
        X_raw_list.append(data)
        y_raw_list.extend([raw_name] * data.shape[0])

    X_raw = np.vstack(X_raw_list)
    enc = LabelEncoder()
    y_raw = enc.fit_transform(y_raw_list)

    X_tr_c, X_tmp_c, y_tr_c, y_tmp_c = train_test_split(
        X_raw, y_raw, test_size=0.30, random_state=42, stratify=y_raw
    )
    X_val_c, X_te_c, y_val_c, y_te_c = train_test_split(
        X_tmp_c, y_tmp_c, test_size=0.50, random_state=42, stratify=y_tmp_c
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(42)

    tr_loader = DataLoader(TensorDataset(torch.tensor(X_tr_c).unsqueeze(1), torch.tensor(y_tr_c, dtype=torch.long)), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val_c).unsqueeze(1), torch.tensor(y_val_c, dtype=torch.long)), batch_size=128, shuffle=False)
    te_loader = DataLoader(TensorDataset(torch.tensor(X_te_c).unsqueeze(1), torch.tensor(y_te_c, dtype=torch.long)), batch_size=128, shuffle=False)

    cnn_15 = PQ_1DCNN(num_classes=len(enc.classes_), dropout_rate=0.2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn_15.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_val_c = 0.0
    best_w = copy.deepcopy(cnn_15.state_dict())

    print("\n[1D-CNN (15 Sınıf)] Eğitim Başlıyor (14 Epoch - Val Seçimli)...")
    for ep in range(14):
        cnn_15.train()
        for bx, by in tr_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(cnn_15(bx), by)
            loss.backward()
            optimizer.step()
        v_acc = evaluate_loader(cnn_15, val_loader, device)
        scheduler.step(v_acc)
        if v_acc > best_val_c:
            best_val_c = v_acc
            best_w = copy.deepcopy(cnn_15.state_dict())

    cnn_15.load_state_dict(best_w)
    te_acc_c = evaluate_loader(cnn_15, te_loader, device)
    print(f"[1D-CNN (15 Sınıf) ] Doğrulama (Val): %{best_val_c:.2f} | Kör Test (Test Set): %{te_acc_c:.2f}")

    results.append({
        "Model": "1D-CNN (Derin Öğrenme)",
        "Sınıf_Sayısı": 15,
        "CV_Mean_Std": "-",
        "Val_Acc": best_val_c,
        "Test_Acc": te_acc_c
    })

    os.makedirs("outputs", exist_ok=True)
    pd.DataFrame(results).to_csv("outputs/xpqrs_15_classes_benchmark.csv", index=False)
    print("=" * 90)
    print("15 Sınıflı Baz Senaryo Sonuçları 'outputs/xpqrs_15_classes_benchmark.csv' dosyasına kaydedildi!")

if __name__ == "__main__":
    run_15_class_benchmark()