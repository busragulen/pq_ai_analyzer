import os
import glob
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import shap
import joblib

def load_raw_xpqrs_data(dataset_folder="XPQRS"):
    csv_files = sorted(glob.glob(os.path.join(dataset_folder, "*.csv")))
    X_list, y_list = [], []
    print(f"XPQRS klasöründeki {len(csv_files)} sınıf okunuyor...")
    for file_path in csv_files:
        class_name = os.path.splitext(os.path.basename(file_path))[0]
        df = pd.read_csv(file_path, header=None)
        if isinstance(df.iloc[0, 0], str):
            df = pd.read_csv(file_path)
        data = df.values.astype(np.float32)
        if data.shape[0] < data.shape[1] and data.shape[1] == 1000:
            data = data.T
        X_list.append(data)
        y_list.extend([class_name] * data.shape[0])
        
    X = np.vstack(X_list)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_list)
    joblib.dump(encoder, "label_encoder.pkl")
    return X, y, encoder

class PQ_1DCNN(nn.Module):
    def __init__(self, num_classes=17):
        super(PQ_1DCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
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
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

def add_awgn_noise(signals, snr_db):
    noisy_signals = np.zeros_like(signals)
    for i in range(len(signals)):
        sig = signals[i]
        sig_power = np.mean(sig ** 2)
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = sig_power / (snr_linear + 1e-10)
        noise = np.random.normal(0, np.sqrt(noise_power), sig.shape)
        noisy_signals[i] = sig + noise
    return noisy_signals.astype(np.float32)

def train_model(model, train_loader, test_loader, device, epochs=12, name="Model"):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
    best_acc = 0.0
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
            
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)
                preds = model(bx).argmax(dim=1)
                correct += (preds == by).sum().item()
                total += by.size(0)
        acc = (correct / total) * 100
        scheduler.step(acc)
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())
        print(f"[{name}] Epoch {epoch+1:02d}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Test Doğruluğu: %{acc:.2f} (En İyi: %{best_acc:.2f})")
        
    model.load_state_dict(best_weights)
    return model

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Çalışma Ortamı: {device}")
    
    X, y, encoder = load_raw_xpqrs_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    test_dataset = TensorDataset(torch.tensor(X_test).unsqueeze(1), torch.tensor(y_test, dtype=torch.long))
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    #  1D-CNN temiz veri
    print("\n--- ADIM 1A: STANDART 1D-CNN EĞİTİMİ (SADECE TEMİZ VERİ) ---")
    clean_train_ds = TensorDataset(torch.tensor(X_train).unsqueeze(1), torch.tensor(y_train, dtype=torch.long))
    clean_train_loader = DataLoader(clean_train_ds, batch_size=64, shuffle=True)
    std_model = PQ_1DCNN(num_classes=len(encoder.classes_)).to(device)
    std_model = train_model(std_model, clean_train_loader, test_loader, device, epochs=10, name="Standart-CNN")
    # noise aware beyaz gürültü
    print("\n--- ADIM 1B: GÜRÜLTÜYE DAYANIKLI 1D-CNN EĞİTİMİ (TEMİZ + AWGN GÜRÜLTÜLÜ VERİ) ---")
    X_noisy_30 = add_awgn_noise(X_train, snr_db=30)
    X_noisy_20 = add_awgn_noise(X_train, snr_db=20)
    
    X_aug = np.vstack([X_train, X_noisy_30, X_noisy_20])
    y_aug = np.concatenate([y_train, y_train, y_train])
    
    aug_train_ds = TensorDataset(torch.tensor(X_aug).unsqueeze(1), torch.tensor(y_aug, dtype=torch.long))
    aug_train_loader = DataLoader(aug_train_ds, batch_size=64, shuffle=True)
    
    robust_model = PQ_1DCNN(num_classes=len(encoder.classes_)).to(device)
    robust_model = train_model(robust_model, aug_train_loader, test_loader, device, epochs=12, name="Robust-CNN")
    
    torch.save(robust_model.state_dict(), "pq_cnn_model.pt")
    print("✅ Gürültüye Dayanıklı 1D-CNN Modeli 'pq_cnn_model.pt' olarak kaydedildi!")
    
    
    print("\n--- ADIM 2: AWGN GÜRÜLTÜ DAYANIKLILIK KARŞILAŞTIRMASI ---")
    snr_levels = [40, 30, 25, 20]
    noise_results = []
    
    sample_idx = np.random.choice(len(X_test), 500, replace=False)
    X_sample, y_sample = X_test[sample_idx], y_test[sample_idx]
    
    for snr in snr_levels:
        X_noisy = add_awgn_noise(X_sample, snr_db=snr)
        bx = torch.tensor(X_noisy).unsqueeze(1).to(device)
        
        std_model.eval()
        robust_model.eval()
        with torch.no_grad():
            std_preds = std_model(bx).argmax(dim=1).cpu().numpy()
            rob_preds = robust_model(bx).argmax(dim=1).cpu().numpy()
            
        std_acc = accuracy_score(y_sample, std_preds) * 100
        rob_acc = accuracy_score(y_sample, rob_preds) * 100
        print(f"SNR = {snr} dB | Standart CNN: %{std_acc:.2f}  --->  Gürültüye Dayanıklı CNN: %{rob_acc:.2f}")
        noise_results.append({
            "SNR_dB": f"{snr} dB",
            "Standart_CNN": std_acc,
            "Robust_CNN": rob_acc
        })
        
    pd.DataFrame(noise_results).to_csv("noise_robustness_results.csv", index=False)
    print("✅ Karşılaştırmalı gürültü sonuçları 'noise_robustness_results.csv' olarak kaydedildi!")