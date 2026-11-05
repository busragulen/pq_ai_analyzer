import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_and_evaluate(csv_path="src/xpqrs_features.csv"):
    print(f"'{csv_path}' okunuyor...")
    df = pd.read_csv(csv_path)
    
    X = df.drop(columns=["Class_Code", "Label"])
    y = df["Label"].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # özellik seçim algo
    selector = SelectKBest(score_func=f_classif, k=12)
    X_train_sel = selector.fit_transform(X_train_scaled, y_train)
    X_test_sel = selector.transform(X_test_scaled)
    
    selected_cols = X.columns[selector.get_support()].tolist()
    print(f"\nSeçilen En Etkili Özellikler ({len(selected_cols)} adet):\n{selected_cols}\n")
    print("-" * 65)
    
    models = {
        "Kübik SVM (DVM)": SVC(kernel='poly', degree=3, C=10, probability=True, random_state=42),
        "RBF SVM (DVM)": SVC(kernel='rbf', C=10, probability=True, random_state=42),
        "Bagged Trees (Topluluk)": BaggingClassifier(n_estimators=50, random_state=42, n_jobs=-1),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    }
    
    best_model = None
    best_acc = 0
    best_name = ""
    
    for name, model in models.items():
        # Tüm özelliklerle eğitim
        model.fit(X_train_scaled, y_train)
        acc_all = accuracy_score(y_test, model.predict(X_test_scaled))
        
        # Seçilmiş özelliklerle eğitim
        model.fit(X_train_sel, y_train)
        acc_sel = accuracy_score(y_test, model.predict(X_test_sel))
        
        print(f"{name:25s} | Tüm Özellikler: %{acc_all*100:.2f} | Özellik Seçimli: %{acc_sel*100:.2f}")
        
        if acc_all >= acc_sel and acc_all > best_acc:
            best_acc = acc_all
            best_model = (model.fit(X_train_scaled, y_train), scaler, None)
            best_name = f"{name} (Tüm Özellikler)"
        elif acc_sel > best_acc:
            best_acc = acc_sel
            best_model = (model, scaler, selector)
            best_name = f"{name} (Özellik Seçimli)"
            
    print("-" * 65)
    print(f"\n🏆 En Başarılı Model: {best_name} -> Doğruluk: %{best_acc*100:.2f}")
    
    joblib.dump(best_model, "models/pq_model_bundle.pkl")
    print("Model 'pq_model_bundle.pkl' olarak kaydedildi.")

if __name__ == "__main__":
    train_and_evaluate()