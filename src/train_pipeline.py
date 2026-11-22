import os
os.environ["PYTHONWARNINGS"] = "ignore"
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import shap
import joblib

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def train_and_evaluate(csv_path="xpqrs_features.csv"):
    if not os.path.exists(csv_path):
        root_csv = os.path.join(PROJECT_ROOT, csv_path)
        if os.path.exists(root_csv):
            csv_path = root_csv

    print(f"'{csv_path}' okunuyor...")
    df = pd.read_csv(csv_path)

    X = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X = X.drop(columns=["Class_Code", "Label"])
    y = df["Label"].values
    num_total_feats = X.shape[1]
    k_best = min(16, num_total_feats)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )
    print(f"Veri Bölünmesi -> Eğitim: {len(X_train)} (%70) | Doğrulama (Val): {len(X_val)} (%15) | Test: {len(X_test)} (%15)")

    scaler = StandardScaler()
    X_train_scaled = np.clip(scaler.fit_transform(X_train), -10.0, 10.0)
    X_val_scaled = np.clip(scaler.transform(X_val), -10.0, 10.0)
    X_test_scaled = np.clip(scaler.transform(X_test), -10.0, 10.0)

    selector = SelectKBest(score_func=f_classif, k=k_best)
    X_train_sel = selector.fit_transform(X_train_scaled, y_train)
    X_val_sel = selector.transform(X_val_scaled)
    X_test_sel = selector.transform(X_test_scaled)

    selected_cols = X.columns[selector.get_support()].tolist()
    print(f"\nANOVA ile Seçilen En Etkili Özellikler ({len(selected_cols)} / {num_total_feats} adet):\n{selected_cols}\n")
    print("=" * 95)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    model_grids = {
        "Kübik SVM (DVM)": {
            "model": SVC(kernel='poly', degree=3, probability=True, random_state=42),
            "params": {
                "C": [1, 10, 50, 100],
                "gamma": ["scale", "auto"],
                "coef0": [0.0, 0.5, 1.0]
            }
        },
        "RBF SVM (DVM)": {
            "model": SVC(kernel='rbf', probability=True, random_state=42),
            "params": {
                "C": [1, 10, 50, 100],
                "gamma": ["scale", "auto", 0.01, 0.1]
            }
        },
        "Random Forest": {
            "model": RandomForestClassifier(oob_score=True, random_state=42),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [None, 20, 30],
                "min_samples_split": [2, 5],
                "criterion": ["gini", "entropy"]
            }
        },
        "Bagged Trees (Topluluk)": {
            "model": BaggingClassifier(oob_score=True, random_state=42),
            "params": {
                "n_estimators": [50, 100],
                "max_samples": [0.8, 1.0],
                "max_features": [0.8, 1.0]
            }
        },
        "YSA / MLP (Çok Katmanlı Algılayıcı)": {
            "model": MLPClassifier(max_iter=300, early_stopping=False, random_state=42),
            "params": {
                "hidden_layer_sizes": [(64, 32), (128, 64)],
                "alpha": [1e-3, 1e-2],
                "learning_rate_init": [0.001, 0.002]
            }
        },
        "Karar Ağacı (Decision Tree)": {
            "model": DecisionTreeClassifier(random_state=42),
            "params": {
                "max_depth": [None, 15, 25],
                "min_samples_split": [2, 5, 10],
                "criterion": ["gini", "entropy"]
            }
        },
        "KNN": {
            "model": KNeighborsClassifier(algorithm="kd_tree"),
            "params": {
                "n_neighbors": [3, 5, 7],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"]
            }
        },
        "Naive Bayes (Gaussian)": {
            "model": GaussianNB(),
            "params": {
                "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]
            }
        }
    }

    best_bundle = None
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_test_acc = 0.0
    best_test_f1 = 0.0
    best_name = ""
    best_params_summary = {}
    best_test_preds = None
    grid_report = []

    all_trained_models = {}
    best_rf_all_features = None

    for name, cfg in model_grids.items():
        print(f"[{name}] için 5-Katlı Çapraz Doğrulama (Grid Search) çalışıyor...")

        # 1. Tüm Özellikler ile Eğitim ve Doğrulama
        gs_all = GridSearchCV(
            cfg["model"], cfg["params"], cv=cv, scoring="accuracy", n_jobs=-1
        )
        gs_all.fit(X_train_scaled, y_train)
        cv_mean_all = gs_all.best_score_ * 100
        cv_std_all = gs_all.cv_results_["std_test_score"][gs_all.best_index_] * 100
        preds_val_all = gs_all.best_estimator_.predict(X_val_scaled)
        val_acc_all = accuracy_score(y_val, preds_val_all)
        val_f1_all = f1_score(y_val, preds_val_all, average="macro")
        preds_test_all = gs_all.best_estimator_.predict(X_test_scaled)
        test_acc_all = accuracy_score(y_test, preds_test_all)
        test_f1_all = f1_score(y_test, preds_test_all, average="macro")

        # 2. Seçili Özellikler ile Eğitim ve Doğrulama
        gs_sel = GridSearchCV(
            cfg["model"], cfg["params"], cv=cv, scoring="accuracy", n_jobs=-1
        )
        gs_sel.fit(X_train_sel, y_train)
        cv_mean_sel = gs_sel.best_score_ * 100
        cv_std_sel = gs_sel.cv_results_["std_test_score"][gs_sel.best_index_] * 100
        preds_val_sel = gs_sel.best_estimator_.predict(X_val_sel)
        val_acc_sel = accuracy_score(y_val, preds_val_sel)
        val_f1_sel = f1_score(y_val, preds_val_sel, average="macro")
        preds_test_sel = gs_sel.best_estimator_.predict(X_test_sel)
        test_acc_sel = accuracy_score(y_test, preds_test_sel)
        test_f1_sel = f1_score(y_test, preds_test_sel, average="macro")

        oob_all_str, oob_sel_str = "-", "-"
        if hasattr(gs_all.best_estimator_, "oob_score_"):
            oob_all_str = f"%{gs_all.best_estimator_.oob_score_*100:.2f}"
            oob_sel_str = f"%{gs_sel.best_estimator_.oob_score_*100:.2f}"

        print(
            f"   ├─ Tüm Özellikler ({num_total_feats}) : 5-Fold CV: %{cv_mean_all:.2f} ± %{cv_std_all:.2f} "
            f"| Val: %{val_acc_all*100:.2f} (Makro F1: %{val_f1_all*100:.2f}) "
            f"| Test: %{test_acc_all*100:.2f} (Makro F1: %{test_f1_all*100:.2f}) | Param: {gs_all.best_params_}"
        )
        print(
            f"   └─ Seçili Özellik ({k_best}) : 5-Fold CV: %{cv_mean_sel:.2f} ± %{cv_std_sel:.2f} "
            f"| Val: %{val_acc_sel*100:.2f} (Makro F1: %{val_f1_sel*100:.2f}) "
            f"| Test: %{test_acc_sel*100:.2f} (Makro F1: %{test_f1_sel*100:.2f}) | Param: {gs_sel.best_params_}"
        )
        if oob_all_str != "-":
            print(f"   └─ OOB (Torba Dışı) Skoru -> Tüm Öz.: {oob_all_str} | Seçili Öz.: {oob_sel_str}")
        print()

        if name == "Random Forest":
            best_rf_all_features = gs_all.best_estimator_

        # Her modelin doğrulama kümesinde daha başarılı olan versiyonunu gürültü testi için sakla
        if val_acc_all >= val_acc_sel:
            all_trained_models[name] = {
                "estimator": gs_all.best_estimator_,
                "use_selector": False
            }
        else:
            all_trained_models[name] = {
                "estimator": gs_sel.best_estimator_,
                "use_selector": True
            }

        grid_report.append({
            "Model": name,
            "CV_Mean_Std_All": f"%{cv_mean_all:.2f} ± %{cv_std_all:.2f}",
            "OOB_All": oob_all_str,
            "Val_Acc_All": val_acc_all * 100,
            "Val_Macro_F1_All": val_f1_all * 100,
            "Acc_All_Features": test_acc_all * 100,
            "Test_Macro_F1_All": test_f1_all * 100,
            "Best_Params_All": str(gs_all.best_params_),
            "CV_Mean_Std_Sel": f"%{cv_mean_sel:.2f} ± %{cv_std_sel:.2f}",
            "OOB_Selected": oob_sel_str,
            "Val_Acc_Selected": val_acc_sel * 100,
            "Val_Macro_F1_Selected": val_f1_sel * 100,
            "Acc_Selected_Features": test_acc_sel * 100,
            "Test_Macro_F1_Selected": test_f1_sel * 100,
            "Best_Params_Selected": str(gs_sel.best_params_)
        })

        # MODEL SEÇİMİ YALNIZCA DOĞRULAMA (VAL) KÜMESİNE GÖRE YAPILIR
        if val_acc_all >= val_acc_sel and val_acc_all > best_val_acc:
            best_val_acc = val_acc_all
            best_val_f1 = val_f1_all
            best_test_acc = test_acc_all
            best_test_f1 = test_f1_all
            best_name = f"{name} (Tüm Özellikler)"
            best_bundle = (gs_all.best_estimator_, scaler, None, best_name)
            best_params_summary = gs_all.best_params_
            best_test_preds = preds_test_all
        elif val_acc_sel > best_val_acc:
            best_val_acc = val_acc_sel
            best_val_f1 = val_f1_sel
            best_test_acc = test_acc_sel
            best_test_f1 = test_f1_sel
            best_name = f"{name} (Özellik Seçimli)"
            best_bundle = (gs_sel.best_estimator_, scaler, selector, best_name)
            best_params_summary = gs_sel.best_params_
            best_test_preds = preds_test_sel

    print("=" * 95)
    print(f"Doğrulama Kümesine Göre Seçilen En Başarılı Model : {best_name}")
    print(f"Optimum Parametreler                             : {best_params_summary}")
    print(f"Doğrulama (Validation) Doğruluğu / Makro F1      : %{best_val_acc*100:.2f} / %{best_val_f1*100:.2f}")
    print(f"Nihai Kör Test (Test Set) Doğruluğu / Makro F1   : %{best_test_acc*100:.2f} / %{best_test_f1*100:.2f}")
    if hasattr(best_bundle[0], "oob_score_") and best_bundle[0].oob_score_:
        print(f"OOB (Torba Dışı) Skoru                           : %{best_bundle[0].oob_score_*100:.2f}")

    models_dir = os.path.join(PROJECT_ROOT, "models")
    outputs_dir = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)

    joblib.dump(best_bundle, os.path.join(models_dir, "pq_model_bundle.pkl"))
    joblib.dump((all_trained_models, scaler, selector), os.path.join(models_dir, "all_ml_models_bundle.pkl"))
    pd.DataFrame(grid_report).to_csv(os.path.join(outputs_dir, "ml_grid_search_results.csv"), index=False)
    print("En iyi model 'models/pq_model_bundle.pkl' olarak kaydedildi.")
    print("Tüm klasik modeller 'models/all_ml_models_bundle.pkl' olarak kaydedildi.")
    print("Grid Search raporu 'outputs/ml_grid_search_results.csv' olarak kaydedildi.")

    print(f"\n[{best_name}] için Sınıf Bazında Precision, Recall, F1-Score ve Karmaşıklık Matrisi kaydediliyor...")
    cls_report_dict = classification_report(y_test, best_test_preds, output_dict=True)
    cls_report_df = pd.DataFrame(cls_report_dict).transpose().reset_index().rename(columns={"index": "Sınıf / Metrik"})
    cls_report_df.to_csv(os.path.join(outputs_dir, "classification_report_best_ml.csv"), index=False)

    labels_sorted = sorted(np.unique(y_test))
    cm = confusion_matrix(y_test, best_test_preds, labels=labels_sorted)
    cm_df = pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted)
    cm_df.to_csv(os.path.join(outputs_dir, "confusion_matrix_best_ml.csv"))
    print("Sınıf bazlı F1 raporu 'outputs/classification_report_best_ml.csv' ve Karmaşıklık Matrisi 'outputs/confusion_matrix_best_ml.csv' kaydedildi!")

    print("\nSHAP Özellik Önem Dereceleri hesaplanıyor (Optimize Random Forest TreeExplainer, Doğrulama Kümesi n=500)...")
    if best_rf_all_features is None:
        best_rf_all_features = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1).fit(X_train_scaled, y_train)

    explainer = shap.TreeExplainer(best_rf_all_features)
    shap_sample = X_val_scaled[:500]
    shap_values = explainer.shap_values(shap_sample)

    if isinstance(shap_values, list):
        mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
    else:
        mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))

    anova_f_scores = selector.scores_
    anova_p_values = selector.pvalues_
    anova_selected_mask = selector.get_support()

    shap_df = pd.DataFrame({
        "Feature": X.columns,
        "SHAP_Importance": mean_abs_shap,
        "ANOVA_F_Score": anova_f_scores,
        "ANOVA_p_value": anova_p_values,
        "ANOVA_Selected_Top16": anova_selected_mask
    }).sort_values(by="SHAP_Importance", ascending=False).reset_index(drop=True)

    shap_df["SHAP_Rank"] = np.arange(1, len(shap_df) + 1)
    shap_df["ANOVA_Rank"] = shap_df["ANOVA_F_Score"].rank(ascending=False).astype(int)

    shap_top_k = set(shap_df.head(k_best)["Feature"])
    anova_top_k = set(selected_cols)
    overlap_count = len(shap_top_k.intersection(anova_top_k))
    print(f"SHAP İlk {k_best} ile ANOVA İlk {k_best} Özellik Örtüşmesi: {overlap_count}/{k_best} (%{overlap_count/k_best*100:.1f})")

    shap_df.to_csv(os.path.join(outputs_dir, "shap_importance.csv"), index=False)
    print("SHAP ve ANOVA karşılaştırmalı analiz sonuçları 'outputs/shap_importance.csv' dosyasına kaydedildi!")


if __name__ == "__main__":
    train_and_evaluate()