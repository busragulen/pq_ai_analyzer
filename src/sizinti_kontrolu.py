import os
import glob
import hashlib
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
if not os.path.exists(os.path.join(PROJECT_ROOT, "xpqrs_features.csv")):
    PROJECT_ROOT = os.getcwd()

SYNTHETIC_HYBRID_CLASSES = {
    "Flicker_with_Harmonics",
    "Harmonics_with_Notch",
    "Sag_with_Harmonics_and_Transient",
    "Swell_with_Harmonics_and_Transient",
    "Sag_with_Harmonics_and_Oscillatory_Transient",
    "Swell_with_Harmonics_and_Oscillatory_Transient",
    "Flicker_with_Sag_and_Harmonics",
    "Flicker_with_Swell_and_Harmonics",
}


def load_csv_matrix(file_path):
    df = pd.read_csv(file_path, header=None)
    if isinstance(df.iloc[0, 0], str):
        df = pd.read_csv(file_path)
    data = df.values.astype(np.float32)
    if data.shape[0] < data.shape[1] and data.shape[1] == 1000:
        data = data.T
    return data


def check_exact_duplicates(dataset_folder):
    csv_files = sorted(glob.glob(os.path.join(dataset_folder, "*.csv")))
    if not csv_files:
        print(f"[UYARI] '{dataset_folder}' bulunamadı; ham kopya kontrolü atlandı.")
        return pd.DataFrame()

    hash_to_sources = {}
    per_file_rows = []

    for fp in csv_files:
        name = os.path.splitext(os.path.basename(fp))[0]
        mat = load_csv_matrix(fp)
        rounded = np.round(mat, 6)
        local_hashes = set()
        within_dups = 0
        for row_idx, row in enumerate(rounded):
            h = hashlib.sha256(row.tobytes()).hexdigest()
            if h in local_hashes:
                within_dups += 1
            else:
                local_hashes.add(h)
            hash_to_sources.setdefault(h, []).append((name, row_idx))

        per_file_rows.append({
            "Dosya": name,
            "Satir_Sayisi": mat.shape[0],
            "Dosya_Ici_Birebir_Kopya": within_dups,
        })

    cross_pairs = {}
    for _, entries in hash_to_sources.items():
        unique_files = sorted(set(e[0] for e in entries))
        if len(unique_files) > 1:
            for i in range(len(unique_files)):
                for j in range(i + 1, len(unique_files)):
                    pair = (unique_files[i], unique_files[j])
                    cross_pairs[pair] = cross_pairs.get(pair, 0) + 1

    print("=" * 90)
    print("1. HAM VERİ KOPYA (EXACT DUPLICATE) DENETİMİ")
    print("=" * 90)
    dup_df = pd.DataFrame(per_file_rows)
    total_within = int(dup_df["Dosya_Ici_Birebir_Kopya"].sum())
    print(f"Taranan CSV dosyası: {len(csv_files)} | Toplam dosya içi birebir kopya satır: {total_within}")
    if cross_pairs:
        print("Dosyalar arası birebir eşleşen satır çiftleri:")
        for (f1, f2), cnt in sorted(cross_pairs.items(), key=lambda x: -x[1]):
            print(f"   -> {f1} <-> {f2}: {cnt} birebir eşleşme")
    else:
        print("Dosyalar arası birebir (exact hash) çakışan sinyal bulunmadı.")
    return dup_df


def assign_source_row_index(df_features):

    row_ids = np.zeros(len(df_features), dtype=int)
    for code, grp in df_features.groupby("Class_Code", sort=False):
        idx = grp.index.to_numpy()
        row_ids[idx] = np.arange(len(idx)) % 1000
    return row_ids


def evaluate_split(X_tr, y_tr, X_val, y_val, X_te, y_te, split_label, models_dict):
    scaler = StandardScaler()
    X_tr_sc = np.clip(scaler.fit_transform(X_tr), -10.0, 10.0)
    X_val_sc = np.clip(scaler.transform(X_val), -10.0, 10.0)
    X_te_sc = np.clip(scaler.transform(X_te), -10.0, 10.0)

    rows = []
    for name, model in models_dict.items():
        model.fit(X_tr_sc, y_tr)
        val_preds = model.predict(X_val_sc)
        te_preds = model.predict(X_te_sc)

        val_acc = accuracy_score(y_val, val_preds) * 100.0
        val_f1 = f1_score(y_val, val_preds, average="macro") * 100.0
        te_acc = accuracy_score(y_te, te_preds) * 100.0
        te_f1 = f1_score(y_te, te_preds, average="macro") * 100.0

        rows.append({
            "Bolme_Stratejisi": split_label,
            "Model": name,
            "Val_Acc": val_acc,
            "Val_Macro_F1": val_f1,
            "Test_Acc": te_acc,
            "Test_Macro_F1": te_f1,
            "Egitim_Ornek": len(y_tr),
            "Val_Ornek": len(y_val),
            "Test_Ornek": len(y_te),
        })
        print(
            f"[{split_label:28s} | {name:16s}] "
            f"Val Acc: %{val_acc:.2f} (F1: %{val_f1:.2f}) | "
            f"Test Acc: %{te_acc:.2f} (F1: %{te_f1:.2f})"
        )
    return rows


def run_leakage_audit():
    features_path = os.path.join(PROJECT_ROOT, "xpqrs_features.csv")
    dataset_folder = os.path.join(PROJECT_ROOT, "data", "XPQRS")
    outputs_dir = os.path.join(PROJECT_ROOT, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    dup_df = check_exact_duplicates(dataset_folder)
    if not dup_df.empty:
        dup_df.to_csv(os.path.join(outputs_dir, "sizinti_kopya_ozeti.csv"), index=False)

    if not os.path.exists(features_path):
        raise FileNotFoundError(f"'{features_path}' bulunamadı!")

    df = pd.read_csv(features_path)
    X = df.drop(columns=["Class_Code", "Label"]).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df["Label"].values
    row_groups = assign_source_row_index(df)

    print("\n" + "=" * 90)
    print("2. SATIR İNDEKSİ SIZINTI DENETİMİ (STANDART STRATIFIED vs GRUP BAZLI BÖLME)")
    print("=" * 90)
    print(f"Toplam örnek: {len(df)} | Benzersiz sınıf: {len(np.unique(y))} | Benzersiz kaynak satır grubu: {len(np.unique(row_groups))}")

    def build_models():
        return {
            "Random Forest": RandomForestClassifier(
                n_estimators=200, max_depth=20, min_samples_split=5, criterion="entropy", random_state=42, n_jobs=-1
            ),
            "RBF SVM (DVM)": SVC(kernel="rbf", C=100, gamma="scale", random_state=42),
            "Kübik SVM (DVM)": SVC(kernel="poly", degree=3, C=50, coef0=1.0, gamma="auto", random_state=42),
        }

    X_tr_s, X_tmp_s, y_tr_s, y_tmp_s = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val_s, X_te_s, y_val_s, y_te_s = train_test_split(
        X_tmp_s, y_tmp_s, test_size=0.50, random_state=42, stratify=y_tmp_s
    )

    rng = np.random.default_rng(42)
    unique_groups = np.unique(row_groups)
    shuffled_groups = rng.permutation(unique_groups)
    n_tr_g = int(round(0.70 * len(shuffled_groups)))
    n_val_g = int(round(0.15 * len(shuffled_groups)))

    tr_groups = set(shuffled_groups[:n_tr_g])
    val_groups = set(shuffled_groups[n_tr_g : n_tr_g + n_val_g])
    te_groups = set(shuffled_groups[n_tr_g + n_val_g :])

    assert len(tr_groups.intersection(val_groups)) == 0
    assert len(tr_groups.intersection(te_groups)) == 0
    assert len(val_groups.intersection(te_groups)) == 0

    mask_tr = np.isin(row_groups, list(tr_groups))
    mask_val = np.isin(row_groups, list(val_groups))
    mask_te = np.isin(row_groups, list(te_groups))

    results = []
    results.extend(
        evaluate_split(X_tr_s, y_tr_s, X_val_s, y_val_s, X_te_s, y_te_s, "Standart_Stratified_Split", build_models())
    )
    results.extend(
        evaluate_split(
            X.iloc[mask_tr], y[mask_tr],
            X.iloc[mask_val], y[mask_val],
            X.iloc[mask_te], y[mask_te],
            "Grup_Bazli_Sizintisiz_Split",
            build_models(),
        )
    )

    res_df = pd.DataFrame(results)

    print("\n" + "-" * 90)
    print("SIZINTI ETKİSİ ÖZETİ (Standart Split - Grup Bazlı Sızıntısız Split):")
    print("-" * 90)
    summary_rows = []
    for model_name in res_df["Model"].unique():
        r_std = res_df[(res_df["Model"] == model_name) & (res_df["Bolme_Stratejisi"] == "Standart_Stratified_Split")].iloc[0]
        r_grp = res_df[(res_df["Model"] == model_name) & (res_df["Bolme_Stratejisi"] == "Grup_Bazli_Sizintisiz_Split")].iloc[0]
        diff_acc = r_std["Test_Acc"] - r_grp["Test_Acc"]
        diff_f1 = r_std["Test_Macro_F1"] - r_grp["Test_Macro_F1"]
        print(
            f"   -> {model_name:16s} | Test Acc Farkı: {diff_acc:+.2f} puan "
            f"(%{r_std['Test_Acc']:.2f} vs %{r_grp['Test_Acc']:.2f}) | "
            f"Test Makro F1 Farkı: {diff_f1:+.2f} puan (%{r_std['Test_Macro_F1']:.2f} vs %{r_grp['Test_Macro_F1']:.2f})"
        )
        summary_rows.append({
            "Model": model_name,
            "Standart_Test_Acc": r_std["Test_Acc"],
            "Grup_Test_Acc": r_grp["Test_Acc"],
            "Acc_Fark_Puan": diff_acc,
            "Standart_Test_Macro_F1": r_std["Test_Macro_F1"],
            "Grup_Test_Macro_F1": r_grp["Test_Macro_F1"],
            "Macro_F1_Fark_Puan": diff_f1,
        })

    out_path = os.path.join(outputs_dir, "sizinti_kontrolu_sonuclari.csv")
    res_df.to_csv(out_path, index=False)
    pd.DataFrame(summary_rows).to_csv(os.path.join(outputs_dir, "sizinti_kontrolu_ozet.csv"), index=False)
    print(f"\nSonuçlar '{out_path}' ve 'outputs/sizinti_kontrolu_ozet.csv' olarak kaydedildi.")


if __name__ == "__main__":
    run_leakage_audit()