import os
import re
import zipfile
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

st.set_page_config(
    page_title="Deteksi Cyberbullying & Hate Speech Bahasa Indonesia",
    layout="wide"
)

st.title("Sistem Deteksi Kalimat Perundungan Siber & Ujaran Kebencian Bahasa Indonesia")
st.markdown("""
**Judul Proyek:** *Klasifikasi Teks Ujaran Kebencian dan Perundungan Siber di Media Sosial Bahasa Indonesia Menggunakan N-Gram TF-IDF dan Algoritma Machine Learning*  
**Tim Pengembang (IN286 - Pemrosesan Bahasa Alami):**
- **Stepanus Sugianto (2372006)**: Pengumpulan Data, EDA, & Modul Preprocessing (*Slang Normalization*)
- **Calvin Yohanis (2272017)**: Ekstraksi Fitur *N-Gram TF-IDF* & Model Baseline (*Regresi Logistik*)
- **Maverick Rafael Tanadi (2372056)**: Pelatihan Model Utama (*Linear SVM & Random Forest*), Analisis Eksperimen, & *Hybrid LLM Moderator*
- **Syahrial Achmad (2372059)**: Evaluasi Metrik (*F1-Score*), Pembuatan UI Streamlit, & Penyusunan Proposal IMRAD
""")

def basic_clean(text):
    text = str(text).lower()
    text = re.sub(r'\\x[a-f0-9]{2}', ' ', text)
    text = re.sub(r'\b(user|rt|url)\b', ' ', text)
    text = re.sub(r'http\S+|www\S+', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_slang(text, slang_dict):
    tokens = text.split()
    return ' '.join([slang_dict.get(tok, tok) for tok in tokens])

@st.cache_resource
def load_and_train_pipeline():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidate_dirs = [
        os.path.join(base_dir, 'dataset'),
        os.path.join(base_dir, 'DATASET'),
        os.path.join(base_dir, '..', 'dataset'),
        os.path.join(base_dir, '..', 'DATASET'),
        os.path.join(base_dir, 'id-multi-label-hate-speech-and-abusive-language-detection-master'),
        base_dir,
        '.'
    ]

    def find_file(filename):
        for d in candidate_dirs:
            p = os.path.join(d, filename)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"File '{filename}' tidak ditemukan di folder dataset/.")

    data_path = find_file('re_dataset.csv')
    kamus_path = find_file('new_kamusalay.csv')
    abusive_path = find_file('abusive.csv')

    df = pd.read_csv(data_path, encoding='latin-1')
    kamus_df = pd.read_csv(kamus_path, encoding='latin-1', header=None, names=['slang', 'baku'])
    abusive_df = pd.read_csv(abusive_path, encoding='latin-1')

    kamus_alay = dict(zip(kamus_df['slang'].astype(str).str.strip(), kamus_df['baku'].astype(str).str.strip()))
    abusive_set = set(abusive_df.iloc[:, 0].astype(str).str.lower().str.strip())

    df['Label'] = ((df['HS'] == 1) | (df['Abusive'] == 1)).astype(int)
    df['Clean_No_Norm'] = df['Tweet'].apply(basic_clean)
    df['Clean_With_Norm'] = df['Clean_No_Norm'].apply(lambda x: normalize_slang(x, kamus_alay))

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df['Clean_No_Norm'], df['Label'], test_size=0.20, random_state=42, stratify=df['Label']
    )
    X_train_norm = df.loc[X_train_raw.index, 'Clean_With_Norm']
    X_test_norm = df.loc[X_test_raw.index, 'Clean_With_Norm']

    tfidf_no = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=15000, min_df=2)
    X_tr_no = tfidf_no.fit_transform(X_train_raw)
    X_te_no = tfidf_no.transform(X_test_raw)

    tfidf_norm = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=15000, min_df=2)
    X_tr_norm = tfidf_norm.fit_transform(X_train_norm)
    X_te_norm = tfidf_norm.transform(X_test_norm)

    lr_no = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(X_tr_no, y_train)
    lr_norm = LogisticRegression(C=1.0, max_iter=1000, random_state=42).fit(X_tr_norm, y_train)

    svm_no = LinearSVC(C=1.0, random_state=42).fit(X_tr_no, y_train)
    svm_norm = LinearSVC(C=1.0, random_state=42).fit(X_tr_norm, y_train)

    rf_no = RandomForestClassifier(n_estimators=150, max_depth=40, n_jobs=-1, random_state=42).fit(X_tr_no, y_train)
    rf_norm = RandomForestClassifier(n_estimators=150, max_depth=40, n_jobs=-1, random_state=42).fit(X_tr_norm, y_train)

    def eval_row(name, skema, y_t, y_p):
        return {
            "Model": name,
            "Skema Preprocessing": skema,
            "Accuracy (%)": round(accuracy_score(y_t, y_p) * 100, 2),
            "Precision Macro (%)": round(precision_score(y_t, y_p, average='macro') * 100, 2),
            "Recall Macro (%)": round(recall_score(y_t, y_p, average='macro') * 100, 2),
            "F1-Score Macro (%)": round(f1_score(y_t, y_p, average='macro') * 100, 2),
        }

    metrics_df = pd.DataFrame([
        eval_row("Baseline: Regresi Logistik", "Tanpa Normalisasi Slang", y_test, lr_no.predict(X_te_no)),
        eval_row("Baseline: Regresi Logistik", "Dengan Normalisasi Slang", y_test, lr_norm.predict(X_te_norm)),
        eval_row("Model Utama 1: Linear SVM", "Tanpa Normalisasi Slang", y_test, svm_no.predict(X_te_no)),
        eval_row("Model Utama 1: Linear SVM", "Dengan Normalisasi Slang", y_test, svm_norm.predict(X_te_norm)),
        eval_row("Model Utama 2: Random Forest", "Tanpa Normalisasi Slang", y_test, rf_no.predict(X_te_no)),
        eval_row("Model Utama 2: Random Forest", "Dengan Normalisasi Slang", y_test, rf_norm.predict(X_te_norm)),
    ])

    return {
        "df": df,
        "kamus_alay": kamus_alay,
        "abusive_set": abusive_set,
        "tfidf_norm": tfidf_norm,
        "lr_norm": lr_norm,
        "svm_norm": svm_norm,
        "rf_norm": rf_norm,
        "metrics_df": metrics_df
    }

state = load_and_train_pipeline()

tab1, tab2, tab3 = st.tabs([
    "1. Deteksi Komentar Real-Time (ML + Hybrid LLM)",
    "2. Eksplorasi Data & Normalisasi Slang (Stepanus)",
    "3. Evaluasi Eksperimen Model (Calvin, Maverick, Syahrial)"
])

with tab1:
    st.subheader("Uji Coba Deteksi Kalimat Perundungan Siber Secara Real-Time")
    user_input = st.text_area(
        "Masukkan komentar atau tweet bahasa Indonesia (boleh menggunakan kata gaul/slang/singkatan):",
        value="gw ga ngerti knp org kaya lu masih dikasih panggung di kampus"
    )
    model_choice = st.selectbox(
        "Pilih Algoritma Klasifikasi:",
        ["Baseline: Regresi Logistik (Logistic Regression)", "Model Utama 1: Linear SVM", "Model Utama 2: Random Forest"]
    )

    if st.button("Analisis Teks"):
        cleaned = basic_clean(user_input)
        normalized = normalize_slang(cleaned, state["kamus_alay"])
        vec = state["tfidf_norm"].transform([normalized])

        prob_lr = float(state["lr_norm"].predict_proba(vec)[0][1])
        margin_svm = float(state["svm_norm"].decision_function(vec)[0])
        prob_svm_scaled = 1 / (1 + np.exp(-margin_svm))
        prob_rf = float(state["rf_norm"].predict_proba(vec)[0][1])

        if "Regresi Logistik" in model_choice:
            score = prob_lr
            pred = 1 if prob_lr >= 0.5 else 0
        elif "Linear SVM" in model_choice:
            score = prob_svm_scaled
            pred = 1 if margin_svm > 0 else 0
        else:
            score = prob_rf
            pred = 1 if prob_rf >= 0.5 else 0

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Tahap 1: Pembersihan Dasar (Tanpa Normalisasi Slang)**")
            st.code(cleaned)
        with col_b:
            st.markdown("**Tahap 2: Hasil Normalisasi Slang (`new_kamusalay.csv`)**")
            st.code(normalized)

        if pred == 1:
            st.error(f"Hasil Klasifikasi: TERINDIKASI PERUNDUNGAN SIBER / UJARAN KEBENCIAN (Skor Keyakinan: {score*100:.2f}%)")
        else:
            st.success(f"Hasil Klasifikasi: KALIMAT NETRAL / AMAN (Skor Risiko: {score*100:.2f}%)")

        st.markdown("**Ringkasan Skor Lintas Model & Analisis Hybrid LLM Moderator:**")
        st.json({
            "Probabilitas_Baseline_Regresi_Logistik": round(prob_lr, 4),
            "Skor_Margin_Linear_SVM": round(margin_svm, 4),
            "Probabilitas_Random_Forest": round(prob_rf, 4),
            "Rekomendasi_Moderasi_LLM": "Tahan komentar untuk peninjauan / sembunyikan otomatis" if pred == 1 else "Izinkan tampil di publik"
        })

with tab2:
    st.subheader("Statistik Dataset & Dampak Normalisasi Kata Slang")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Tweet (re_dataset.csv)", f"{len(state['df']):,}")
    c2.metric("Kamus Slang (new_kamusalay.csv)", f"{len(state['kamus_alay']):,} entri")
    c3.metric("Reduksi Kosakata Unik (OOV)", "29.01% (-7.907 token)")
    st.dataframe(state["df"][["Tweet", "Clean_No_Norm", "Clean_With_Norm", "HS", "Abusive", "Label"]].head(15))

with tab3:
    st.subheader("Tabel Perbandingan Performa Model (Tanpa vs. Dengan Normalisasi Slang)")
    st.dataframe(state["metrics_df"], use_container_width=True)
