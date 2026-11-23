# Deteksi Cyberbullying & Ujaran Kebencian Bahasa Indonesia (IN286 - NLP)

Repositori proyek mata kuliah **IN286 - Pemrosesan Bahasa Alami** dengan judul:
> **Klasifikasi Teks Ujaran Kebencian dan Perundungan Siber di Media Sosial Bahasa Indonesia Menggunakan N-Gram TF-IDF dan Algoritma Machine Learning**

## Anggota Kelompok
- **Stepanus Sugianto (2372006)** — Pengumpulan data, EDA, & modul preprocessing (*slang normalization*)
- **Calvin Yohanis (2272017)** — Ekstraksi fitur *N-Gram TF-IDF* & model *baseline* (Regresi Logistik)
- **Maverick Rafael Tanadi (2372056)** — Pelatihan model utama (*Linear SVM* & *Random Forest*), analisis eksperimen, & *Hybrid LLM Moderator*
- **Syahrial Achmad (2372059)** — Evaluasi metrik (*Macro F1-Score*), antarmuka *Streamlit*, & penyusunan proposal IMRAD

## Struktur Repositori
```text
.
├── app_streamlit.py                                   # Aplikasi web Streamlit
├── Klasifikasi_Cyberbullying_TFIDF_ML_Kelompok.ipynb  # Notebook eksperimen lengkap
├── requirements.txt                                   # Daftar dependensi Python
├── dataset/
│   ├── re_dataset.csv                                 # 13.169 tweet berlabel (Ibrohim & Budi, 2019)
│   ├── new_kamusalay.csv                              # 15.167 pasangan kata slang-baku
│   ├── abusive.csv                                    # 125 daftar kata kasar
│   ├── citation.bib                                   # Referensi BibTeX dataset
│   └── ...                                            # Dokumentasi gold standard dataset
└── docs/
    └── Proposal Kelompok UTS IN286 - Deteksi Cyberbullying.pdf
```

## Cara Menjalankan Secara Lokal
```bash
pip install -r requirements.txt
streamlit run app_streamlit.py
```

## Sumber & Sitasi Dataset
Dataset berasal dari penelitian:
> Muhammad Okky Ibrohim and Indra Budi. 2019. *Multi-label Hate Speech and Abusive Language Detection in Indonesian Twitter*. In ALW3: 3rd Workshop on Abusive Language Online, 46–57. (Lisensi: CC BY-NC-SA 4.0)
