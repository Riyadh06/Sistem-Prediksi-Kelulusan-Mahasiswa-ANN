# 🎓 Prediksi Kelulusan Mahasiswa Menggunakan Artificial Neural Network (ANN) Backpropagation Berbasis Web

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%7C%20Keras-orange.svg)](https://tensorflow.org/)
[![UI Library](https://img.shields.io/badge/Web%20UI-Streamlit-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Academic Project](https://img.shields.io/badge/Course-Artificial%20Intelligence%20(S1)-green.svg)](#)

Aplikasi berbasis web dan eksperimen komparatif untuk memprediksi status kelulusan mahasiswa (*Tepat Waktu* vs *Tidak Tepat Waktu*) menggunakan algoritma **Artificial Neural Network (Multilayer Perceptron)** dengan metode optimasi **Backpropagation**.

---

## 📌 Daftar Isi
- [Latar Belakang & Deskripsi](#-latar-belakang--deskripsi)
- [Fitur Utama Aplikasi](#-fitur-utama-aplikasi)
- [Arsitektur Model ANN](#-arsitektur-model-ann)
- [Hasil Eksperimen & Evaluasi](#-hasil-eksperimen--evaluasi)
- [Struktur Direktori](#-struktur-direktori)
- [Panduan Instalasi & Penggunaan](#-panduan-instalasi--penggunaan)
- [Teknologi yang Digunakan](#-teknologi-yang-digunakan)
- [Catatan Akademik & Batasan Model](#-catatan-akademik--batasan-model)

---

## 📖 Latar Belakang & Deskripsi

Kelulusan mahasiswa tepat waktu merupakan indikator kunci keberhasilan akademik dan akreditasi perguruan tinggi. Proyek ini mengembangkan model kecerdasan buatan (*Machine Learning / Deep Learning*) untuk melakukan klasifikasi biner terhadap status kelulusan mahasiswa:
1. **Lulus Tepat Waktu (Kelas 1)**
2. **Tidak Tepat Waktu (Kelas 0)**

Model dilatih menggunakan dataset akademik sintetis dengan alur kerja bebas dari kebocoran data (*zero data leakage*), kemudian diintegrasikan ke antarmuka web interaktif menggunakan **Streamlit** untuk mempermudah simulasi dan pengambilan keputusan oleh dosen wali maupun mahasiswa.

---

## ✨ Fitur Utama Aplikasi

1. **Input Interaktif (Formulir Multi-Kolom):**
   - Slider dinamis untuk IPK, Total SKS, Persentase Kehadiran, Jam Belajar, dan Jumlah Organisasi.
   - Pilihan dropdown untuk Kategori Penghasilan Orang Tua dan Kualitas Internet.
2. **Preset Uji Cepat (Sidebar):**
   - *Mahasiswa Teladan (Potensi Tepat Waktu)*: Memuat otomatis parameter mahasiswa berprestasi.
   - *Mahasiswa Berisiko (Potensi Molor)*: Memuat profil mahasiswa yang membutuhkan intervensi akademik.
   - *Custom*: Pengisian manual sesuai kebutuhan.
3. **Visualisasi Output Real-Time:**
   - Kartu status kelulusan dengan gradien warna responsif (Hijau untuk Tepat Waktu, Merah untuk Berisiko).
   - *Progress bar* tingkat keyakinan probabilitas model.
   - Rekomendasi akademik terarah dan solutif.
4. **Panel Diagnostik Teknis:**
   - Menampilkan vektor fitur mentah, dimensi vektor 11 dimensi ternormalisasi, dan nilai probabilitas keluaran aktivasi Sigmoid.

---

## 🧠 Arsitektur Model ANN

```
Input Layer (11 Dimensi Hasil Preprocessing)
        │
        ▼
Hidden Layer 1 (16 Neuron, Aktivasi ReLU, Inisialisasi He-Normal)
        │
        ▼
Hidden Layer 2 (8 Neuron, Aktivasi ReLU, Inisialisasi He-Normal)
        │
        ▼
Output Layer (1 Neuron, Aktivasi Sigmoid) --> Probabilitas Biner [0, 1]
```

- **Optimizer:** Adam (*learning rate* = 0.001)
- **Fungsi Kerugian (Loss):** Binary Crossentropy
- **Regularisasi:** EarlyStopping (`patience=15`, `restore_best_weights=True`)
- **Pipeline Preprocessing:**
  - Fitur numerik: `SimpleImputer(median)` $\rightarrow$ `StandardScaler()`
  - Fitur kategorikal: `SimpleImputer(most_frequent)` $\rightarrow$ `OneHotEncoder(handle_unknown='ignore')`
  - Transformer diisolasi hanya pada data latih (*Zero Data Leakage*).

---

## 📊 Hasil Eksperimen & Evaluasi

Pengujian dilakukan pada data uji (*testing set*) sebesar **121 sampel** (20% dari 603 data bersih) dengan pemisahan terstratifikasi (*Stratified Split*):

| Metrik Evaluasi | Skenario 1 (Model Lengkap: Ada IPK & SKS) | Skenario 2 (Early Warning: Tanpa IPK & SKS) |
|---|:---:|:---:|
| **Akurasi (Accuracy)** | **95.04%** | 76.86% *(Accuracy Paradox)* |
| **Presisi (Precision)** | **0.9677** | 0.7686 |
| **Recall (Sensitivitas)** | **0.9677** | 1.0000 |
| **F1-Score** | **0.9677** | 0.8692 |
| **ROC-AUC Score** | **0.9942** | **0.3283** |
| *F1-Score Kelas "Tidak Tepat Waktu"* | **0.8929** (Precision 89%, Recall 89%) | **0.0000** (Model Collapse) |

> 📌 **Temuan Ilmiah Penting:** Pada Skenario 2 terjadi fenomena ***Accuracy Paradox*** di mana akurasi tampak 76.86% namun model memprediksi 100% data sebagai kelas mayoritas (Recall kelas minoritas = 0%). Hal ini membuktikan bahwa metrik akurasi tidak dapat diandalkan sendirian pada *imbalanced data*, serta membuktikan bahwa **IPK dan Total SKS merupakan faktor penentu mutlak** pada kelulusan mahasiswa.

---

## 📁 Struktur Direktori

```text
├── model_prediksi_kelulusan_web/      # Direktori model & artefak deployment
│   ├── model_ann_kelulusan.keras     # Arsitektur & bobot jaringan ANN
│   ├── pipeline_preprocessor.joblib  # Pipeline Scikit-Learn (Scaler, OHE, Imputer)
│   └── metadata_model.json           # Skema fitur masukan & metrik evaluasi
├── Prediksi_Kelulusan_ANN_Backpropagation.ipynb # Notebook lengkap eksperimen (Tahap 1 - 11)
├── student_data_raw.xlsx             # Dataset mentah sintetis
├── app.py                            # Kode aplikasi antarmuka web (Streamlit)
├── run_app.bat                       # Script runner otomatis Windows
├── requirements.txt                  # Daftar pustaka Python yang dibutuhkan
├── .gitignore                        # File pengabaian Git
└── README.md                         # Dokumentasi proyek
```

---

## 🚀 Panduan Instalasi & Penggunaan

### 1. Kloning Repositori
```bash
git clone https://github.com/<username-anda>/prediksi-kelulusan-ann.git
cd prediksi-kelulusan-ann
```

### 2. Buat & Aktifkan Virtual Environment (Opsional namun Disarankan)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Pasang Dependensi
```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi Web Streamlit
* **Cara Cepat (Windows):** Klik ganda pada file `run_app.bat`.
* **Cara Terminal:**
  ```bash
  streamlit run app.py
  ```
Buka browser pada alamat: **`http://localhost:8501`**.

---

## 🛠️ Teknologi yang Digunakan

- **Bahasa Pemrograman:** Python 3.10+
- **Deep Learning / Jaringan Syaraf:** TensorFlow & Keras
- **Machine Learning & Preprocessing:** Scikit-Learn, Joblib
- **Manipulasi Data:** Pandas, NumPy, OpenPyXL
- **Visualisasi Data:** Matplotlib, Seaborn
- **Pengembangan Web:** Streamlit

---

## ⚠️ Catatan Akademik & Batasan Model

1. **Dataset Sintetis:** Dataset yang digunakan adalah data simulasi akademik untuk keperluan mata kuliah Artificial Intelligence, bukan data riil yang diambil dari Pangkalan Data Pendidikan Tinggi (PDDikti).
2. **Kompensasi Bobot (Weighted Sum):** ANN bekerja dengan menjumlahkan bobot fitur, bukan sistem berbasis aturan (*If-Else*). Pada kondisi ekstrem di mana IPK dan SKS bernilai maksimal namun kehadiran 0%, model tetap memprediksi lulus karena sinyal positif akademik mengompensasi penalti kehadiran. Di dunia nyata, sistem ini idealnya digabungkan dengan *rule-based filtering* (Kehadiran $\ge$ 75% sebagai prasyarat mutlak).

---

## 👨‍💻 Kontributor
- **Nama Mahasiswa:** [Nama Anda]
- **NIM:** [NIM Anda]
- **Program Studi:** Teknik Informatika / Ilmu Komputer (S1)
- **Mata Kuliah:** Artificial Intelligence (Semester 5)
