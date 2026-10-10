# Sentiment Dashboard

Analisis sentimen teks berbahasa Indonesia (positif / netral / negatif) dengan dashboard web interaktif. Proyek ini membandingkan dua pendekatan: baseline **TF-IDF + SVM** dan **IndoBERT** hasil fine-tune.

> 🚧 **Status: dalam pengembangan.** Lihat bagian [Roadmap](#roadmap) untuk progres terkini.

## Fitur

- Analisis satu teks beserta confidence score
- Analisis batch lewat upload CSV
- Dashboard: distribusi sentimen, tren per hari, kata paling sering muncul per kelas, tabel hasil dengan filter
- Perbandingan hasil dua model
- Riwayat analisis tersimpan per batch

## Arsitektur

```
Next.js (dashboard)  →  FastAPI (inference)  →  Model (scikit-learn / IndoBERT)
                               ↓
                     SQLite (dev lokal) / PostgreSQL (deploy, nanti)
```

| Bagian     | Teknologi                                         |
|------------|---------------------------------------------------|
| `ml/`      | Python, pandas, scikit-learn, Hugging Face        |
| `backend/` | FastAPI, SQLAlchemy, Alembic, SQLite, pytest      |
| `frontend/`| Next.js (App Router), TypeScript, Tailwind, Recharts |

## Struktur repo

```
sentiment-dashboard/
├── .venv/       # satu virtual env untuk ml/ dan backend/ (tidak di-commit)
├── ml/          # training & evaluasi model
├── backend/     # REST API
├── frontend/    # dashboard web
├── .env.example
├── AGENTS.md    # panduan untuk AI agent
└── README.md
```

## Memulai

Proyek berjalan **tanpa Docker** dan **tanpa PostgreSQL** di dev lokal. Database adalah file SQLite (`backend/sentiment.sqlite3`).

### Prasyarat

- Python 3.11+ (diuji dengan 3.13)
- Node.js 20+ (untuk frontend, Fase 3)

### Urutan menjalankan

**venv → training → `alembic upgrade head` → uvicorn.** Backend memuat model saat start, jadi training harus selesai lebih dulu.

#### 1. Environment dan virtual env (satu `.venv` di root untuk `ml/` dan `backend/`)

Jalankan dari root repo.

Windows PowerShell:

```powershell
Copy-Item .env.example .env
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r ml/requirements.txt -r backend/requirements.txt
```

bash (macOS/Linux/Git Bash):

```bash
cp .env.example .env
python -m venv .venv
source .venv/bin/activate
pip install -r ml/requirements.txt -r backend/requirements.txt
```

#### 2. Unduh data dan latih model baseline (dari root)

```bash
python ml/download_data.py        # SmSA -> ml/data/smsa/
python ml/train_baseline.py       # model + metrics -> ml/artifacts/
```

Hasil: `tfidf_svm.joblib`, `metrics.json`, `confusion_matrix.png` di `ml/artifacts/` (tidak di-commit).

#### 3. Migrasi database dan jalankan backend (dari `backend/`)

```bash
cd backend
alembic upgrade head              # membuat backend/sentiment.sqlite3
uvicorn app.main:app --reload
```

Dokumentasi API otomatis: http://localhost:8000/docs

#### 4. Tes dan lint (dari `backend/`)

```bash
pytest -q                         # backend/tests + ml/tests
ruff check . ../ml && ruff format --check . ../ml
```

#### 5. Frontend (Fase 3, belum ada)

```bash
cd frontend
npm install
npm run dev                       # http://localhost:3000
```

## API (backend, Fase 1)

| Method | Endpoint                  | Fungsi                              |
|--------|---------------------------|-------------------------------------|
| POST   | `/predict`                | Prediksi sentimen satu teks         |
| POST   | `/predict/batch`          | Upload CSV (kolom `text`)           |
| GET    | `/batches`                | Daftar batch analisis               |
| GET    | `/batches/{id}/stats`     | Statistik untuk dashboard           |

## Dataset dan atribusi

- Dataset: **SmSA** (analisis sentimen tingkat dokumen berbahasa Indonesia; label positif / netral / negatif), bagian dari benchmark [IndoNLU](https://github.com/IndoNLP/indonlu) (`dataset/smsa_doc-sentiment-prosa/`). Split resmi dipakai apa adanya: train 11.000, valid 1.260, test 500.
- Dataset dan file model **tidak** disimpan di repo ini; `ml/download_data.py` mengunduhnya. Repo Hugging Face `indonlp/indonlu` hanya berisi skrip pemuat (tanpa berkas data), sehingga data sebenarnya diambil dari GitHub IndoNLU. Detail dan cara verifikasi: [`ml/README.md`](ml/README.md).
- Sitasi yang diminta IndoNLU bila memakai komponennya:

  ```bibtex
  @inproceedings{wilie2020indonlu,
    title={IndoNLU: Benchmark and Resources for Evaluating Indonesian Natural Language Understanding},
    author={Bryan Wilie and Karissa Vincentio and Genta Indra Winata and Samuel Cahyawijaya and X. Li and Zhi Yuan Lim and S. Soleman and R. Mahendra and Pascale Fung and Syafri Bahar and A. Purwarianti},
    booktitle={Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing},
    year={2020}
  }
  ```

- **Lisensi** (sesuai sumber, diperiksa saat dokumentasi ini ditulis): berkas `LICENSE` di repo IndoNLU adalah **Apache License 2.0**, sedangkan badge di README IndoNLU dan metadata kartu dataset Hugging Face `indonlp/indonlu` menyebut **MIT**. Sumbernya sendiri tidak konsisten; keduanya lisensi permisif yang mewajibkan atribusi/pemberitahuan lisensi. Lisensi MIT proyek ini hanya berlaku untuk kode di repo ini, bukan untuk datasetnya. Periksa ulang sumber asli sebelum redistribusi data.

## Hasil evaluasi

Dievaluasi pada **test set SmSA resmi (500 teks)**. Test set tidak dipakai untuk training, tuning, maupun memilih preprocessing; pemilihan model memakai validation set (1.260 teks).

| Model          | Accuracy | Macro-F1 |
|----------------|----------|----------|
| TF-IDF + SVM   | 0,806    | 0,758    |
| IndoBERT       | –        | –        |

Per kelas (TF-IDF + SVM, test):

| Kelas    | Precision | Recall | F1    | Support |
|----------|-----------|--------|-------|---------|
| positive | 0,876     | 0,813  | 0,843 | 208     |
| neutral  | 0,741     | 0,489  | 0,589 | 88      |
| negative | 0,767     | 0,936  | 0,843 | 204     |

Confusion matrix (baris = label asli, kolom = prediksi):

| asli \ prediksi | positive | neutral | negative |
|-----------------|----------|---------|----------|
| positive        | 169      | 10      | 29       |
| neutral         | 16       | 43      | 29       |
| negative        | 8        | 5       | 191      |

**Kelas neutral paling lemah** (recall hanya 48,9%): 29 dari 88 teks netral diprediksi negatif. Kelas netral juga paling sedikit datanya.

## Catatan keputusan teknis

- **SQLite untuk dev, tanpa Docker.** Lebih sederhana untuk satu pengembang. SQLAlchemy + Alembic dipakai tanpa fitur khusus satu database, sehingga PostgreSQL cukup dengan mengganti `DATABASE_URL`.
- **Satu `.venv` di root.** `ml/` dan backend memakai versi `scikit-learn` yang sama (1.9.1) karena file `.joblib` sensitif terhadap versi.
- **`ml/preprocess.py` diimpor, bukan disalin.** Training dan backend memakai fungsi yang sama agar tidak ada selisih preprocessing (training-serving skew).
- **Split resmi SmSA, bukan split acak.** Dipakai apa adanya supaya hasil bisa dibandingkan. Tidak ada teks test yang muncul di train (dicek setelah preprocessing: 0).
- **Pemilihan model memakai macro-F1 di validation set**, bukan accuracy, karena kelas tidak seimbang. Kandidat: LinearSVC (terkalibrasi) dan Logistic Regression dengan C ∈ {0,1; 1; 10}. Terpilih LinearSVC terkalibrasi, C = 10 (macro-F1 valid 0,860).
- **Kalibrasi probabilitas** (`CalibratedClassifierCV`) agar confidence di API/dashboard bermakna; LinearSVC sendiri tidak punya probabilitas.
- **Statistik dashboard:** hitungan memakai agregat SQL yang portabel; tren harian dan kata teratas dihitung di Python karena pemotongan tanggal tidak portabel antar-database.
- **Registry model di backend** (`model_registry`), sehingga IndoBERT nanti ditambahkan sebagai model kedua tanpa menyalin endpoint.
- **Keterbatasan:** test set kecil (500), jadi angka bisa bergeser beberapa poin. Skor test (0,758) lebih rendah daripada validation (0,860), dan kelas netral sulit dikenali baseline.

## Deploy dengan PostgreSQL (nanti)

Kode database ditulis portabel (SQLAlchemy + Alembic, tanpa tipe/fungsi khusus satu database). Untuk deploy cukup pasang driver (`pip install psycopg`) dan ganti `DATABASE_URL`, misalnya `postgresql+psycopg://USER:PASSWORD@HOST:5432/DBNAME`, lalu jalankan `alembic upgrade head`.

## Roadmap

- [ ] **Fase 1**: baseline TF-IDF + SVM dan backend FastAPI
- [ ] **Fase 2**: fine-tune IndoBERT dan integrasi sebagai model kedua
- [ ] **Fase 3**: frontend dashboard dan perbandingan model
- [ ] **Fase 4**: polish, screenshot, dokumentasi
- [ ] Ide lanjutan: scraping ulasan (misalnya Play Store), deploy online

## Lisensi

MIT. Lihat berkas [LICENSE](LICENSE).
