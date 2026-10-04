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

Proyek berjalan **tanpa Docker**. Database dev lokal adalah file SQLite.

### Prasyarat

- Python 3.11+ (diuji dengan 3.13)
- Node.js 20+ (untuk frontend, Fase 3)

### 1. Konfigurasi environment

```bash
cp .env.example .env              # Windows PowerShell: Copy-Item .env.example .env
# lalu sesuaikan nilainya
```

### 2. Virtual environment (satu untuk ml/ dan backend/)

Jalankan dari root repo:

```bash
python -m venv .venv
source .venv/bin/activate         # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r ml/requirements.txt -r backend/requirements.txt
```

### 3. Unduh data dan latih model baseline

```bash
python ml/download_data.py        # SmSA -> ml/data/smsa/
python ml/train_baseline.py       # model + metrics -> ml/artifacts/
```

Detail ada di [`ml/README.md`](ml/README.md).

### 4. Jalankan backend

```bash
cd backend
alembic upgrade head              # membuat backend/sentiment.sqlite3
uvicorn app.main:app --reload
```

Dokumentasi API otomatis: http://localhost:8000/docs

### 5. Jalankan frontend

```bash
cd frontend
npm install
npm run dev
```

Buka http://localhost:3000

> Perintah di atas akan berfungsi penuh setelah fase terkait selesai dikerjakan.

## API (rencana)

| Method | Endpoint                  | Fungsi                              |
|--------|---------------------------|-------------------------------------|
| POST   | `/predict`                | Prediksi sentimen satu teks         |
| POST   | `/predict/batch`          | Upload CSV (kolom `text`)           |
| GET    | `/batches`                | Daftar batch analisis               |
| GET    | `/batches/{id}/stats`     | Statistik untuk dashboard           |

## Dataset dan model

- Dataset: SmSA dari [IndoNLU](https://github.com/IndoNLP/indonlu) (positif / netral / negatif)
- Dataset dan file model **tidak** disimpan di repo ini. Lihat `ml/README.md` untuk cara mengunduhnya.

## Hasil evaluasi

_Akan diisi setelah training selesai._

| Model          | Accuracy | Macro-F1 |
|----------------|----------|----------|
| TF-IDF + SVM   | –        | –        |
| IndoBERT       | –        | –        |

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
