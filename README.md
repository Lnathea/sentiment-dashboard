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
                          PostgreSQL
```

| Bagian     | Teknologi                                         |
|------------|---------------------------------------------------|
| `ml/`      | Python, pandas, scikit-learn, Hugging Face        |
| `backend/` | FastAPI, SQLAlchemy, Alembic, PostgreSQL, pytest  |
| `frontend/`| Next.js (App Router), TypeScript, Tailwind, Recharts |

## Struktur repo

```
sentiment-dashboard/
├── ml/          # training & evaluasi model
├── backend/     # REST API
├── frontend/    # dashboard web
├── docker-compose.yml
├── .env.example
├── AGENTS.md    # panduan untuk AI agent
└── README.md
```

## Memulai

### Prasyarat

- Python 3.11+
- Node.js 20+
- Docker (untuk PostgreSQL lokal)

### 1. Konfigurasi environment

```bash
cp .env.example .env
# lalu sesuaikan nilainya
```

### 2. Database

```bash
docker compose up -d db
```

### 3. Latih model baseline

```bash
cd ml
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train_baseline.py
```

### 4. Jalankan backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
alembic upgrade head
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
- Dataset dan file model **tidak** disimpan di repo ini. Lihat `ml/README.md` (akan ditambahkan) untuk cara mengunduhnya.

## Hasil evaluasi

_Akan diisi setelah training selesai._

| Model          | Accuracy | Macro-F1 |
|----------------|----------|----------|
| TF-IDF + SVM   | –        | –        |
| IndoBERT       | –        | –        |

## Roadmap

- [ ] **Fase 1**: baseline TF-IDF + SVM dan backend FastAPI
- [ ] **Fase 2**: fine-tune IndoBERT dan integrasi sebagai model kedua
- [ ] **Fase 3**: frontend dashboard dan perbandingan model
- [ ] **Fase 4**: polish, screenshot, dokumentasi
- [ ] Ide lanjutan: scraping ulasan (misalnya Play Store), deploy online

## Lisensi

MIT. Lihat berkas [LICENSE](LICENSE).
