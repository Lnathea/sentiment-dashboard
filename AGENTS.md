# AGENTS.md

Panduan untuk AI agent (Claude Code, Cursor, Copilot Agent, dll.) yang bekerja di repo ini.
Baca file ini sebelum mengubah apa pun.

## Ringkasan proyek

**Sentiment Dashboard**: aplikasi analisis sentimen teks berbahasa Indonesia (positif / netral / negatif) dengan dashboard web. Dua model dibandingkan: baseline TF-IDF + SVM dan IndoBERT hasil fine-tune.

Pemilik repo adalah mahasiswa yang ingin memahami hasilnya. Jelaskan keputusan penting secara singkat dan jangan menyembunyikan kegagalan.

## Struktur repo

```
sentiment-dashboard/
├── ml/              # training & evaluasi model (Python)
│   ├── data/        # dataset (TIDAK di-commit)
│   └── artifacts/   # model hasil training (TIDAK di-commit)
├── backend/         # FastAPI + SQLAlchemy + Alembic + PostgreSQL
├── frontend/        # Next.js (App Router) + Tailwind + Recharts
├── docker-compose.yml   # PostgreSQL untuk dev lokal
├── .env.example
├── AGENTS.md
└── README.md
```

## Stack

- **ML**: Python 3.11+, pandas, scikit-learn, joblib. IndoBERT lewat Hugging Face Transformers (training dijalankan di Google Colab, bukan lokal).
- **Backend**: FastAPI, Pydantic v2, SQLAlchemy 2.x, Alembic, PostgreSQL, pytest.
- **Frontend**: Next.js (App Router, TypeScript), Tailwind CSS, Recharts.

## Perintah penting

Sesuaikan jika struktur berubah, dan perbarui bagian ini di commit yang sama.

```bash
# Database lokal
docker compose up -d db

# ML
cd ml && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train_baseline.py          # train + evaluasi baseline

# Backend
cd backend && python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --reload     # http://localhost:8000/docs
pytest -q                          # jalankan tes

# Frontend
cd frontend
npm install
npm run dev                        # http://localhost:3000
npm run lint && npm run build      # cek sebelum commit
```

## Aturan kerja

1. **Kerjakan satu fase sekali.** Jangan menambah fitur di luar lingkup fase yang diminta. Setelah selesai, berhenti dan laporkan.
2. **Jangan mengarang.** Jangan membuat data, angka evaluasi, atau hasil tes palsu. Jika dataset gagal diunduh atau tes gagal, katakan apa adanya.
3. **Hindari kebocoran data.** Split train/val/test dibuat sekali dengan random seed tetap. Data test tidak boleh dipakai untuk training, tuning, atau memilih preprocessing.
4. **Lapor jujur di akhir tiap fase**: apa yang jalan, apa yang belum, angka evaluasi asli (accuracy, macro-F1, confusion matrix), dan cara menjalankannya.
5. **Tulis tes** untuk endpoint backend dan logika preprocessing. Jalankan tes sebelum menyatakan selesai.
6. **Perubahan kecil dan fokus.** Satu commit untuk satu hal, dengan pesan commit gaya Conventional Commits (`feat:`, `fix:`, `chore:`, `docs:`, `test:`).
7. **Tanya dulu** sebelum menambah dependensi besar, mengubah skema database secara destruktif, atau mengubah struktur folder.

## Yang TIDAK boleh di-commit

- File `.env` atau secret apa pun (API key, password database, token)
- Dataset mentah dan file model: `*.joblib`, `*.pt`, `*.bin`, `*.safetensors`, isi `ml/data/` dan `ml/artifacts/`
- `node_modules/`, `.venv/`, `.next/`, `__pycache__/`

Semua nilai konfigurasi dibaca dari environment variable. Tambahkan variabel baru ke `.env.example` (tanpa nilai asli).

## Konvensi kode

- **Python**: type hints di fungsi publik, format dengan `ruff format`, lint dengan `ruff check`. Nama berkas dan fungsi `snake_case`.
- **TypeScript**: strict mode, komponen fungsional, hindari `any`.
- **API**: respons JSON konsisten, error memakai kode HTTP yang sesuai dan pesan yang jelas. Validasi semua input (ukuran file upload, kolom CSV, panjang teks).
- **Label sentimen**: selalu `positive`, `neutral`, `negative` (huruf kecil, bahasa Inggris) di API dan database. Terjemahkan hanya di tampilan frontend.
- **Model baru** ditambahkan lewat registry/interface model di backend, bukan dengan menyalin endpoint.

## Definisi "selesai" untuk sebuah fase

- Kode berjalan dari awal dengan perintah di README
- Tes lulus (`pytest -q`, dan `npm run lint && npm run build` untuk frontend)
- README diperbarui jika ada perubahan cara menjalankan
- Laporan akhir ditulis sesuai aturan nomor 4

## Rencana fase

1. **Fase 1**: `ml/` (baseline) dan `backend/` (FastAPI + database)
2. **Fase 2**: notebook Colab fine-tune IndoBERT dan integrasi sebagai model kedua
3. **Fase 3**: frontend (input teks, upload CSV, dashboard, perbandingan model)
4. **Fase 4**: polish, README dengan screenshot, catatan rencana scraping
