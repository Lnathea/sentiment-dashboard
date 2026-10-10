# ml/: data, training, evaluasi

Folder ini berisi pipeline baseline **TF-IDF + SVM** untuk analisis sentimen SmSA.
Semua perintah dijalankan dari **root repo** dengan `.venv` aktif (lihat [README utama](../README.md)).

| File | Fungsi |
|------|--------|
| `download_data.py` | Mengunduh SmSA ke `ml/data/smsa/` dan mencatat asal-usulnya |
| `preprocess.py` | Normalisasi teks. **Dipakai training dan backend** (diimpor, jangan disalin) |
| `train_baseline.py` | Memilih model di validation, mengevaluasi di test, menyimpan artefak |
| `finetune_indobert.py` | Fine-tune IndoBERT (dijalankan di Colab GPU), lihat [bagian ini](#fine-tune-indobert-di-colab) |
| `requirements-indobert.txt` | Dependensi untuk Colab saja (jangan dipasang di `.venv` lokal) |
| `notebooks/colab_finetune_indobert.ipynb` | Notebook pemanggil untuk Colab |
| `tests/` | Tes untuk preprocessing |

`ml/data/` dan `ml/artifacts/` **tidak di-commit** (lihat `.gitignore`).

## 1. Mengunduh dataset

```bash
python ml/download_data.py            # lewati jika data sudah ada
python ml/download_data.py --force    # unduh ulang
```

### Sumber

Dataset: **SmSA** dari [IndoNLU](https://github.com/IndoNLP/indonlu), split resmi apa adanya. Skrip mencoba sumber berurutan dan **tidak pernah membuat data sendiri**:

1. **Hugging Face** `indonlp/indonlu` lewat HTTP biasa. Repo itu hanya berisi skrip pemuat (`indonlu.py`, butuh library `datasets`) tanpa berkas data mentah, jadi langkah ini gagal.
2. **GitHub IndoNLU** (dipakai, ini yang berhasil):
   `https://raw.githubusercontent.com/IndoNLP/indonlu/master/dataset/smsa_doc-sentiment-prosa/{train,valid,test}_preprocess.tsv`

   `test_preprocess.tsv` adalah file test **berlabel** (bukan versi `masked_label`).

Jika semua sumber gagal, skrip berhenti dengan kode keluar 1 dan tidak menulis apa pun.

Hasil: `ml/data/smsa/{train,valid,test}.tsv` (tanpa header, format `teks<TAB>label`) dan `source.json`.

### Cara verifikasi

Skrip sudah memvalidasi saat mengunduh: UTF-8 valid, ada pemisah TAB, dan label hanya `positive` / `neutral` / `negative`. Anda bisa memeriksa ulang:

1. **Buka `ml/data/smsa/source.json`.** Isinya sumber yang dipakai (`source`), waktu unduh, riwayat percobaan tiap sumber (termasuk alasan gagal), serta jumlah baris, jumlah per label, dan SHA-256 tiap berkas.
2. **Bandingkan jumlah baris dan label.** Pada unduhan yang dipakai proyek ini:

   | Split | Baris | positive | neutral | negative |
   |-------|-------|----------|---------|----------|
   | train | 11.000 | 6.416 | 1.148 | 3.436 |
   | valid | 1.260 | 735 | 131 | 394 |
   | test  | 500 | 208 | 88 | 204 |

3. **Hitung ulang hash berkas** dan cocokkan dengan `source.json`:

   ```powershell
   # Windows PowerShell
   Get-FileHash ml\data\smsa\train.tsv -Algorithm SHA256
   ```

   ```bash
   # bash
   sha256sum ml/data/smsa/*.tsv
   ```

   Hash unduhan proyek ini (awalan): train `50f38cee…`, valid `6ab41ddc…`, test `4e8016da…`.

> Jika hash Anda berbeda tetapi jumlah baris dan label sama, kemungkinan besar isi `master` di GitHub IndoNLU berubah sejak unduhan awal (url sumber menunjuk `master`, bukan commit tertentu). Dalam kasus itu angka evaluasi Anda bisa sedikit berbeda dari yang ada di README utama. Jika jumlah baris berbeda, anggap data tidak valid dan unduh ulang dengan `--force`.

Lisensi dan sitasi dataset: lihat bagian "Dataset dan atribusi" di [README utama](../README.md).

## 2. Menjalankan training

```bash
python ml/train_baseline.py
```

Butuh sekitar setengah menit di laptop biasa (unduhan proyek ini: 23,6 detik). Protokol (tanpa kebocoran data):

1. Split resmi train / valid / test dipakai apa adanya.
2. Setiap teks lewat `ml.preprocess.preprocess`: NFKC, huruf kecil, hapus URL dan `@mention`, buang simbol `#` (kata dipertahankan), pendekkan karakter berulang ≥3 menjadi 2, rapikan spasi. Fungsi ini dipanggil di luar pipeline, jadi file `.joblib` hanya berisi objek scikit-learn.
3. TF-IDF (unigram + bigram, `min_df=2`, `sublinear_tf`) di-*fit* hanya pada **train**.
4. Kandidat: LinearSVC terkalibrasi dan Logistic Regression, masing-masing dengan `C` ∈ {0,1; 1; 10} (6 kandidat). Pemilihan berdasarkan **macro-F1 di valid** saja; jika seri, dipilih accuracy lebih tinggi, lalu log-loss lebih rendah.
5. Model terpilih dievaluasi **sekali** pada test.

Seed tetap (`random_state=42`).

> Mengubah `preprocess.py` memengaruhi training **dan** backend. Latih ulang model setelah mengubahnya.

### Keluaran (`ml/artifacts/`)

| File | Isi |
|------|-----|
| `tfidf_svm.joblib` | Pipeline terpilih + label + metadata (dimuat backend saat start) |
| `metrics.json` | Semua angka evaluasi |
| `confusion_matrix.png` | Confusion matrix pada test |

Versi `scikit-learn` saat training harus sama dengan versi di backend (`requirements.txt` keduanya dikunci identik) karena file `.joblib` sensitif terhadap versi.

## 3. Membaca `metrics.json`

| Kunci | Arti |
|-------|------|
| `dataset`, `split_sizes` | Dataset dan ukuran train / valid / test |
| `label_order` | Urutan label di `per_class` dan `confusion_matrix`: `positive`, `neutral`, `negative` |
| `tfidf` | Parameter TF-IDF yang dipakai |
| `validation.candidates` | Satu entri per kandidat: `classifier`, `params.C`, `accuracy`, `macro_f1`, `log_loss`, `fit_seconds`. **Semua angka di sini dari validation, bukan test** |
| `validation.selected_index` | Indeks (mulai 0) kandidat terpilih di `candidates` |
| `validation.reason` | Penjelasan teks mengapa kandidat itu terpilih |
| `test.accuracy`, `test.macro_f1` | Skor akhir pada test (dipakai sekali) |
| `test.per_class` | `precision`, `recall`, `f1-score`, `support` per kelas |
| `test.confusion_matrix` | `labels` dan `matrix`: **baris = label asli, kolom = prediksi** |
| `data_notes` | Jumlah teks test yang persis ada di train (mentah dan setelah preprocessing). Nilai 0 berarti tidak ada tumpang tindih |
| `model` | Metadata model: classifier, `C`, versi scikit-learn, waktu training, seed |
| `elapsed_seconds` | Lama seluruh skrip |

Tips membaca:

- **Macro-F1** adalah rata-rata F1 ketiga kelas tanpa bobot, jadi kelas kecil (neutral) sama pentingnya dengan kelas besar. Karena kelas tidak seimbang, angka ini lebih jujur daripada accuracy.
- **Bandingkan `validation` dengan `test`.** Kandidat terpilih adalah yang terbaik di valid, sehingga skor valid cenderung sedikit optimistis. Selisih besar ke test (di unduhan proyek ini valid 0,860 vs test 0,758) layak dicatat, apalagi test hanya 500 teks sehingga angkanya berfluktuasi.
- **`recall` rendah** pada satu kelas berarti banyak teks kelas itu terlewat. Di baseline ini neutral paling lemah.
- **`log_loss`** (di validation) menilai kualitas probabilitas/confidence, bukan hanya benar-salahnya prediksi. Semakin kecil semakin baik.

Tampilkan ringkasan cepat di PowerShell:

```powershell
$m = Get-Content ml\artifacts\metrics.json -Raw | ConvertFrom-Json
$m.model.classifier; $m.model.params.C
$m.test.accuracy; $m.test.macro_f1
```

## 4. Membaca `confusion_matrix.png`

![contoh bentuk](artifacts/confusion_matrix.png)

> Gambar di atas hanya tampil jika Anda sudah menjalankan training (file ada di `ml/artifacts/`).

- **Sumbu Y (True)** = label asli, **sumbu X (Predicted)** = label prediksi. Urutan: positive, neutral, negative.
- Setiap sel menampilkan **jumlah teks** dan, dalam kurung, **persentase dari kelas aslinya** (dinormalisasi per baris, jadi tiap baris berjumlah 100%).
- Warna biru mengikuti persentase itu (colorbar "fraction of true class"), bukan jumlah mentah. Kelas kecil tetap terlihat jelas.
- **Diagonal** = prediksi benar. Persentase diagonal pada tiap baris adalah **recall** kelas tersebut.
- **Di luar diagonal** = jenis kesalahan. Contoh pada hasil proyek ini: baris *neutral* hanya 43 dari 88 (49%) yang benar, dan 29 teks netral salah diprediksi *negative*.
- Judul gambar memuat classifier, `C`, accuracy, dan macro-F1 pada test.

Angka di gambar sama persis dengan `test.confusion_matrix.matrix` di `metrics.json`.

## Fine-tune IndoBERT di Colab

Training IndoBERT butuh GPU, jadi dijalankan di Google Colab, **bukan** di komputer lokal. Skrip `finetune_indobert.py` memakai data split resmi SmSA dan fungsi `preprocess()` yang sama dengan baseline, lalu memilih epoch terbaik dari **macro-F1 di validation**. Test dievaluasi sekali di akhir.

> Hasil evaluasi IndoBERT belum ada sampai Anda menjalankannya. Tidak ada angka yang disalin dari tempat lain.

### Langkah

1. Pastikan branch `fase-2-indobert` sudah di-push ke GitHub (notebook mengkloning repo publik).
2. Buka notebook dari GitHub di Colab:
   `https://colab.research.google.com/github/Lnathea/sentiment-dashboard/blob/fase-2-indobert/ml/notebooks/colab_finetune_indobert.ipynb`
   (atau Colab → *File → Open notebook → GitHub*, tempel `Lnathea/sentiment-dashboard`, pilih branch dan berkas notebook).
3. **Runtime → Change runtime type → T4 GPU.** Variabel `BRANCH` di sel pertama bisa diubah jika memakai branch lain.
4. **Runtime → Run all.** Notebook akan: cek GPU, `git clone`, `pip install -r ml/requirements-indobert.txt`, `python ml/download_data.py`, `python ml/finetune_indobert.py`, lalu membuat `indobert_artifacts.zip` dan mengunduhnya.
5. Di komputer lokal:
   - ekstrak zip ke `ml/artifacts/indobert/` (isi: `model.safetensors`, `config.json`, berkas tokenizer, `metrics_indobert.json`, `confusion_matrix_indobert.png`);
   - **salin** `metrics_indobert.json` dan `confusion_matrix_indobert.png` ke `ml/artifacts/`.

   ```powershell
   # Windows PowerShell (dari root repo; sesuaikan lokasi zip)
   Expand-Archive "$HOME\Downloads\indobert_artifacts.zip" ml\artifacts\indobert -Force
   Copy-Item ml\artifacts\indobert\metrics_indobert.json ml\artifacts\
   Copy-Item ml\artifacts\indobert\confusion_matrix_indobert.png ml\artifacts\
   ```

   ```bash
   # bash
   unzip -o ~/Downloads/indobert_artifacts.zip -d ml/artifacts/indobert
   cp ml/artifacts/indobert/metrics_indobert.json ml/artifacts/indobert/confusion_matrix_indobert.png ml/artifacts/
   ```

Isi `ml/artifacts/` tidak di-commit.

### Parameter

Nilai bawaan (ubah lewat argumen CLI, mis. `python ml/finetune_indobert.py --epochs 3 --batch-size 16 --grad-accum 2`):

| Argumen | Bawaan |
|---------|--------|
| `--model-name` | `indobenchmark/indobert-base-p1` |
| `--max-length` | 128 |
| `--learning-rate` | 2e-5 |
| `--batch-size` (per perangkat) / `--grad-accum` | 32 / 1 |
| `--epochs` | 4 |
| `--weight-decay` / `--warmup-ratio` | 0,01 / 0,1 |
| `--seed` | 42 |
| `--output-dir` | `ml/artifacts/indobert` |

fp16 aktif otomatis jika GPU tersedia (`--no-fp16` untuk mematikan). Jika memori GPU kurang, turunkan `--batch-size` dan naikkan `--grad-accum` agar batch efektifnya tetap 32.

### Label

Model memakai `id2label = {0: "negative", 1: "neutral", 2: "positive"}` (tersimpan di `config.json`). Laporan (`metrics_indobert.json`, confusion matrix) memakai urutan baseline `positive, neutral, negative` agar kedua model bisa dibandingkan langsung.

### Membaca `metrics_indobert.json`

Skemanya sama dengan `metrics.json` baseline (lihat [bagian 3](#3-membaca-metricsjson)), dengan perbedaan:

- `hyperparameters` menggantikan `tfidf`.
- `validation.candidates` berisi **satu entri per epoch** (`params.epoch`). `log_loss` adalah cross-entropy rata-rata di validation, dan `fit_seconds` bernilai `null` (tidak diukur per epoch). `selected_index` menunjuk epoch terbaik.
- `model` memuat tambahan `model_name` (`indobert`), `base_model`, `id2label`, `best_epoch`, `torch_version`, `transformers_version`, dan `device`.
- `data_notes` sama dengan baseline (jumlah teks test yang persis ada di train).

### Tes lokal

Tes untuk fungsi bantunya berjalan tanpa torch (bagian yang butuh torch otomatis di-skip):

```bash
cd backend
pytest -q ../ml/tests/test_finetune_indobert.py
```

