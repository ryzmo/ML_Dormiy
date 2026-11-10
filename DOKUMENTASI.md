# 📦 AI Smart Inventory System — Dokumentasi Lengkap & Ringkas

Link dataset: https://www.kaggle.com/competitions/retail-products-classification?utm_source=chatgpt.com 

Sistem Cerdas Manajemen Inventaris berbasis **Pencarian Semantik NLP (Sentence-BERT)**, **Machine Learning Decision Engine**, dan **Sistem Rekomendasi Pengganti Barang**.

---

## 📌 Ringkasan Cepat: Siapa Melakukan Apa?

Untuk mematuhi batasan bahwa **Deep Learning hanya sebagai Feature Extractor**:

```text
[Input Teks / Foto]
       │
       ▼
[Deep Learning: Feature Extractor]  --> Hanya mengubah teks/gambar mentah jadi deretan angka (vektor 384/512 dimensi).
       │                                Model: Sentence-BERT / CLIP. (TIDAK mengambil keputusan bisnis).
       ▼ (Embedding Vector)
[Machine Learning: Decision Maker]  --> Otak pengambil keputusan sistem:
                                        1. Ranking Retrieval: k-NN / Cosine Similarity (menentukan barang teratas).
                                        2. Klasifikasi Kategori: Linear SVM Classifier (menentukan kategori produk).
                                        3. Rekomendasi Alternatif: Content-Based Pairwise Graph (mencari pengganti saat stok kosong).
                                        4. Paket Kegiatan: Rule-Based Constraint Engine (menyusun bundle kebutuhan acara).
```

---

## 📁 Struktur File Proyek (`smart_inventory_system/`)

Semua file telah ditata rapi dan modular:

```text
smart_inventory_system/
├── api/                                 # KONEKTOR KE BACKEND
│   ├── inventory_service.py             # Class engine utama (search, recommend, bundle)
│   └── server.py                        # REST API FastAPI (endpoint /search, /recommend, /bundles)
│
├── data/                                # DATASET & KATALOG
│   ├── processed/
│   │   ├── inventory_catalog.csv        # 2.100 barang lengkap (stok, lokasi rak, kondisi fisik, gambar)
│   │   └── combination_rules.csv        # Aturan paket kegiatan asrama
│   └── evaluation/
│       └── ground_truth_queries.csv     # 21 kueri uji standar untuk evaluasi model
│
├── models/                              # MODEL YANG SUDAH JADI (SIAP PAKAI)
│   ├── sbert_finetuned/                 # [DL Feature Extractor] Bobot model Sentence-BERT
│   │   └── model.safetensors            # File weights (90.8 MB)
│   └── classifier/                      # [ML Decision Maker] Model Supervised Scikit-Learn
│       └── category_classifier_svm.pkl  # Linear SVM Classifier hasil training
│
├── notebooks/                           # EKSPERIMEN JUPYTER (01 - 05 LENGKAP)
│   ├── 01_eda_and_preprocessing.ipynb   # Tahap 1: EDA Matriks 2x2 & data cleaning
│   ├── 02_sentence_bert_search.ipynb    # Tahap 2: Baseline SBERT Pretrained (E2)
│   ├── 03_sentence_bert_finetuning.ipynb# Tahap 3: Evaluasi SBERT Fine-Tuning (E3)
│   ├── 04_clip_multimodal_search.ipynb  # Tahap 4: Pencarian berbasis gambar (E4)
│   ├── 05_recommendation_and_combination.ipynb # Tahap 5: Rekomendasi alternatif & bundle
│   └── retail_pipeline.ipynb            # Pipeline klasifikasi kategori produk
│
├── plots/                               # GRAFIK UNTUK LAPORAN / DASHBOARD
│   ├── 01_evaluation_comparison_bar.png # Grafik perbandingan performa E1 vs E2 vs E3
│   ├── 02_inventory_category_distribution.png # Grafik distribusi 21 kategori produk
│   ├── 03_warehouse_location_allocation.png   # Grafik penempatan rak barang gudang
│   └── 04_similarity_distribution.png  # Grafik kurva kemiripan semantik antar-barang
│
├── demo_system.py                       # Skrip CLI untuk menguji semua fungsi sekaligus
├── .gitignore                           # Mencegah file model/arsip besar ditolak GitHub
└── README.md                            # Panduan teknis repositori
```

---

## 📊 Hasil Evaluasi Ilmiah (Recall & MRR)

Hasil eksperimen menunjukkan peningkatan performa yang signifikan dan kredibel tanpa *overfitting*:

| Metrik Evaluasi | E1: TF-IDF Baseline | E2: SBERT Pretrained | E3: SBERT Fine-Tuned (Optimal) | Peningkatan vs E1 |
| :--- | :---: | :---: | :---: | :---: |
| **Recall@1 (Top-1 Match)** | 71.43% | 76.19% | **85.71%** | **+14.28%** 🚀 |
| **Recall@3 (Top-3 Match)** | 90.48% | 85.71% | **100.00%** | **+9.52%** |
| **Recall@5 (Top-5 Match)** | 90.48% | 95.24% | **100.00%** | **+9.52%** |
| **MRR (Ranking Quality)** | 0.8095 | 0.8310 | **0.9286** | **+0.1191** 📈 |

> *Catatan Ilmiah: Angka **Recall@1 = 85.71%** dan **MRR = 0.9286** adalah capaian ideal di dunia nyata; artinya ~86% pencarian langsung tepat sasaran di urutan pertama, dan sisanya selalu ditemukan di urutan ke-2 atau ke-3.*

---

## 💡 Contoh Alur Nyata (*Step-by-Step Flow*)

### 1. Skenario Input Teks: `"audio splitter headset 3.5mm"`
1. **Ekstraksi Fitur (DL)**: Teks masuk ke Sentence-BERT $\rightarrow$ menghasilkan vektor angka 384 dimensi.
2. **Pengambilan Keputusan (ML)**: Algoritma *Cosine Nearest Neighbor* menghitung jarak vektor kueri ke 2.100 barang di database:
   - **Rank 1**: *For iPod iPad RED HEART Audio Headset Splitter 3.5mm* (Skor: 0.4789, Lokasi: Rak Logistik B1).
3. **Decision Rekomendasi (ML)**: Jika stok barang utama habis, model *Content-Based Scikit-Learn* otomatis memutuskan barang alternatif sejenis:
   - *Alternatif*: *CHANNEL PLUS 2-Way Splitter* (Stok: 10 di Rak B2).

### 2. Skenario Input Gambar: Foto Fisik Alat di Gudang
1. **Ekstraksi Fitur (DL)**: Foto produk diolah oleh CLIP Vision Transformer $\rightarrow$ menghasilkan vektor 512 dimensi.
2. **Pengambilan Keputusan (ML)**: Vektor dimasukkan ke model `category_classifier_svm.pkl`:
   - **Keputusan**: Terklasifikasi otomatis ke kategori `"All Electronics"` dengan confidence tinggi.
   - **Keputusan Pencocokan**: Algoritma k-NN mencocokkan fisik barang dengan database untuk menampilkan nama resmi, stok, dan lokasi rak penyimpanannya.

---

## 🛠️ Cara Mengintegrasikan ke Backend

### Opsi A: Import Class Langsung (Python Django / Flask / Fastify)
```python
from api.inventory_service import InventoryEngine

engine = InventoryEngine()

# 1. Pencarian Semantik Bahasa Alami
hasil = engine.search("charger kabel smartphone", top_k=3)

# 2. Rekomendasi Pengganti Jika Stok Kosong
alternatif = engine.recommend_alternatives("Stapler", top_n=2)

# 3. Paket Kebutuhan Kegiatan Asrama
paket = engine.get_event_bundles("Dekorasi Panggung Acara")
```

### Opsi B: Menjalankan Sebagai Microservice REST API (FastAPI)
Jalankan server API:
```bash
python smart_inventory_system/api/server.py
```
Akses antarmuka dokumentasi Swagger interaktif di browser:  
👉 **`http://localhost:8000/docs`**

- `POST /api/search` $\rightarrow$ Kirim JSON `{"query": "nama barang", "top_k": 5}`.
- `POST /api/recommend` $\rightarrow$ Kirim JSON `{"item_name_or_id": "Stapler", "top_n": 3}`.
- `GET /api/bundles` $\rightarrow$ Ambil rekomendasi paket kegiatan.

