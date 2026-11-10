# Smart Inventory System (Retail Domain SBERT Fine-Tuned)

Repositori ini berisi implementasi sistem cerdas pengelolaan inventaris gudang asrama/kantor berbasis **Pencarian Semantik NLP (Sentence-BERT Fine-Tuned)**, **Rekomendasi Alternatif Barang**, dan **Aturan Kombinasi Paket Kegiatan**.

Proyek ini menggunakan konfigurasi optimal dari eksperimen `06_retail_products_classification` yang telah terbukti menghasilkan skor evaluasi kredibel (**Recall@1: 85.71%**, **Recall@5: 100%**, **MRR: 0.9286**) tanpa overfitting.

---

## 📁 Struktur Folder

```text
smart_inventory_system/
├── api/
│   ├── inventory_service.py     # Engine inti (Python class) untuk integrasi backend
│   └── server.py                # REST API wrapper (FastAPI) siap pakai
├── data/
│   ├── processed/
│   │   ├── inventory_catalog.csv  # 2.100 barang lengkap dengan stok, lokasi, kondisi
│   │   └── combination_rules.csv  # Aturan bundle kegiatan (dekorasi, rapat, pembersihan)
│   ├── evaluation/
│   │   └── ground_truth_queries.csv # Kueri uji realistis antar 21 kategori
│   └── retail_products_preprocessed.csv # Data hasil preprocessing & EDA
├── models/
│   └── sbert_finetuned/         # Bobot model Sentence-BERT yang di-fine-tune
├── notebooks/
│   ├── 01_eda_and_preprocessing.ipynb # Analisis EDA & data cleaning
│   ├── 02_sentence_bert_search.ipynb  # Baseline pencarian semantik (E2)
│   ├── 03_sentence_bert_finetuning.ipynb # Evaluasi fine-tuning SBERT (E3)
│   ├── 04_clip_multimodal_search.ipynb # Eksperimen pencarian multimodal citra & teks (E4)
│   ├── 05_recommendation_and_combination.ipynb # Rekomendasi & rule bundle
│   └── retail_pipeline.ipynb    # End-to-end classification pipeline
├── plots/                       # Visualisasi & grafik metrik untuk laporan/backend
│   ├── 01_evaluation_comparison_bar.png
│   ├── 02_inventory_category_distribution.png
│   ├── 03_warehouse_location_allocation.png
│   └── 04_similarity_distribution.png
├── reports/
│   └── evaluation_report.md     # Ringkasan ilmiah hasil metrik evaluasi
├── demo_system.py               # Demo interaktif command line
├── .gitignore                   # Proteksi push file besar (>100MB)
└── README.md
```

---

## 📊 Hasil Evaluasi Model

| Metrik Evaluasi | E1: TF-IDF Baseline | E2: SBERT Pretrained | E3: SBERT Fine-Tuned (Optimal) | Peningkatan vs E1 |
| :--- | :---: | :---: | :---: | :---: |
| **Recall@1 (Top-1 Match)** | 71.43% | 76.19% | **85.71%** | **+14.28%** |
| **Recall@3 (Top-3 Match)** | 90.48% | 85.71% | **100.00%** | **+9.52%** |
| **Recall@5 (Top-5 Match)** | 90.48% | 95.24% | **100.00%** | **+9.52%** |
| **MRR (Ranking Quality)** | 0.8095 | 0.8310 | **0.9286** | **+0.1191** |

---

## 🚀 Panduan Menjalankan

### 1. Menjalankan Demo Terminal
```bash
cd smart_inventory_system
python demo_system.py
```

### 2. Mengintegrasikan ke Backend (Python Native)
```python
from api.inventory_service import InventoryEngine

engine = InventoryEngine()

# 1. Cari barang dengan bahasa alami
results = engine.search("kabel charger smartphone", top_k=3)

# 2. Rekomendasi barang serupa jika stok habis
alternatives = engine.recommend_alternatives("Stapler", top_n=3)

# 3. Ambil paket barang untuk kegiatan
bundles = engine.get_event_bundles("Dekorasi Panggung")
```

### 3. Menjalankan REST API Microservice (FastAPI)
```bash
pip install fastapi uvicorn
python api/server.py
```
Akses dokumentasi interaktif Swagger API di: `http://localhost:8000/docs`

