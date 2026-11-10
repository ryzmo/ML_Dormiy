# Dokumentasi Arsitektur AI: Pemisahan Feature Extractor (DL) & Decision Making (ML)

Dokumen ini memetakan secara detail di mana **Deep Learning (hanya sebagai Feature Extractor)** dan **Machine Learning (sebagai Pengambil Keputusan / Decision Maker)** digunakan di setiap file pada folder `smart_inventory_system/`.

---

## 1. Ringkasan Peta Komponen & File

| Modul / Fitur | File Implementasi | Deep Learning (Feature Extractor Saja) | Machine Learning (Decision Making) |
| :--- | :--- | :--- | :--- |
| **Pencarian Teks Semantik** | `api/inventory_service.py`<br>`notebooks/02_sentence_bert_search.ipynb`<br>`notebooks/03_sentence_bert_finetuning.ipynb` | **Sentence-BERT (`all-MiniLM-L6-v2`)**<br>Mengekstrak teks kueri & katalog menjadi vektor berdimensi 384. | **Instance-Based ML / Metric Learning (k-NN / Cosine Similarity)**<br>Menghitung jarak vektor, merangking relevansi, dan mengambil keputusan Top-K barang terpilih. |
| **Rekomendasi Barang Pengganti** | `api/inventory_service.py`<br>`notebooks/05_recommendation_and_combination.ipynb` | **Sentence-BERT Embedding Matrix**<br>Menyediakan ruang fitur semantik katalog. | **Pairwise Distance Matrix & Filter Heuristik**<br>Menentukan barang alternatif terdekat dalam klaster kategori yang sama saat stok utama kosong. |
| **Pencarian Multimodal (Citra)** | `notebooks/04_clip_multimodal_search.ipynb` | **Vision Transformer (CLIP Vision Encoder)**<br>Mengekstrak piksel foto produk menjadi vektor citra berdimensi 512. | **Cross-Modal Nearest Neighbor Retrieval**<br>Menentukan kecocokan antara fitur citra dengan fitur katalog inventaris. |
| **Klasifikasi Kategori Otomatis** | `notebooks/retail_pipeline.ipynb` | **TF-IDF / SBERT Embedding**<br>Representasi fitur numerik dari teks deskripsi produk. | **Supervised Classifier (SVM / Logistic Regression / Naive Bayes)**<br>Membuat *decision boundary* untuk memprediksi kategori produk dari 21 kelas ritel. |
| **Aturan Paket Kegiatan Asrama** | `api/inventory_service.py`<br>`notebooks/05_recommendation_and_combination.ipynb` | *Tidak menggunakan DL* | **Rule-Based Decision Engine & Constraint Solver**<br>Menentukan bundle peralatan berdasarkan kebutuhan acara dan batasan stok fisik. |

---

## 2. Bedah Teknis per File (File by File)

### A. [`api/inventory_service.py`](file:///c:/Users/farha/OneDrive/Documents/A-PPTI23/CAWU4/ML/ASSIGNMENT/smart_inventory_system/api/inventory_service.py) (Backend Core Engine)

File ini adalah mesin inti yang akan dipanggil oleh backend web/aplikasi.

#### 1. Bagian Deep Learning (Feature Extractor)
- **Kode**:
  ```python
  self.model = SentenceTransformer(self.model_path, local_files_only=True)
  self.corpus_embeddings = self.model.encode(self.df['doc'].tolist(), normalize_embeddings=True)
  ```
- **Peran**: Hanya bertindak sebagai konverter pasif teks mentah $\rightarrow$ matriks angka $2.100 \times 384$. Model neural network berhenti sampai di sini.

#### 2. Bagian Machine Learning (Decision Maker)
- **Kode Retrieval**:
  ```python
  # Menghitung metric similarity
  sims = cosine_similarity(q_emb, self.corpus_embeddings)[0]
  # Decision making: menentukan ranking barang paling relevan
  top_idx = np.argsort(sims)[::-1][:top_k]
  ```
- **Kode Rekomendasi Pengganti**:
  ```python
  # Decision making: menentukan barang substitusi saat stok = 0
  sim_scores = self.item_sim_matrix[target_idx].copy()
  sim_scores[target_idx] = -1.0  # cegah pilih diri sendiri
  same_cat_indices = np.where(self.df['category'] == target_cat)[0]
  sorted_indices = same_cat_indices[np.argsort(sim_scores[same_cat_indices])[::-1][:top_n]]
  ```
- **Peran**: Mengambil keputusan barang mana yang ditampilkan ke user, memfilter stok, dan menentukan alternatif terdekat.

---

### B. [`notebooks/03_sentence_bert_finetuning.ipynb`](file:///c:/Users/farha/OneDrive/Documents/A-PPTI23/CAWU4/ML/ASSIGNMENT/smart_inventory_system/notebooks/03_sentence_bert_finetuning.ipynb) (Fine-Tuning SBERT)

- **Deep Learning**:
  - Model Transformer di-fine-tune dengan fungsi loss *MultipleNegativesRankingLoss*.
  - Tujuannya bukan klasifikasi langsung, melainkan **merapatkan jarak vektor** barang sejenis di ruang embedding (menjadi feature extractor domain ritel yang lebih peka).
- **Machine Learning (Evaluasi Keputusan)**:
  - Menggunakan metrik evaluasi Information Retrieval: **Recall@1, Recall@3, Recall@5, dan Mean Reciprocal Rank (MRR)**.
  - Membandingkan keputusan ranking dari 3 model:
    1. **E1 (TF-IDF + Cosine Nearest Neighbor)** $\rightarrow$ Recall@1: 71.43%
    2. **E2 (SBERT Pretrained + Cosine Nearest Neighbor)** $\rightarrow$ Recall@1: 76.19%
    3. **E3 (SBERT Fine-Tuned + Cosine Nearest Neighbor)** $\rightarrow$ Recall@1: **85.71%**, MRR: **0.9286**

---

### C. [`notebooks/04_clip_multimodal_search.ipynb`](file:///c:/Users/farha/OneDrive/Documents/A-PPTI23/CAWU4/ML/ASSIGNMENT/smart_inventory_system/notebooks/04_clip_multimodal_search.ipynb) (Pencarian Berbasis Foto)

- **Deep Learning (Feature Extractor)**:
  - `CLIPProcessor` dan `CLIPModel.get_image_features()`.
  - Mengonversi piksel foto JPG produk (`train/train/B004BI5F1G.jpg`) menjadi representasi vektor numerik berdimensi 512.
- **Machine Learning (Decision Maker)**:
  - Menggunakan algoritma **Cross-Modal Cosine Nearest Neighbor** untuk mencocokkan vektor foto dengan basis data katalog barang di gudang.

---

### D. [`notebooks/05_recommendation_and_combination.ipynb`](file:///c:/Users/farha/OneDrive/Documents/A-PPTI23/CAWU4/ML/ASSIGNMENT/smart_inventory_system/notebooks/05_recommendation_and_combination.ipynb) (Rekomendasi & Bundle)

- **Machine Learning**:
  - **Content-Based Filtering**: Algoritma kemiripan berbasis fitur konten (*item attributes*).
  - **Rule-Based Bundle Decision**:
    - Input: Kebutuhan acara (misal: "Dekorasi Panggung Acara").
    - Decision: Memilih kombinasi peralatan (alat kerajinan, perkakas pemasangan, pencahayaan & audio, perlengkapan luar ruang) dengan constraint stok fisik gudang.

---

## 3. Kesimpulan untuk Laporan Akademis / Sidang

> *"Dalam sistem ini, Deep Learning (Sentence-BERT dan CLIP) **dibatasi peranannya hanya sebagai Feature Representation Engine** untuk memproyeksikan data teks dan citra ke dalam ruang vektor berdimensi tinggi. Seluruh proses **pengambilan keputusan (Decision Making)**—mulai dari penentuan ranking relevansi kueri, pencarian tetangga terdekat (Nearest Neighbor), rekomendasi barang substitusi, hingga klasifikasi kategori—sepenuhnya dijalankan oleh **algoritma Machine Learning klasik Scikit-Learn** dan heuristik sistem pendukung keputusan."*

