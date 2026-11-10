"""
Smart Inventory System - Backend Service Engine
Dirancang khusus untuk diintegrasikan ke Backend (FastAPI, Flask, Express, Django).
Menyediakan interface siap pakai untuk:
1. Pencarian Semantik Produk (SBERT Fine-Tuned Domain Retail)
2. Rekomendasi Alternatif Barang (Content-Based Cosine Similarity)
3. Bundling Paket Kebutuhan Asrama (Rule-Based Kombinasi Kegiatan)
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class InventoryEngine:
    def __init__(self, 
                 catalog_path: Optional[str] = None, 
                 rules_path: Optional[str] = None, 
                 model_path: Optional[str] = None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = os.path.abspath(os.path.join(base_dir, '..'))
        
        self.catalog_path = catalog_path or os.path.join(root_dir, 'data', 'processed', 'inventory_catalog.csv')
        self.rules_path = rules_path or os.path.join(root_dir, 'data', 'processed', 'combination_rules.csv')
        self.model_path = model_path or os.path.join(root_dir, 'models', 'sbert_finetuned')
        
        # 1. Load Data
        self.df = pd.read_csv(self.catalog_path)
        self.rules = pd.read_csv(self.rules_path) if os.path.exists(self.rules_path) else pd.DataFrame()
        
        # Siapkan representasi dokumen
        self.df['doc'] = self.df['category'] + ': ' + self.df['item_name'] + '. ' + self.df['description'].fillna('')
        
        # 2. Load Model Offline
        if os.path.exists(self.model_path):
            self.model = SentenceTransformer(self.model_path, local_files_only=True)
        else:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
        # 3. Compute Embeddings & Item Similarity Matrix
        self.corpus_embeddings = self.model.encode(self.df['doc'].tolist(), batch_size=64, normalize_embeddings=True)
        self.item_sim_matrix = cosine_similarity(self.corpus_embeddings, self.corpus_embeddings)

    def search(self, query: str, top_k: int = 5, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Pencarian Semantik Produk Berdasarkan Query Bahasa Alami
        """
        q_emb = self.model.encode([query], normalize_embeddings=True)
        sims = cosine_similarity(q_emb, self.corpus_embeddings)[0]
        
        # Filter kategori jika ada
        if category and category.lower() != 'all':
            valid_indices = self.df[self.df['category'].str.lower() == category.lower()].index.values
            if len(valid_indices) == 0:
                return []
            filtered_sims = sims[valid_indices]
            sorted_local = np.argsort(filtered_sims)[::-1][:top_k]
            top_idx = valid_indices[sorted_local]
        else:
            top_idx = np.argsort(sims)[::-1][:top_k]
            
        results = []
        for rank, idx in enumerate(top_idx, 1):
            r = self.df.iloc[idx]
            results.append({
                "rank": rank,
                "item_id": r['item_id'],
                "item_name": r['item_name'],
                "category": r['category'],
                "score": round(float(sims[idx]), 4),
                "quantity": int(r['quantity']),
                "unit": r['unit'],
                "location": r['location'],
                "condition": r['condition'],
                "availability": bool(r['availability']),
                "image_path": r['image_path']
            })
        return results

    def recommend_alternatives(self, item_name_or_id: str, top_n: int = 3) -> Dict[str, Any]:
        """
        Rekomendasi Alternatif Barang Jika Stok Habis atau Pilihan Serupa
        """
        # Cari item target
        match = self.df[(self.df['item_id'] == item_name_or_id) | 
                        (self.df['item_name'].str.contains(item_name_or_id, case=False, na=False))]
        if match.empty:
            return {"target_item": None, "alternatives": []}
            
        target_idx = match.index[0]
        target_row = self.df.iloc[target_idx]
        target_cat = target_row['category']
        
        sim_scores = self.item_sim_matrix[target_idx].copy()
        sim_scores[target_idx] = -1.0  # hindari diri sendiri
        
        # Utamakan kategori yang sama
        cat_mask = (self.df['category'] == target_cat).values
        same_cat_indices = np.where(cat_mask)[0]
        
        if len(same_cat_indices) > 1:
            sorted_indices = same_cat_indices[np.argsort(sim_scores[same_cat_indices])[::-1][:top_n]]
        else:
            sorted_indices = np.argsort(sim_scores)[::-1][:top_n]
            
        alternatives = []
        for idx in sorted_indices:
            r = self.df.iloc[idx]
            alternatives.append({
                "item_id": r['item_id'],
                "item_name": r['item_name'],
                "category": r['category'],
                "similarity_score": round(float(sim_scores[idx]), 4),
                "quantity": int(r['quantity']),
                "location": r['location'],
                "availability": bool(r['availability'])
            })
            
        return {
            "target_item": {
                "item_id": target_row['item_id'],
                "item_name": target_row['item_name'],
                "category": target_row['category'],
                "quantity": int(target_row['quantity']),
                "location": target_row['location']
            },
            "alternatives": alternatives
        }

    def get_event_bundles(self, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Mendapatkan Paket Kombinasi Barang Berdasarkan Kebutuhan Kegiatan Asrama
        """
        if self.rules.empty:
            return []
            
        df_rules = self.rules
        event_col = 'need' if 'need' in df_rules.columns else 'event_type'
        role_col = 'role_required' if 'role_required' in df_rules.columns else 'role_in_event'
        cat_col = 'target_category' if 'target_category' in df_rules.columns else 'needed_category'
        qty_col = 'default_qty' if 'default_qty' in df_rules.columns else 'min_quantity'
        constraint_col = 'constraints' if 'constraints' in df_rules.columns else 'constraint_notes'
        
        if event_type:
            df_rules = df_rules[df_rules[event_col].str.contains(event_type, case=False, na=False)]
            
        bundles = []
        for event, grp in df_rules.groupby(event_col):
            items_payload = []
            for _, r in grp.iterrows():
                needed_cat = r[cat_col]
                # Cari 1 item representatif di gudang
                cat_match = self.df[self.df['category'] == needed_cat]
                sample_item = cat_match.iloc[0] if not cat_match.empty else None
                
                items_payload.append({
                    "role": r[role_col],
                    "category": needed_cat,
                    "min_quantity": int(r[qty_col]),
                    "constraint": r[constraint_col],
                    "available_inventory_sample": {
                        "item_id": sample_item['item_id'] if sample_item is not None else "N/A",
                        "item_name": sample_item['item_name'] if sample_item is not None else "N/A",
                        "stock": int(sample_item['quantity']) if sample_item is not None else 0,
                        "location": sample_item['location'] if sample_item is not None else "N/A"
                    }
                })
            bundles.append({
                "event_type": event,
                "required_items": items_payload
            })
        return bundles

if __name__ == '__main__':
    engine = InventoryEngine()
    print("\n--- TEST SEARCH ENGINE ---")
    res = engine.search("audio splitter headset", top_k=2)
    for r in res:
        print(f"[{r['rank']}] {r['item_name']} | Score: {r['score']} | Stock: {r['quantity']} ({r['location']})")
        
    print("\n--- TEST ALTERNATIVE REC ---")
    alt = engine.recommend_alternatives("audio", top_n=2)
    print(f"Target: {alt['target_item']['item_name']}")
    for a in alt['alternatives']:
        print(f"-> Alternatif: {a['item_name']} | Score: {a['similarity_score']}")
