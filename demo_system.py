"""
Interactive Terminal Demo - Smart Inventory System
Menggunakan SBERT Fine-Tuned (Retail Domain 06)
"""

import os
import sys

# Tambahkan api ke path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'api'))

from inventory_service import InventoryEngine

def main():
    print("=" * 65)
    print("DEMO SISTEM INVENTARIS CERDAS (SMART INVENTORY SYSTEM)")
    print("=" * 65)
    
    print("\nInisialisasi Inventory Engine...")
    engine = InventoryEngine()
    print(f"Total Barang Terindeks: {len(engine.df)} item (21 Kategori)")
    
    # 1. Pencarian Semantik
    queries = [
        "audio splitter headset 3.5mm",
        "charger adapter kabel telepon genggam",
        "peralatan perbaikan rumah tangga obeng baut"
    ]
    
    for q in queries:
        print(f"\n[PENCARIAN SEMANTIK]: '{q}'")
        results = engine.search(q, top_k=2)
        for r in results:
            print(f"  {r['rank']}. {r['item_name']} ({r['category']})")
            print(f"     Skor Relevansi : {r['score']}")
            print(f"     Stok & Lokasi  : {r['quantity']} {r['unit']} | Lokasi: {r['location']}")
            print(f"     Status Fisik   : Kondisi {r['condition']} | {'Tersedia' if r['availability'] else 'Kosong'}")

    # 2. Rekomendasi Barang Pengganti
    item_target = "Stapler"
    rec = engine.recommend_alternatives(item_target, top_n=2)
    print(f"\n[REKOMENDASI ALTERNATIF]: '{item_target}'")
    if rec['target_item']:
        print(f"  Barang Utama: {rec['target_item']['item_name']} ({rec['target_item']['category']})")
        for idx, alt in enumerate(rec['alternatives'], 1):
            print(f"  Alternatif {idx}: {alt['item_name']}")
            print(f"    Skor Kemiripan : {alt['similarity_score']}")
            print(f"    Stok & Lokasi  : {alt['quantity']} Unit ({alt['location']})")

    # 3. Bundling Kegiatan Asrama
    event = "Dekorasi Panggung Acara"
    bundles = engine.get_event_bundles(event)
    print(f"\n[PAKET KOMBINASI KEGIATAN]: '{event}'")
    for b in bundles:
        for it in b['required_items']:
            sample = it['available_inventory_sample']
            print(f"  * [{it['role']}]: {sample['item_name']}")
            print(f"    Dibutuhkan : {it['min_quantity']} | Stok Gudang: {sample['stock']} ({sample['location']})")
            print(f"    Constraint : {it['constraint']}")

    print("\n" + "=" * 65)
    print("SUCCESS: Sistem siap diintegrasikan langsung ke backend aplikasi!")
    print("=" * 65)

if __name__ == '__main__':
    main()

