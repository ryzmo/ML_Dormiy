"""
FastAPI Server Wrapper - Siap Diintegrasikan ke Backend / Microservice
Endpoint:
- GET  /api/health
- POST /api/search
- POST /api/recommend
- GET  /api/bundles
"""

from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from inventory_service import InventoryEngine

app = FastAPI(
    title="AI Smart Inventory Service API",
    description="Microservice untuk Pencarian Semantik, Rekomendasi Barang, dan Bundling Kegiatan Asrama berbasis Sentence-BERT Fine-Tuned.",
    version="1.0.0"
)

# Inisialisasi engine singleton
engine = InventoryEngine()

class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    category: Optional[str] = None

class RecommendRequest(BaseModel):
    item_name_or_id: str
    top_n: Optional[int] = 3

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "service": "AI Smart Inventory Service",
        "catalog_size": len(engine.df),
        "model_loaded": engine.model_path
    }

@app.post("/api/search")
def search_items(req: SearchRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query pencarian tidak boleh kosong")
    results = engine.search(query=req.query, top_k=req.top_k, category=req.category)
    return {
        "query": req.query,
        "total_results": len(results),
        "results": results
    }

@app.post("/api/recommend")
def get_recommendations(req: RecommendRequest):
    rec = engine.recommend_alternatives(item_name_or_id=req.item_name_or_id, top_n=req.top_n)
    if not rec["target_item"]:
        raise HTTPException(status_code=404, detail=f"Item '{req.item_name_or_id}' tidak ditemukan di katalog")
    return rec

@app.get("/api/bundles")
def get_bundles(event_type: Optional[str] = Query(None, description="Jenis kegiatan, misal: Dekorasi Panggung")):
    bundles = engine.get_event_bundles(event_type=event_type)
    return {
        "total_bundles": len(bundles),
        "bundles": bundles
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

