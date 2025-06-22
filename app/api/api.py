from fastapi import APIRouter, Query
from app.core.embed import EmbeddingManager

router = APIRouter()
manager = EmbeddingManager()
manager.load_index()

@router.get("/query")
def ask_question(q: str = Query(..., description="Your technical question")):
    results = manager.query(q)
    return {"query": q, "answers": results}
