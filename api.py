from fastapi.staticfiles import StaticFiles
import os
import json
import math
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CONFIG
GEMINI_API_KEY = "AIzaSyBWfUf1GPvpHZS7C8N-ivIoBrJa6S4TbAA"
EMBED_MODEL = "models/text-embedding-004"
CHAT_MODEL = "gemini-2.5-flash-lite"
TOP_K = 5

# Load data and init API
genai.configure(api_key=GEMINI_API_KEY)
app = FastAPI(title="Sophic NCERT Tutor API", version="1.0.0")
app.mount("/", StaticFiles(directory=".", html=True), name="static")


# Add CORS middleware for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load and validate data
try:
    with open("chunks_with_gemini_embeddings.json", "r", encoding="utf-8") as f:
        DATA = json.load(f)
    logger.info(f"Loaded {len(DATA)} chunks from embeddings file")
except FileNotFoundError:
    logger.error("chunks_with_gemini_embeddings.json not found!")
    raise

# Prepare numpy arrays for fast cosine similarity
EMBS = []
META = []
TEXTS = []

valid_chunks = 0
for item in DATA:
    emb = item.get("embedding")
    if isinstance(emb, list) and len(emb) > 10:
        EMBS.append(emb)
        META.append({
            "source_file": item.get("source_file", ""),
            "grade": item.get("grade", ""),
            "subject": item.get("subject", ""),
            "book": item.get("book", ""),
            "chapter": item.get("chapter", ""),
            "chunk_index": item.get("chunk_index", 0),
        })
        TEXTS.append(item.get("text", ""))
        valid_chunks += 1

if valid_chunks == 0:
    logger.error("No valid embeddings found! Check your embeddings file.")
    raise ValueError("No valid embeddings")

EMBS = np.array(EMBS, dtype=np.float32)
norms = np.linalg.norm(EMBS, axis=1, keepdims=True) + 1e-10
EMBS = EMBS / norms

logger.info(f"Prepared {valid_chunks} valid embeddings for search")

class AskBody(BaseModel):
    query: str
    grade: Optional[str] = None
    subject: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    model: str
    chunks_loaded: int

def embed_query(text: str) -> np.ndarray:
    """Get embedding for user query"""
    try:
        res = genai.embed_content(model=EMBED_MODEL, content=text)
        q = res["embedding"]
        q = np.array(q, dtype=np.float32)
        q = q / (np.linalg.norm(q) + 1e-10)
        return q
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail="Failed to embed query")

def top_k_indices(query_emb: np.ndarray, k: int, grade_filter: str = None, subject_filter: str = None) -> List[int]:
    """Find top-k most similar chunks with optional filtering"""
    sims = EMBS @ query_emb
    
    valid_indices = []
    for i, (sim, meta) in enumerate(zip(sims, META)):
        if grade_filter and meta['grade'].lower() != grade_filter.lower():
            continue
        if subject_filter and subject_filter.lower() not in meta['subject'].lower():
            continue
        valid_indices.append((i, sim))
    
    if not valid_indices:
        idx = np.argpartition(sims, -k)[-k:]
        return idx[np.argsort(-sims[idx])].tolist()
    
    valid_indices.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in valid_indices[:k]]

def build_context(indices: List[int]) -> str:
    """Build context from retrieved chunks"""
    parts = []
    for i, idx in enumerate(indices, 1):
        meta = META[idx]
        txt = TEXTS[idx][:2000]
        parts.append(
            f"[Source {i}: {meta['source_file']} | Grade {meta['grade']} | "
            f"{meta['subject']} | {meta['book']} | Chapter {meta['chapter']}]\n{txt}"
        )
    return "\n\n---\n\n".join(parts)

def is_math_question(query: str) -> bool:
    """Simple heuristic to detect math questions"""
    math_keywords = ['calculate', 'solve', 'equation', 'formula', 'derivative', 'integral', 
                     'algebra', 'geometry', 'trigonometry', 'physics', 'chemistry', 'find the value']
    return any(keyword in query.lower() for keyword in math_keywords)

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model=CHAT_MODEL,
        chunks_loaded=len(EMBS)
    )

@app.post("/ask")
def ask(body: AskBody):
    try:
        q = body.query.strip()
        if not q:
            return {"answer": "What can I help you with?", "sources": []}
        
        # Always get embedding and context (but we might not use it)
        q_emb = embed_query(q)
        idxs = top_k_indices(q_emb, TOP_K, body.grade, body.subject)
        context = build_context(idxs)
        
        # Single smart prompt that handles everything
        prompt = f"""You are Sophic, a helpful NCERT tutor. The student asked: "{q}"

Here's some potentially relevant information from NCERT textbooks:
{context}

Instructions:
- If this is casual conversation (greetings, "how are you", random chat), respond naturally without forcing textbook content
- If this is a real academic question, use the textbook information to give a helpful explanation
- Be conversational and natural, not robotic
- Don't mention the textbook excerpts unless the question actually needs them

Your response:"""

        model = genai.GenerativeModel(CHAT_MODEL)
        resp = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.8,
                max_output_tokens=1000,
            )
        )
        
        answer = resp.text if hasattr(resp, "text") else str(resp)
        
        # Only show sources if it seems like an academic question
        sources = []
        if any(word in q.lower() for word in ['what', 'how', 'why', 'explain', 'solve', 'calculate']):
            for idx in idxs:
                sources.append({**META[idx], "similarity_rank": idxs.index(idx) + 1})
        
        return {"answer": answer, "sources": sources}
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return {"answer": "Something went wrong, try again?", "sources": []}


@app.get("/")
def root():
    """Root endpoint with API info"""
    return {
        "message": "Sophic NCERT Tutor API",
        "model": CHAT_MODEL,
        "endpoints": ["/ask", "/health"],
        "chunks_loaded": len(EMBS)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
