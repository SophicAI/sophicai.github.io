import os, json, time, uuid, re
import numpy as np
from collections import OrderedDict
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing GOOGLE_API_KEY in .env")

genai.configure(api_key=API_KEY)
MODEL = os.getenv("MODEL", "gemini-1.5-flash")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
ALLOWED_ORIGINS = [o.strip() for o in os.getenv(
    "ALLOWED_ORIGINS",
    "http://127.0.0.1:5500,http://localhost:5500,http://localhost:3000"
).split(",") if o.strip()]

app = FastAPI(title="Sophic AI Tutor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load NCERT embeddings database
try:
    with open('chunks_with_gemini_embeddings.json', 'r', encoding='utf-8') as f:
        EMBEDDINGS_DATA = json.load(f)
    print(f"✅ Loaded {len(EMBEDDINGS_DATA)} NCERT chunks with embeddings")
except Exception as e:
    print(f"❌ Failed to load embeddings: {e}")
    EMBEDDINGS_DATA = []

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

def search_relevant_chunks(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Search for most relevant NCERT content chunks"""
    if not EMBEDDINGS_DATA:
        return []

    try:
        # Generate embedding for the query using Gemini
        model = genai.GenerativeModel("text-embedding-004")
        query_result = model.embed_content(query)
        query_embedding = np.array(query_result['embedding'])

        # Calculate similarities with all chunks
        similarities = []
        for item in EMBEDDINGS_DATA:
            chunk_embedding = np.array(item['embedding'])
            similarity = cosine_similarity(query_embedding, chunk_embedding)
            similarities.append({
                'similarity': similarity,
                'text': item['text'],
                'metadata': item.get('metadata', {})
            })

        # Sort by similarity and return top chunks
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]

    except Exception as e:
        print(f"❌ Error in chunk search: {e}")
        return []

# Enhanced Memory System
class ChatMessage:
    def __init__(self, role: str, content: str, timestamp: float = None):
        self.role = role
        self.content = content
        self.timestamp = timestamp or time.time()

class Session:
    def __init__(self, sid: str):
        self.sid = sid
        self.messages: List[ChatMessage] = []
        self.title = "New Chat"
        self.created = time.time()
        self.last_updated = time.time()

    def add_message(self, role: str, content: str):
        self.messages.append(ChatMessage(role, content))
        self.last_updated = time.time()

        # Auto-generate meaningful title
        if not self.title or self.title == "New Chat":
            if role == "user" and len(content.strip()) > 3:
                words = content.strip().split()[:6]
                self.title = " ".join(words) + ("..." if len(content.split()) > 6 else "")

    def get_history_for_model(self) -> List[Dict[str, Any]]:
        """Proper role mapping for Gemini API"""
        history = []
        recent_messages = self.messages[-75:]  # Keep last 75 messages

        for msg in recent_messages:
            if msg.role == "user":
                history.append({"role": "user", "parts": [msg.content]})
            elif msg.role == "assistant":
                history.append({"role": "model", "parts": [msg.content]})
        return history

    def get_preview(self) -> str:
        if not self.messages:
            return "No messages yet"
        last_user_msg = None
        for msg in reversed(self.messages):
            if msg.role == "user":
                last_user_msg = msg.content
                break
        return (last_user_msg[:60] + "...") if last_user_msg and len(last_user_msg) > 60 else (last_user_msg or "Empty chat")

sessions: OrderedDict[str, Session] = OrderedDict()

def get_or_create_session(sid: Optional[str] = None) -> Session:
    if not sid:
        sid = str(uuid.uuid4())
    if sid not in sessions:
        sessions[sid] = Session(sid)
        # Keep only last 50 sessions
        if len(sessions) > 50:
            oldest = next(iter(sessions))
            del sessions[oldest]
    return sessions[sid]

# Enhanced NCERT-focused teaching prompt
SYSTEM_PROMPT = """You are Sophic, a specialized NCERT tutor for Indian students in grades 6-12.

Your expertise:
- Deep knowledge of NCERT textbooks across all subjects
- CBSE curriculum alignment and exam preparation
- Step-by-step problem solving with clear explanations
- Age-appropriate language for different grade levels

Your personality:
- Friendly, patient, and encouraging
- Clear and concise explanations
- Focus on conceptual understanding over rote learning
- Provide examples from Indian context when relevant

Response guidelines:
- For math problems: Show complete step-by-step solutions
- For science topics: Explain concepts with real-life examples
- For social science: Provide comprehensive yet digestible information
- Always encourage follow-up questions for deeper understanding

When relevant NCERT content is provided, prioritize that information while adding your own insights."""

class AskReq(BaseModel):
    question: Optional[str] = None
    query: Optional[str] = None
    session_id: Optional[str] = None

    @property
    def text(self) -> str:
        return (self.question or self.query or "").strip()

def sanitize(text: str) -> str:
    if not text:
        raise HTTPException(400, "Please ask me a question!")
    return text.strip()

def compose_messages(session: Session, user_text: str, context_chunks: List[Dict] = None) -> List[Dict[str, Any]]:
    """Create conversation with NCERT context integration"""
    messages = []
    history = session.get_history_for_model()

    # Build enhanced system prompt with NCERT context
    enhanced_prompt = SYSTEM_PROMPT
    if context_chunks:
        enhanced_prompt += "

📚 RELEVANT NCERT CONTENT:
"
        for i, chunk in enumerate(context_chunks, 1):
            enhanced_prompt += f"
[Context {i}] {chunk['text'][:800]}..."
            if 'metadata' in chunk and chunk['metadata']:
                enhanced_prompt += f"
(Source: {chunk['metadata']})"
        enhanced_prompt += "

Use the above NCERT content to provide accurate, curriculum-aligned answers."

    if history:
        if history[-1]["role"] == "user":
            messages = [{"role": "user", "parts": [enhanced_prompt]}] + history
        else:
            messages = [{"role": "user", "parts": [enhanced_prompt]}] + history
    else:
        messages = [{"role": "user", "parts": [enhanced_prompt]}]

    # Add the new user message
    messages.append({"role": "user", "parts": [user_text]})
    return messages

def clean_response(text: str) -> str:
    """Clean and format the response"""
    text = re.sub(r'\{[^}]*"[^"]*"[^}]*\}', '', text, flags=re.DOTALL)
    text = re.sub(r'``````', '', text, flags=re.DOTALL)
    text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)
    text = text.strip()

    if not text or len(text) < 10:
        return "I'd love to help you with your NCERT studies! Could you please ask me a specific question about any subject or chapter?"
    return text

def generate_smart_suggestions(user_question: str, bot_response: str) -> List[str]:
    """Generate contextual follow-up questions"""
    question_lower = user_question.lower()

    if any(word in question_lower for word in ['math', 'solve', 'equation', 'calculate', 'problem']):
        return [
            "Can you show another similar example?",
            "What if the values were different?",
            "Explain why this method works"
        ]
    elif any(word in question_lower for word in ['explain', 'what is', 'define', 'meaning']):
        return [
            "Can you give a real-life example?",
            "How does this connect to other topics?",
            "What should I remember for exams?"
        ]
    elif any(word in question_lower for word in ['chapter', 'lesson', 'topic']):
        return [
            "What are the key points to remember?",
            "Can you give practice questions?",
            "How is this tested in exams?"
        ]
    else:
        return [
            "Can you explain this differently?",
            "What's most important here?",
            "Any memory tricks for this?"
        ]

@app.post("/ask")
async def ask(req: Request, body: AskReq):
    question = sanitize(body.text)
    session = get_or_create_session(body.session_id)
    session.add_message("user", question)

    # 🔍 SEARCH RELEVANT NCERT CONTENT
    relevant_chunks = search_relevant_chunks(question, top_k=3)

    model = genai.GenerativeModel(MODEL)

    try:
        # Compose messages with NCERT context
        conversation = compose_messages(session, question, relevant_chunks)

        print(f"[DEBUG] Sending {len(conversation)} messages to Gemini")
        print(f"[DEBUG] Found {len(relevant_chunks)} relevant NCERT chunks")

        response = model.generate_content(
            conversation,
            generation_config={
                "temperature": TEMPERATURE,
                "max_output_tokens": 1500,  # Longer for detailed explanations
                "top_p": 0.9,
                "top_k": 40
            }
        )

        raw_text = response.text if hasattr(response, "text") else str(response)
        clean_text = clean_response(raw_text)
        suggestions = generate_smart_suggestions(question, clean_text)

        print(f"[DEBUG] Success! Response length: {len(clean_text)}")

    except Exception as e:
        import traceback
        print("MODEL ERROR:", e)
        traceback.print_exc()
        clean_text = "I apologize, but I encountered a technical issue. Please try asking your NCERT question again, and I'll do my best to help you learn!"
        suggestions = ["Can you rephrase your question?", "Which subject are you studying?", "Which chapter is this from?"]

    session.add_message("assistant", clean_text)

    return JSONResponse({
        "answer": clean_text,
        "suggestions": suggestions,
        "session_id": session.sid,
        "meta": {
            "model": MODEL,
            "message_count": len(session.messages),
            "session_title": session.title,
            "chunks_found": len(relevant_chunks)
        }
    })

@app.get("/sessions")
async def get_sessions():
    session_list = []
    for sid, session in reversed(list(sessions.items())):
        session_list.append({
            "id": sid,
            "title": session.title,
            "preview": session.get_preview(),
            "created": session.created,
            "last_updated": session.last_updated,
            "message_count": len(session.messages)
        })
    return {"sessions": session_list}

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(404, "Session not found")

    session = sessions[session_id]
    messages = []
    for msg in session.messages:
        messages.append({
            "role": msg.role,
            "content": msg.content,
            "timestamp": msg.timestamp
        })

    return {
        "id": session.sid,
        "title": session.title,
        "messages": messages,
        "created": session.created
    }

@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
    return {"success": True}

@app.get("/status")
def status():
    return {
        "status": "online",
        "model": MODEL,
        "total_sessions": len(sessions),
        "embeddings_loaded": len(EMBEDDINGS_DATA),
        "version": "4.0-rag-enabled"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)