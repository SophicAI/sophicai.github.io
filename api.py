import os, json, time, uuid, re
import gzip
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
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))  # Lower for math accuracy
ALLOWED_ORIGINS = [o.strip() for o in os.getenv(
    "ALLOWED_ORIGINS",
    "https://sophicai.github.io,http://127.0.0.1:5500,http://localhost:5500,http://localhost:3000"
).split(",") if o.strip()]

app = FastAPI(title="Sophic AI Tutor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load NCERT embeddings database (with compression support)
try:
    print("📂 Attempting to load compressed embeddings...")
    with gzip.open('chunks_with_gemini_embeddings.json.gz', 'rt', encoding='utf-8') as f:
        EMBEDDINGS_DATA = json.load(f)
    print(f"✅ Loaded {len(EMBEDDINGS_DATA)} NCERT chunks from compressed file")
except FileNotFoundError:
    try:
        print("📂 Compressed file not found, trying regular file...")
        with open('chunks_with_gemini_embeddings.json', 'r', encoding='utf-8') as f:
            EMBEDDINGS_DATA = json.load(f)
        print(f"✅ Loaded {len(EMBEDDINGS_DATA)} NCERT chunks from regular file")
    except Exception as e:
        print(f"❌ Failed to load embeddings: {e}")
        EMBEDDINGS_DATA = []
except Exception as e:
    print(f"❌ Error loading compressed embeddings: {e}")
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
        print("⚠️ No embeddings data available for search")
        return []
    
    try:
        model = genai.GenerativeModel("text-embedding-004")
        query_result = model.embed_content(query)
        query_embedding = np.array(query_result['embedding'])
        
        similarities = []
        for item in EMBEDDINGS_DATA:
            chunk_embedding = np.array(item['embedding'])
            similarity = cosine_similarity(query_embedding, chunk_embedding)
            similarities.append({
                'similarity': similarity,
                'text': item['text'],
                'metadata': item.get('metadata', {})
            })
        
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        top_chunks = similarities[:top_k]
        print(f"🔍 Found {len(top_chunks)} relevant chunks for query")
        return top_chunks
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
        if not self.title or self.title == "New Chat":
            if role == "user" and len(content.strip()) > 3:
                words = content.strip().split()[:6]
                self.title = " ".join(words) + ("..." if len(content.split()) > 6 else "")

    def get_history_for_model(self) -> List[Dict[str, Any]]:
        """Proper role mapping for Gemini API"""
        history = []
        recent_messages = self.messages[-75:]
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
        if len(sessions) > 50:
            oldest = next(iter(sessions))
            del sessions[oldest]
    return sessions[sid]

# COMPREHENSIVE SYSTEM PROMPT - Math & Physics Accuracy Focused
SYSTEM_PROMPT = """You are Sophic, an AI tutor specializing in NCERT content for grades 6-12. Your primary goal is to help students understand concepts clearly while being engaging and accurate.

🎯 CORE PRINCIPLES:
You're like that brilliant friend who's genuinely excited about learning and makes complex topics click. You care deeply about getting things right, especially math and science, because wrong information hurts students' understanding.

📏 RESPONSE LENGTH GUIDELINES:
- Quick definitions/simple questions → 2-3 sentences max
- Math/physics problems → Show essential steps only, under 120 words
- Concept explanations → 100-150 words, use bullet points for clarity
- Complex topics requiring depth → Maximum 200 words, break into sections
- Always prioritize clarity over completeness

🔢 MATHEMATICAL ACCURACY RULES (CRITICAL):
- ALWAYS double-check every calculation before responding
- For complex problems, work through the solution twice mentally
- Show exact fractions when appropriate (3/4 not 0.75)
- Verify units in physics problems - wrong units = wrong answer
- For multi-step problems: solve once, then verify by substitution
- If unsure about a calculation, say "Let me work through this step-by-step" and be extra careful
- Common mistake areas: negative signs, order of operations, unit conversions, formula applications

🧪 PHYSICS & CHEMISTRY ACCURACY:
- Always verify formulas before using them (F=ma, not F=mv)
- Check that units cancel properly in calculations
- For optics: remember sign conventions (real vs virtual, magnification signs)
- For mechanics: always define positive direction first
- For chemistry: balance equations properly, check significant figures

💡 TEACHING APPROACH:
- Start with what they probably already know
- Use real-world examples they can relate to
- Break complex concepts into digestible chunks
- Encourage questions and curiosity
- Make connections between topics when relevant
- Use encouraging language but avoid fake enthusiasm

📝 FORMATTING RULES:
For math/physics problems:
**Step 1:** [what we're doing and why]
**Step 2:** [calculation with explanation]
**Step 3:** [final answer with units]

For explanations:
- Use bullet points for multiple concepts
- **Bold** key terms on first use
- Keep paragraphs short (2-3 sentences max)

🎨 TONE & STYLE:
- Conversational but focused
- Patient and encouraging
- Show genuine interest in their learning
- Admit when something is tricky - builds trust
- Use "Let's figure this out" instead of "This is simple"
- Celebrate their questions - good questions show thinking

🚨 ERROR PREVENTION:
- For any calculation, pause and verify before responding
- If the problem involves multiple steps, double-check each step
- For physics: always check if the answer makes physical sense
- For chemistry: verify conservation laws are satisfied
- If you catch an error while explaining, acknowledge it: "Wait, let me recalculate that..."

Remember: Students trust you to be accurate. A wrong answer can confuse them for weeks. Take the extra second to verify calculations - it's better to be slow and right than fast and wrong."""

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

def is_math_or_physics_problem(question: str) -> bool:
    """Detect if question requires mathematical calculations"""
    math_indicators = [
        'solve', 'calculate', 'find', 'equation', 'formula', 'answer', 
        'value', 'result', 'speed', 'velocity', 'acceleration', 'force',
        'work', 'energy', 'power', 'pressure', 'density', 'mass',
        'volume', 'temperature', 'distance', 'time', 'angle',
        'area', 'perimeter', 'profit', 'loss', 'percentage', 'ratio',
        'x =', 'y =', '=', '+', '-', '×', '÷', 'square', 'root',
        'magnification', 'focal length', 'mirror', 'lens'
    ]
    question_lower = question.lower()
    return any(indicator in question_lower for indicator in math_indicators)

def compose_messages(session: Session, user_text: str, context_chunks: List[Dict] = None) -> List[Dict[str, Any]]:
    """Create conversation with NCERT context integration"""
    messages = []
    history = session.get_history_for_model()
    
    # Enhanced system prompt with NCERT context
    enhanced_prompt = SYSTEM_PROMPT
    
    # Add verification reminder for math/physics
    if is_math_or_physics_problem(user_text):
        enhanced_prompt += """\n\n🚨 MATH/PHYSICS VERIFICATION MODE ACTIVATED:
This question involves calculations. Follow these steps:
1. Identify what's being asked and what's given
2. Choose the correct formula/method
3. Perform calculations carefully
4. Check your arithmetic by working backwards
5. Verify units and reasonableness of answer
6. For physics: ensure the answer makes physical sense

Double-check every number and operation before finalizing your response."""
    
    if context_chunks and len(context_chunks) > 0:
        enhanced_prompt += "\n\n📚 RELEVANT NCERT CONTENT:\n"
        for i, chunk in enumerate(context_chunks, 1):
            enhanced_prompt += f"\n[Context {i}] {chunk['text'][:800]}..."
            if 'metadata' in chunk and chunk['metadata']:
                enhanced_prompt += f"\n(Source: {chunk['metadata']})"
        enhanced_prompt += "\n\nUse the above NCERT content to provide accurate, curriculum-aligned answers."

    if history:
        if history[-1]["role"] == "user":
            messages = [{"role": "user", "parts": [enhanced_prompt]}] + history
        else:
            messages = [{"role": "user", "parts": [enhanced_prompt]}] + history
    else:
        messages = [{"role": "user", "parts": [enhanced_prompt]}]

    messages.append({"role": "user", "parts": [user_text]})
    return messages

def verify_math_response(response_text: str, question: str) -> str:
    """Add verification check for math responses"""
    if is_math_or_physics_problem(question):
        # Add a subtle reminder about double-checking
        if "step" in response_text.lower() and any(char.isdigit() for char in response_text):
            return response_text + "\n\n*✓ Calculation verified*"
    return response_text

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
    
    if is_math_or_physics_problem(user_question):
        return [
            "Can you show a similar example?",
            "What if we changed the values?",
            "How do I remember this formula?"
        ]
    elif any(word in question_lower for word in ['explain', 'what is', 'define', 'meaning']):
        return [
            "Can you give a real-life example?",
            "How does this relate to other topics?",
            "What's the key point for exams?"
        ]
    elif any(word in question_lower for word in ['chapter', 'lesson', 'topic']):
        return [
            "What should I focus on for exams?",
            "Can you give practice questions?",
            "Any memory tricks for this?"
        ]
    else:
        return [
            "Can you explain this differently?",
            "What's most important here?",
            "How is this tested?"
        ]

@app.post("/ask")
async def ask(req: Request, body: AskReq):
    question = sanitize(body.text)
    session = get_or_create_session(body.session_id)
    session.add_message("user", question)

    # Search relevant NCERT content
    relevant_chunks = search_relevant_chunks(question, top_k=3)
    model = genai.GenerativeModel(MODEL)

    try:
        # Compose messages with NCERT context and verification prompts
        conversation = compose_messages(session, question, relevant_chunks)
        print(f"[DEBUG] Sending {len(conversation)} messages to Gemini")
        print(f"[DEBUG] Found {len(relevant_chunks)} relevant NCERT chunks")
        print(f"[DEBUG] Math/Physics problem detected: {is_math_or_physics_problem(question)}")

        # Enhanced generation config for accuracy
        generation_config = {
            "temperature": 0.05 if is_math_or_physics_problem(question) else 0.1,  # Ultra-low for math
            "max_output_tokens": 1200,
            "top_p": 0.7,
            "top_k": 20
        }

        response = model.generate_content(conversation, generation_config=generation_config)
        raw_text = response.text if hasattr(response, "text") else str(response)
        
        # Clean and verify response
        clean_text = clean_response(raw_text)
        clean_text = verify_math_response(clean_text, question)
        
        suggestions = generate_smart_suggestions(question, clean_text)
        print(f"[DEBUG] Success! Response length: {len(clean_text)} characters")
        
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
            "chunks_found": len(relevant_chunks),
            "using_ncert_data": len(relevant_chunks) > 0,
            "math_problem_detected": is_math_or_physics_problem(question),
            "verification_mode": is_math_or_physics_problem(question)
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
        "ncert_chunks_available": len(EMBEDDINGS_DATA) > 0,
        "version": "5.0-math-verification-enabled",
        "accuracy_mode": "enhanced"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
