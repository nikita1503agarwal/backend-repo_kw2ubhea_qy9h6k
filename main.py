import os
import uuid
import hashlib
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Literal, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import FileResponse
import jwt  # PyJWT
from pydantic import BaseModel, EmailStr

from database import db  # may be None if env not set

# ----------------------------
# Config
# ----------------------------
SECRET_KEY = os.getenv("JWT_SECRET", "change-this-in-prod")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
AUDIO_DIR = os.path.join(UPLOAD_DIR, "audio")
TTS_DIR = os.path.join(UPLOAD_DIR, "tts")
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TTS_DIR, exist_ok=True)

SUPPORTED_LANGUAGES = [
    "hi", "en-IN", "bn", "te", "mr", "ta", "ur", "gu", "kn", "ml", "pa", "or", "as"
]

DB_AVAILABLE = db is not None

# In-memory fallbacks if DB is not available (ephemeral per process)
users_mem: Dict[str, Dict[str, Any]] = {}
voices_mem: Dict[str, Dict[str, Any]] = {}
jobs_mem: Dict[str, Dict[str, Any]] = {}

# ----------------------------
# App setup
# ----------------------------
app = FastAPI(title="Vaakya API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Auth utils (PyJWT + PBKDF2)
# ----------------------------
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

def get_password_hash(password: str) -> str:
    salt = secrets.token_hex(16)
    iterations = 100_000
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), iterations)
    return f"pbkdf2${iterations}${salt}${dk.hex()}"

def verify_password(plain_password: str, stored: str) -> bool:
    try:
        scheme, iter_s, salt, hash_hex = stored.split("$")
        if scheme != "pbkdf2":
            return False
        iterations = int(iter_s)
        dk = hashlib.pbkdf2_hmac("sha256", plain_password.encode(), bytes.fromhex(salt), iterations)
        return secrets.compare_digest(dk.hex(), hash_hex)
    except Exception:
        return False

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except jwt.PyJWTError:
        raise credentials_exception

    user = None
    if DB_AVAILABLE:
        user = db["users"].find_one({"_id": user_id})
    else:
        user = users_mem.get(user_id)
    if not user:
        raise credentials_exception
    return user

# ----------------------------
# Schemas
# ----------------------------
class RegisterModel(BaseModel):
    name: str
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    accessToken: str
    tokenType: str = "bearer"

class TTSRequest(BaseModel):
    text: str
    voice: str
    language: str
    speed: float = 1.0
    pitch: float = 1.0

# ----------------------------
# Health & DB test
# ----------------------------
@app.get("/")
def read_root():
    return {"message": "Vaakya backend is running"}

@app.get("/test")
def test_database():
    response = {
        "backend": "✅ Running",
        "database": "✅ Connected & Working" if DB_AVAILABLE else "❌ Not initialized (using in-memory fallback)",
        "database_url": "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set",
        "database_name": "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set",
        "connection_status": "Connected" if DB_AVAILABLE else "Disabled",
        "collections": []
    }
    if DB_AVAILABLE:
        try:
            response["collections"] = db.list_collection_names()
        except Exception as e:
            response["database"] = f"⚠️ Connected but error: {str(e)[:80]}"
    return response

# ----------------------------
# Simple data access helpers (DB or memory)
# ----------------------------

def users_find_one(filter: Dict[str, Any]):
    if DB_AVAILABLE:
        return db["users"].find_one(filter)
    # memory mode
    if "email" in filter:
        for u in users_mem.values():
            if u.get("email") == filter["email"]:
                return u
    if "_id" in filter:
        return users_mem.get(filter["_id"])
    return None


def users_insert_one(doc: Dict[str, Any]):
    if DB_AVAILABLE:
        db["users"].insert_one(doc)
    else:
        users_mem[doc["_id"]] = doc


def voices_list_all() -> List[Dict[str, Any]]:
    if DB_AVAILABLE:
        return list(db["voices"].find({}, {"_id": 0}))
    return list(voices_mem.values())


def voices_upsert_by_id(doc: Dict[str, Any]):
    if DB_AVAILABLE:
        db["voices"].update_one({"id": doc["id"]}, {"$set": doc}, upsert=True)
    else:
        voices_mem[doc["id"]] = doc


def jobs_insert(doc: Dict[str, Any]):
    if DB_AVAILABLE:
        db["jobs"].insert_one(doc)
    else:
        jobs_mem[doc["_id"]] = doc


def jobs_find_one(filter: Dict[str, Any]):
    if DB_AVAILABLE:
        return db["jobs"].find_one(filter)
    # memory mode supports lookup by _id and type and user_id
    job = jobs_mem.get(filter.get("_id"))
    if not job:
        return None
    if filter.get("type") and job.get("type") != filter["type"]:
        return None
    if filter.get("user_id") and job.get("user_id") != filter["user_id"]:
        return None
    return job


def jobs_find(filter: Dict[str, Any]) -> List[Dict[str, Any]]:
    if DB_AVAILABLE:
        return list(db["jobs"].find(filter, {"_id": 0}))
    res = []
    for j in jobs_mem.values():
        ok = True
        for k, v in filter.items():
            if j.get(k) != v:
                ok = False
                break
        if ok:
            res.append({k: v for k, v in j.items() if k != "_id"})
    return res

# ----------------------------
# Auth endpoints
# ----------------------------
@app.post("/api/auth/register", response_model=TokenResponse)
def register(payload: RegisterModel):
    existing = users_find_one({"email": payload.email})
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user_id = str(uuid.uuid4())
    users_insert_one({
        "_id": user_id,
        "name": payload.name,
        "email": payload.email,
        "password_hash": get_password_hash(payload.password),
        "role": "user",
        "created_at": datetime.now(timezone.utc)
    })
    access = create_access_token({"sub": user_id})
    return TokenResponse(accessToken=access)

@app.post("/api/auth/login", response_model=TokenResponse)
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = users_find_one({"email": form_data.username})
    if not user or not verify_password(form_data.password, user.get("password_hash", "")):
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    access = create_access_token({"sub": user["_id"]})
    return TokenResponse(accessToken=access)

# ----------------------------
# Voices
# ----------------------------
@app.get("/api/voices")
def list_voices() -> List[Dict[str, Any]]:
    voices = voices_list_all()
    if not voices:
        seed_voices = [
            {"id": "voice_hindi_male", "name": "Arjun", "language": "hi", "gender": "male", "model_ref": "coqui:hi-male-1", "sample_url": None},
            {"id": "voice_hindi_female", "name": "Anaya", "language": "hi", "gender": "female", "model_ref": "coqui:hi-female-1", "sample_url": None},
            {"id": "voice_enIN_neutral", "name": "Neer", "language": "en-IN", "gender": "neutral", "model_ref": "coqui:en-in-1", "sample_url": None},
        ]
        for v in seed_voices:
            voices_upsert_by_id(v)
        voices = seed_voices
    return voices

# ----------------------------
# STT upload (batch) - placeholder processing
# ----------------------------
@app.post("/api/stt/upload")
def stt_upload(file: UploadFile = File(...), language: Optional[str] = Form(None), user: dict = Depends(get_current_user)):
    if language and language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported language")

    job_id = str(uuid.uuid4())
    # Save file
    filename = f"{job_id}_{file.filename}"
    dest_path = os.path.join(AUDIO_DIR, filename)
    with open(dest_path, "wb") as f:
        f.write(file.file.read())

    job_doc = {
        "_id": job_id,
        "user_id": user["_id"],
        "type": "stt",
        "status": "completed",  # placeholder
        "params": {"filename": filename, "language": language},
        "result": {
            "transcript": "This is a placeholder transcript. Replace with model-server transcription.",
            "language": language or "auto",
            "segments": [],
        },
        "created_at": datetime.now(timezone.utc),
        "completed_at": datetime.now(timezone.utc),
    }
    jobs_insert(job_doc)

    return {"jobId": job_id}

@app.get("/api/stt/job/{job_id}")
def get_stt_job(job_id: str, user: dict = Depends(get_current_user)):
    job = jobs_find_one({"_id": job_id, "user_id": user["_id"], "type": "stt"})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {
        "status": job["status"],
        "progress": 1.0 if job["status"] == "completed" else 0.2,
        "transcript": job["result"].get("transcript"),
        "language": job["result"].get("language"),
        "segments": job["result"].get("segments", []),
    }

# ----------------------------
# TTS synth - placeholder generates a tiny WAV beep file
# ----------------------------
import wave
import struct
import math

def _generate_beep(target_path: str, duration_s: float = 0.5, freq: float = 440.0, sample_rate: int = 16000):
    n_samples = int(duration_s * sample_rate)
    with wave.open(target_path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        for i in range(n_samples):
            value = int(32767.0 * 0.2 * math.sin(2 * math.pi * freq * (i / sample_rate)))
            data = struct.pack('<h', value)
            wf.writeframesraw(data)

@app.post("/api/tts/synthesize")
def tts_synthesize(req: TTSRequest, user: dict = Depends(get_current_user)):
    if req.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail="Unsupported language")
    job_id = str(uuid.uuid4())
    out_filename = f"{job_id}.wav"
    out_path = os.path.join(TTS_DIR, out_filename)
    # Placeholder: generate simple beep as synthesized audio
    _generate_beep(out_path, duration_s=min(2.0, max(0.5, len(req.text) / 50.0)))

    job_doc = {
        "_id": job_id,
        "user_id": user["_id"],
        "type": "tts",
        "status": "completed",
        "params": req.model_dump(),
        "result": {"url_to_audio": f"/api/tts/audio/{out_filename}"},
        "created_at": datetime.now(timezone.utc),
        "completed_at": datetime.now(timezone.utc),
    }
    jobs_insert(job_doc)
    return {"jobId": job_id}

@app.get("/api/tts/job/{job_id}")
def tts_job(job_id: str, user: dict = Depends(get_current_user)):
    job = jobs_find_one({"_id": job_id, "user_id": user["_id"], "type": "tts"})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"status": job["status"], "url_to_audio": job["result"].get("url_to_audio")}

@app.get("/api/tts/audio/{filename}")
def get_tts_audio(filename: str, user: dict = Depends(get_current_user)):
    path = os.path.join(TTS_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path, media_type="audio/wav")

# ----------------------------
# Admin endpoints (basic)
# ----------------------------
@app.get("/api/admin/jobs")
def admin_jobs(status: Optional[str] = None, user: dict = Depends(get_current_user)):
    # Simple RBAC: role field on user
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Forbidden")
    q: Dict[str, Any] = {}
    if status:
        q["status"] = status
    jobs = jobs_find(q)
    return jobs

@app.post("/api/admin/voice")
def admin_add_voice(name: str = Form(...), language: str = Form(...), gender: str = Form(...), model_ref: str = Form(None), user: dict = Depends(get_current_user)):
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Forbidden")
    vid = str(uuid.uuid4())
    vdoc = {"id": vid, "name": name, "language": language, "gender": gender, "model_ref": model_ref}
    voices_upsert_by_id(vdoc)
    return {"ok": True, "voice": vdoc}

# ----------------------------
# WebSocket placeholder for streaming STT
# ----------------------------
class ConnectionManager:
    def __init__(self):
        self.active: Dict[str, WebSocket] = {}

    async def connect(self, job_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active[job_id] = websocket

    def disconnect(self, job_id: str):
        self.active.pop(job_id, None)

manager = ConnectionManager()

@app.websocket("/api/stt/stream")
async def stt_stream(websocket: WebSocket):
    # For demo, accept without auth; in prod, validate query token
    job_id = str(uuid.uuid4())
    await manager.connect(job_id, websocket)
    try:
        # Send interim messages as placeholder
        await websocket.send_json({"type": "status", "message": "connected", "jobId": job_id})
        partial = ""
        while True:
            _ = await websocket.receive_bytes()
            partial = "..."  # placeholder
            await websocket.send_json({"type": "partial", "text": partial})
    except WebSocketDisconnect:
        manager.disconnect(job_id)

# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
