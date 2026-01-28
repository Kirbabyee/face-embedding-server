from fastapi import FastAPI, UploadFile, File, Form
from deepface import DeepFace
from supabase import create_client
import numpy as np
import tempfile, os
from datetime import datetime

app = FastAPI()

@app.get("/")
def health():
    return {"status": "ok"}

# ✅ read from ENV (Railway Variables)
SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

if not SUPABASE_URL or not SUPABASE_KEY:
    # optional: this will help you see the problem in logs if env missing
    print("⚠️ Missing SUPABASE_URL / SUPABASE_KEY env vars")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

MODEL = "ArcFace"
THRESH = 0.35

def embed_image(path: str) -> np.ndarray:
    rep = DeepFace.represent(
        img_path=path,
        model_name=MODEL,
        detector_backend="opencv",
        enforce_detection=True
    )
    emb = rep[0]["embedding"] if isinstance(rep, list) else rep["embedding"]
    return np.array(emb, dtype=np.float32)

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(1.0 - np.dot(a, b))

@app.post("/enroll")
async def enroll(
    user_id: str = Form(...),
    image: UploadFile = File(...)
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await image.read())
        path = tmp.name

    try:
        embedding = embed_image(path)

        supabase.table("face_embeddings").insert({
            "student_id": user_id,
            "embedding": embedding.tolist()
        }).execute()

        supabase.table("students").update({
            "face_registered_at": datetime.utcnow().isoformat()
        }).eq("id", user_id).execute()

        return {"ok": True}

    except Exception as e:
        return {"ok": False, "error": str(e)}

    finally:
        try:
            os.unlink(path)
        except:
            pass

@app.post("/verify")
async def verify(
    user_id: str = Form(...),
    image: UploadFile = File(...)
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await image.read())
        path = tmp.name

    try:
        rows = (
            supabase
            .table("face_embeddings")
            .select("embedding")
            .eq("student_id", user_id)
            .execute()
        )

        if not rows.data:
            return {"ok": False, "error": "not_enrolled"}

        stored = [np.array(r["embedding"], dtype=np.float32) for r in rows.data]
        probe = embed_image(path)

        best = min(cosine_distance(probe, ref) for ref in stored)

        return {
            "ok": True,
            "verified": best <= THRESH,
            "distance": best,
            "threshold": THRESH
        }

    except Exception as e:
        return {"ok": False, "error": str(e)}

    finally:
        try:
            os.unlink(path)
        except:
            pass
