from fastapi import FastAPI, UploadFile, File, Form
from deepface import DeepFace
from supabase import create_client
import numpy as np
import tempfile, os
from datetime import datetime


from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def health():
    return {"status": "ok"}


SUPABASE_URL = "https://ucfundmbawljngzowzgd.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InVjZnVuZG1iYXdsam5nem93emdkIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2NzU5Njk0NCwiZXhwIjoyMDgzMTcyOTQ0fQ.Skc7bnElRtJagyaL8JCieLR_5lFsTwOHteF9RE7pFmU"

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

MODEL = "ArcFace"
METRIC = "cosine"
THRESH = 0.35

def embed_image(path: str) -> np.ndarray:
    rep = DeepFace.represent(
        img_path=path,
        model_name=MODEL,
        detector_backend="opencv",  # avoid retinaface issues
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
    user_id: str = Form(...),   # student_id (UUID)
    image: UploadFile = File(...)
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await image.read())
        path = tmp.name

    try:
        embedding = embed_image(path)

        # 1️⃣ insert embedding
        supabase.table("face_embeddings").insert({
            "student_id": user_id,
            "embedding": embedding.tolist()
        }).execute()

        # 2️⃣ update students table
        supabase.table("students").update({
            "face_registered_at": datetime.utcnow().isoformat()
        }).eq("id", user_id).execute()

        return {"ok": True}

    except Exception as e:
        return {"ok": False, "error": str(e)}

    finally:
        os.unlink(path)


@app.post("/verify")
async def verify(
    user_id: str = Form(...),
    image: UploadFile = File(...)
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await image.read())
        path = tmp.name

    try:
        # 1️⃣ get stored embeddings
        rows = (
            supabase
            .table("face_embeddings")
            .select("embedding")
            .eq("student_id", user_id)
            .execute()
        )

        if not rows.data:
            return {"ok": False, "error": "not_enrolled"}

        stored = [
            np.array(r["embedding"], dtype=np.float32)
            for r in rows.data
        ]

        # 2️⃣ embed live image
        probe = embed_image(path)

        # 3️⃣ compare
        best = min(
            cosine_distance(probe, ref)
            for ref in stored
        )

        return {
            "ok": True,
            "verified": best <= THRESH,
            "distance": best,
            "threshold": THRESH
        }

    except Exception as e:
        return {"ok": False, "error": str(e)}

    finally:
        os.unlink(path)

