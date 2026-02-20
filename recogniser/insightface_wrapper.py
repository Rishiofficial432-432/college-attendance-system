# recogniser/insightface_wrapper.py --------------------------------
import numpy as np
import insightface
import sys, os

# Allow root-level imports (db, models, config) when running from project dir
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from db import get_db
from models import Student


class FaceMatcher:
    """
    Loads InsightFace detection + ArcFace embedding model *once*.

    Provides:
      match(bgr_img) -> (student_id|None, student_name|None, cosine_similarity)
    """

    def __init__(self, presence_threshold: float = 0.45):
        # Load model – CPU by default.  For GPU: ctx_id=0, providers=['CUDAExecutionProvider']
        self.app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=-1)   # -1 = CPU

        self.threshold = presence_threshold
        self._load_gallery()

    # ------------------------------------------------------------------
    def _load_gallery(self):
        """Reload embeddings from DB into memory."""
        self.id2emb: dict[int, np.ndarray] = {}
        self.id2name: dict[int, str] = {}
        with get_db() as db:
            for stu in db.query(Student).all():
                self.id2emb[stu.id] = np.frombuffer(stu.embedding, dtype=np.float32).copy()
                self.id2name[stu.id] = stu.name

    def reload(self):
        """Call after enrolling a new student so in-memory gallery is fresh."""
        self._load_gallery()

    # ------------------------------------------------------------------
    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    # ------------------------------------------------------------------
    def get_embedding(self, bgr_img: np.ndarray) -> np.ndarray | None:
        """Return the ArcFace embedding for the largest face in the image, or None."""
        faces = self.app.get(bgr_img)
        if not faces:
            return None
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        return face.embedding.astype(np.float32)

    # ------------------------------------------------------------------
    def match(self, bgr_img: np.ndarray) -> tuple:
        """
        Detect faces in bgr_img, pick the largest, match against gallery.

        Returns
        -------
        (student_id|None, student_name|None, cosine_similarity, bbox|None)
        where bbox = [x1, y1, x2, y2] (integers) or None if no face found
        """
        if not self.id2emb:
            return None, None, 0.0, None

        faces = self.app.get(bgr_img)
        if not faces:
            return None, None, 0.0, None

        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        bbox = [int(v) for v in face.bbox[:4]]   # [x1, y1, x2, y2]
        emb  = face.embedding.astype(np.float32)

        best_id, best_name, best_score = None, None, -1.0
        for sid, stored_emb in self.id2emb.items():
            score = self._cosine(emb, stored_emb)
            if score > best_score:
                best_score, best_id = score, sid
                best_name = self.id2name[sid]

        if best_score >= self.threshold:
            return best_id, best_name, best_score, bbox
        return None, None, best_score, bbox

