"""
Wrapper for InsightFace models to perform face detection and embedding extraction.
"""
import numpy as np
import insightface
import sys, os
import cv2

# Allow root-level imports (db, models, config) when running from project dir
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from db import get_db
from models import Student
from config import cfg


class FaceMatcher:
    """
    Loads InsightFace detection + ArcFace embedding model *once*.

    Provides:
      match(bgr_img) -> (student_id|None, student_name|None, cosine_similarity)
    """

    def __init__(self, presence_threshold: float = 0.45, det_size: tuple = None):
        # Get detection size from config or use default for better detection
        if det_size is None:
            det_size = tuple(cfg.get("detection_size", [640, 640]))
        
        # Load model – CPU by default. For GPU: ctx_id=0, providers=['CUDAExecutionProvider']
        self.app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=-1, det_size=det_size)
        self.det_size = det_size

        self.threshold = presence_threshold
        self._load_gallery()

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

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    @staticmethod
    def _normalise(v: np.ndarray) -> np.ndarray:
        """L2-normalise so cosine similarity == dot product."""
        norm = np.linalg.norm(v)
        return v / (norm + 1e-12)

    def get_embedding(self, bgr_img: np.ndarray) -> np.ndarray | None:
        """Return the L2-normalised ArcFace embedding for the largest face, or None."""
        faces = self.app.get(bgr_img)
        if not faces:
            return None
        
        # Pick the largest face
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

    def match(self, bgr_img: np.ndarray) -> tuple:
        """
        Detect faces in bgr_img, pick the largest, match against gallery.

        Returns
        -------
        (student_id|None, student_name|None, cosine_similarity, bbox|None)
        where bbox = [x1, y1, x2, y2] (integers) in original image coordinates, or None if no face found
        """
        if not self.id2emb:
            return None, None, 0.0, None

        faces = self.app.get(bgr_img)
        if not faces:
            return None, None, 0.0, None

        # Pick the largest face
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        
        # Get bounding box
        bbox = [int(v) for v in face.bbox[:4]]
        emb  = self._normalise(face.embedding.astype(np.float32))

        best_id, best_name, best_score = None, None, -1.0
        for sid, stored_emb in self.id2emb.items():
            score = self._cosine(emb, stored_emb)
            if score > best_score:
                best_score, best_id = score, sid
                best_name = self.id2name[sid]

        if best_score >= self.threshold:
            return best_id, best_name, best_score, bbox
        return None, None, best_score, bbox

    def detect_all(self, bgr_img: np.ndarray) -> list:
        """
        Detect all faces in an image and match against gallery.
        
        Returns
        -------
        list of dicts with keys: student_id, student_name, confidence, bbox
        """
        if not self.id2emb:
            return []

        faces = self.app.get(bgr_img)
        if not faces:
            return []

        results = []
        for face in faces:
            bbox = [int(v) for v in face.bbox[:4]]
            emb = self._normalise(face.embedding.astype(np.float32))

            best_id, best_name, best_score = None, None, -1.0
            for sid, stored_emb in self.id2emb.items():
                score = self._cosine(emb, stored_emb)
                if score > best_score:
                    best_score, best_id = score, sid
                    best_name = self.id2name.get(sid)

            if best_score >= self.threshold:
                results.append({
                    'student_id': best_id,
                    'student_name': best_name,
                    'confidence': best_score,
                    'bbox': bbox
                })
            else:
                results.append({
                    'student_id': None,
                    'student_name': 'Unknown',
                    'confidence': best_score,
                    'bbox': bbox
                })

        return results
