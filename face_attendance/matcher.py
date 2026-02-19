import numpy as np
import sqlite3
import os


class FaceMatcher:
    """
    Cosine-similarity face matcher backed by a SQLite gallery.
    Replaces the old SVM classifier — no training step required.

    Gallery table schema (auto-created):
        CREATE TABLE face_embeddings (
            person_id   INTEGER PRIMARY KEY,
            person_name TEXT    NOT NULL,
            embedding   BLOB    NOT NULL   -- 512 float32 values as raw bytes
        );
    """

    def __init__(self, db_path='models/gallery.db', threshold=0.45):
        """
        Args:
            db_path (str): Path to the SQLite DB file.
            threshold (float): Minimum cosine similarity to accept a match.
                               ArcFace embeddings work well at 0.45–0.55.
        """
        self.db_path = db_path
        self.threshold = threshold
        self._gallery = {}   # {person_id: (person_name, np.ndarray)}
        self.is_fitted = False

        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else '.', exist_ok=True)
        self._init_db()
        self._load_gallery()

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    def _init_db(self):
        """Create the face_embeddings table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS face_embeddings (
                person_id   INTEGER PRIMARY KEY,
                person_name TEXT    NOT NULL,
                embedding   BLOB    NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def _load_gallery(self):
        """Load all stored embeddings from SQLite into memory."""
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute("SELECT person_id, person_name, embedding FROM face_embeddings")
        rows = cur.fetchall()
        conn.close()

        self._gallery = {}
        for pid, name, blob in rows:
            emb = np.frombuffer(blob, dtype=np.float32).copy()
            self._gallery[pid] = (name, emb)

        self.is_fitted = len(self._gallery) > 0

    # ------------------------------------------------------------------
    # Enrollment
    # ------------------------------------------------------------------

    def enroll(self, person_id, person_name, embedding):
        """
        Store (or update) a person's mean embedding in the SQLite gallery.

        Args:
            person_id (int): Unique integer ID for this person.
            person_name (str): Human-readable name.
            embedding (np.ndarray): 512-dim normalised ArcFace embedding.
        """
        emb = embedding.astype(np.float32)
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO face_embeddings (person_id, person_name, embedding)
            VALUES (?, ?, ?)
        """, (person_id, person_name, emb.tobytes()))
        conn.commit()
        conn.close()

        # Update in-memory cache
        self._gallery[person_id] = (person_name, emb)
        self.is_fitted = True
        print(f"[GALLERY] Enrolled: {person_name} (ID={person_id})")

    def get_enrolled_names(self):
        """Return list of all enrolled person names."""
        return [name for (name, _) in self._gallery.values()]

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, embedding):
        """
        Find the closest match using cosine similarity.

        Args:
            embedding (np.ndarray): 512-dim normalised query embedding.

        Returns:
            int: Predicted person_id.
            str: Predicted person_name.
            float: Cosine similarity score (0–1).
        """
        if not self.is_fitted:
            raise RuntimeError("Gallery is empty. Enroll faces first.")

        best_id, best_name, best_score = None, "Unknown", -1.0
        for pid, (name, known_emb) in self._gallery.items():
            score = float(np.dot(embedding, known_emb) /
                          (np.linalg.norm(embedding) * np.linalg.norm(known_emb) + 1e-12))
            if score > best_score:
                best_score, best_id, best_name = score, pid, name

        return best_id, best_name, best_score

    def is_match(self, similarity):
        """Return True if similarity exceeds the threshold."""
        return similarity >= self.threshold

    # ------------------------------------------------------------------
    # Persistence shims (kept for API compatibility with pipeline.py)
    # ------------------------------------------------------------------

    def save(self, directory):
        """No-op: SQLite DB is already persistent at self.db_path."""
        pass

    def load(self, directory):
        """Reload gallery from SQLite (in case DB was updated externally)."""
        self._load_gallery()
