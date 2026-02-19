import numpy as np


class FeatureExtractor:
    """
    ArcFace embedding extractor (via InsightFace).
    Replaces the old LBP histogram extractor.

    InsightFace already computes the 512-dim ArcFace embedding inside
    FaceAnalysis.get(), so this class is a thin wrapper that pulls the
    embedding from a Face object and normalises it.
    """

    def extract(self, face):
        """
        Extract a normalised 512-dim ArcFace embedding from an InsightFace Face object.

        Args:
            face: InsightFace Face object (returned by FaceAnalysis.get()).

        Returns:
            numpy.ndarray: Normalised embedding of shape (512,), or None on failure.
        """
        if face is None:
            return None

        emb = face.embedding
        if emb is None:
            return None

        emb = emb.astype(np.float32)

        # L2-normalise so cosine similarity == dot product
        norm = np.linalg.norm(emb)
        if norm < 1e-6:
            return None

        return emb / norm

    def extract_from_list(self, faces):
        """
        Extract and average embeddings from a list of Face objects.
        Useful during enrollment to build a robust mean embedding.

        Args:
            faces (list): List of InsightFace Face objects.

        Returns:
            numpy.ndarray: Mean normalised embedding, or None if list is empty.
        """
        embeddings = [self.extract(f) for f in faces if f is not None]
        embeddings = [e for e in embeddings if e is not None]

        if not embeddings:
            return None

        mean_emb = np.mean(np.stack(embeddings, axis=0), axis=0).astype(np.float32)
        norm = np.linalg.norm(mean_emb)
        if norm < 1e-6:
            return None

        return mean_emb / norm
