import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from .matcher import FaceMatcher
from .attendance import AttendanceLogger


class FaceAttendanceSystem:
    """
    InsightFace-powered attendance pipeline.

    Flow:
        Frame → RetinaFace (detect) → ArcFace (512-dim embedding)
             → Cosine similarity vs. SQLite gallery → AttendanceLogger

    No training step required — just enroll faces once.
    """

    def __init__(self,
                 model_name='buffalo_l',
                 db_path='models/gallery.db',
                 attendance_file='attendance.csv',
                 threshold=0.45,
                 ctx_id=0):
        """
        Args:
            model_name (str): InsightFace model pack ('buffalo_l' = ArcFace + RetinaFace).
            db_path (str): Path to SQLite gallery DB.
            attendance_file (str): Path to attendance CSV.
            threshold (float): Cosine similarity threshold (0.45–0.55 for ArcFace).
            ctx_id (int): 0 = CPU, -1 = GPU.
        """
        print("[INFO] Loading InsightFace model (first run downloads ~200 MB)...")
        self.app = FaceAnalysis(
            name=model_name,
            providers=['CPUExecutionProvider']
        )
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))

        self.matcher = FaceMatcher(db_path=db_path, threshold=threshold)
        self.attendance = AttendanceLogger(file_path=attendance_file)

        # is_trained is True when at least one person is enrolled
        self.is_trained = self.matcher.is_fitted
        print(f"[INFO] System ready. Enrolled: {self.matcher.get_enrolled_names()}")

    # ------------------------------------------------------------------
    # Enrollment (replaces the old train() method)
    # ------------------------------------------------------------------

    def enroll_person(self, person_id, person_name, n_images=12, source=0):
        """
        Capture `n_images` frames from `source`, extract ArcFace embeddings,
        average them, and store the mean in the SQLite gallery.

        Args:
            person_id (int): Unique integer ID.
            person_name (str): Human-readable name.
            n_images (int): Number of face samples to capture.
            source (int|str): Webcam index or RTSP URL.
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        embeddings = []
        saved = 0
        print(f"[ENROLL] Starting enrollment for {person_name} (ID={person_id})...")

        while saved < n_images:
            ret, frame = cap.read()
            if not ret:
                continue

            faces = self.app.get(frame)

            if not faces:
                cv2.putText(frame, "No face detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                # Use the largest face
                face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) *
                                                 (f.bbox[3] - f.bbox[1]))
                emb = face.embedding.astype(np.float32)
                embeddings.append(emb)
                saved += 1

                x1, y1, x2, y2 = map(int, face.bbox[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Saved {saved}/{n_images}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow(f"Enroll – {person_name}", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to abort
                break

        cap.release()
        cv2.destroyAllWindows()

        if not embeddings:
            raise RuntimeError("No faces captured — enrollment aborted.")

        mean_emb = np.mean(np.stack(embeddings, axis=0), axis=0).astype(np.float32)
        self.matcher.enroll(person_id, person_name, mean_emb)
        self.is_trained = True
        print(f"[ENROLL] Done. {len(embeddings)} samples averaged for {person_name}.")

    def enroll_from_dataset(self, dataset_dir):
        """
        Build the gallery from a folder of images.
        Expected structure: dataset_dir/<person_name>/<image>.jpg

        Each person gets one mean embedding computed from all their images.

        Args:
            dataset_dir (str): Root directory of the dataset.
        """
        print(f"[ENROLL] Building gallery from dataset: {dataset_dir}")
        person_id = 0

        for person_name in sorted(os.listdir(dataset_dir)):
            person_dir = os.path.join(dataset_dir, person_name)
            if not os.path.isdir(person_dir):
                continue

            print(f"  Processing: {person_name}")
            embeddings = []

            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                img = cv2.imread(img_path)
                if img is None:
                    continue

                faces = self.app.get(img)
                if not faces:
                    # Fallback: try treating the whole image as a face crop
                    # (InsightFace may still extract an embedding if the crop is large enough)
                    continue

                face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) *
                                                 (f.bbox[3] - f.bbox[1]))
                if face.embedding is not None:
                    embeddings.append(face.embedding.astype(np.float32))

            if not embeddings:
                print(f"  [WARN] No valid faces found for {person_name} — skipping.")
                continue

            mean_emb = np.mean(np.stack(embeddings, axis=0), axis=0).astype(np.float32)
            self.matcher.enroll(person_id, person_name, mean_emb)
            person_id += 1

        self.is_trained = self.matcher.is_fitted
        print(f"[ENROLL] Gallery built. {person_id} person(s) enrolled.")

    # Alias kept for backward compatibility with app.py's "Train Model" button
    def train(self, dataset_dir):
        self.enroll_from_dataset(dataset_dir)

    # ------------------------------------------------------------------
    # Recognition
    # ------------------------------------------------------------------

    def recognize(self, frame):
        """
        Run recognition on a live frame and draw results.

        Args:
            frame (np.ndarray): BGR image.

        Returns:
            np.ndarray: Annotated frame.
        """
        if not self.is_trained:
            cv2.putText(frame, "No faces enrolled yet", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return frame

        faces = self.app.get(frame)

        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox[:4])
            emb = face.embedding.astype(np.float32)

            pid, name, sim = self.matcher.predict(emb)

            if self.matcher.is_match(sim):
                color = (0, 255, 0)
                self.attendance.mark(name)
            else:
                name = "Unknown"
                color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{name} ({sim:.2f})",
                        (x1, max(y1 - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return frame

    def recognize_image(self, image, write_to_db=True):
        """
        Recognise the largest face in a single image.

        Args:
            image (np.ndarray): BGR image.
            write_to_db (bool): Whether to log attendance.

        Returns:
            dict | None: {'name', 'confidence', 'box'} or None.
        """
        if not self.is_trained:
            return None

        faces = self.app.get(image)
        if not faces:
            return None

        # Largest face
        face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) *
                                         (f.bbox[3] - f.bbox[1]))
        emb = face.embedding.astype(np.float32)
        x1, y1, x2, y2 = map(int, face.bbox[:4])

        pid, name, sim = self.matcher.predict(emb)

        if self.matcher.is_match(sim):
            if write_to_db:
                self.attendance.mark(name)
            return {
                'name': name,
                'confidence': sim,
                'box': (x1, y1, x2 - x1, y2 - y1)
            }

        return None

    def process_images_dir(self, img_dir, dry_run=False):
        """
        Batch-process a directory of images.

        Args:
            img_dir (str): Path to directory of images.
            dry_run (bool): If True, prints results without writing to DB.
        """
        from pathlib import Path

        img_path = Path(img_dir)
        if not img_path.is_dir():
            raise ValueError(f"{img_dir} is not a directory")

        if not self.is_trained:
            print("[ERROR] No faces enrolled. Run enrollment first.")
            return

        supported = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        files = [p for p in img_path.iterdir() if p.suffix.lower() in supported]

        if not files:
            print(f"No image files found in {img_dir}")
            return

        print(f"\nProcessing {len(files)} image(s) from {img_dir}")
        print(f"Dry run: {dry_run}\n" + "-" * 60)

        for f in files:
            img = cv2.imread(str(f))
            if img is None:
                print(f"✖ {f.name} → Unable to read")
                continue

            result = self.recognize_image(img, write_to_db=not dry_run)

            if result:
                print(f"✔ {f.name} → {result['name']} (confidence: {result['confidence']:.2f})")
            else:
                print(f"✖ {f.name} → No face detected or below threshold")

        print("-" * 60)
        print(f"\nDone. Check {self.attendance.file_path} for results.\n")

    # ------------------------------------------------------------------
    # Backward-compat shims (old pipeline used load_models / reducer)
    # ------------------------------------------------------------------

    def load_models(self):
        """Reload gallery from SQLite (in case DB was updated externally)."""
        self.matcher.load(None)
        self.is_trained = self.matcher.is_fitted
