import numpy as np
from insightface.app import FaceAnalysis


class FaceDetector:
    """
    InsightFace-based face detector using RetinaFace.
    Replaces the old YOLOv8 person detector.
    """

    def __init__(self, model_name='buffalo_l', det_size=(640, 640), ctx_id=0):
        """
        Initialize InsightFace detector.

        Args:
            model_name (str): InsightFace model pack. 'buffalo_l' = ArcFace + RetinaFace.
            det_size (tuple): Detection input size (width, height).
            ctx_id (int): 0 = CPU, -1 = GPU.
        """
        self.app = FaceAnalysis(
            name=model_name,
            allowed_modules=['detection', 'recognition']
        )
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)

    def detect(self, frame):
        """
        Detect faces in a frame.

        Args:
            frame (numpy.ndarray): BGR image.

        Returns:
            list[tuple]: Bounding boxes as (x, y, w, h).
            list[Face]: InsightFace Face objects (contain .embedding, .bbox, etc.).
        """
        faces = self.app.get(frame)

        boxes = []
        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox[:4])
            w, h = x2 - x1, y2 - y1
            boxes.append((x1, y1, w, h))

        return boxes, faces
