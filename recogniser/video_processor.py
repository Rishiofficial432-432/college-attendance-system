# recogniser/video_processor.py ------------------------------------
import cv2
import os
import json
import uuid
import pathlib
import datetime
import sys

# Allow root-level imports when running from project dir
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from db import get_db
from models import Session as DBSession, Attendance, AttendanceLog
from recogniser.insightface_wrapper import FaceMatcher
from config import cfg


def process_video(
    video_path: str,
    session_id: int,
    progress_callback=lambda perc: None,
    delete_frames_after: bool = False,
) -> dict:
    """
    Sample a frame every `frame_interval_sec` seconds, run face recognition,
    and persist results to attendance + attendance_log.

    Parameters
    ----------
    video_path        : absolute path to the video file
    session_id        : PK of the target session row
    progress_callback : called with int 0-100 after each frame
    delete_frames_after: if True, remove stored frames after DB commit

    Returns
    -------
    dict with keys:
      present_ids       – set of student IDs marked present
      processed_frames  – number of frames actually sampled
    """
    interval  = cfg.get("frame_interval_sec", 5)
    max_frames = cfg.get("max_frames_to_process", 1000)
    threshold = cfg.get("presence_threshold", 0.45)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(interval * fps)))
    frame_idxs = list(range(0, total_frames, step))[:max_frames]

    if not frame_idxs:
        cap.release()
        return {"present_ids": set(), "processed_frames": 0}

    matcher = FaceMatcher(presence_threshold=threshold)
    present_ids: set[int] = set()

    audit_root = pathlib.Path(cfg.get("audit_frames_folder", "stored_frames"))
    audit_root.mkdir(parents=True, exist_ok=True)

    saved_paths: list[str] = []

    with get_db() as db:
        sess = db.query(DBSession).filter(DBSession.id == session_id).first()
        if not sess:
            cap.release()
            raise RuntimeError(f"Session {session_id} not found in DB")

        # ── Pre-load students already marked present for this session ──────────
        # Handles re-processing the same video: avoids UNIQUE constraint errors.
        existing = db.query(Attendance.student_id)\
                     .filter(Attendance.session_id == session_id)\
                     .all()
        present_ids = {row[0] for row in existing}

        for i, fno in enumerate(frame_idxs):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
            ok, frame = cap.read()
            if not ok:
                progress_callback(int((i + 1) / len(frame_idxs) * 100))
                continue

            stu_id, stu_name, conf, bbox = matcher.match(frame)

            # ── draw bounding box on a copy of the frame ────────────────────
            annotated = frame.copy()
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                color = (0, 200, 0) if stu_id is not None else (0, 0, 220)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"{stu_name} ({conf:.2f})" if stu_id else f"Unknown ({conf:.2f})"
                cv2.putText(annotated, label,
                            (x1, max(y1 - 8, 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # ── audit frame (annotated) ─────────────────────────────────────
            img_name = f"{uuid.uuid4().hex}.jpg"
            img_path = str(audit_root / img_name)
            cv2.imwrite(img_path, annotated)
            saved_paths.append(img_path)

            db.add(AttendanceLog(
                session_id=session_id,
                student_id=stu_id,
                frame_path=img_path,
                frame_ts=fno / fps,
                confidence=conf,
                bbox=json.dumps(bbox) if bbox else json.dumps([]),
            ))

            # ── first-time attendance (skip if already in DB) ───────────────
            if stu_id is not None and stu_id not in present_ids:
                try:
                    db.add(Attendance(
                        session_id=session_id,
                        student_id=stu_id,
                        first_seen=datetime.datetime.utcnow(),
                        confidence=conf,
                    ))
                    db.flush()           # catch constraint violations early
                    present_ids.add(stu_id)
                except Exception:
                    db.rollback()        # discard only this failed INSERT
                    present_ids.add(stu_id)

            progress_callback(int((i + 1) / len(frame_idxs) * 100))

        # commit happens at exit of context manager


    cap.release()

    # Optional cleanup
    if delete_frames_after:
        for p in saved_paths:
            try:
                os.remove(p)
            except OSError:
                pass

    return {"present_ids": present_ids, "processed_frames": len(frame_idxs)}
