# recogniser/video_processor.py ------------------------------------
import cv2
import os
import json
import uuid
import pathlib
import datetime
import sys
import subprocess
import numpy as np

# Allow root-level imports when running from project dir
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from db import get_db
from models import Session as DBSession, Attendance, AttendanceLog
from recogniser.insightface_wrapper import FaceMatcher
from config import cfg


def _get_video_info_ffmpeg(video_path: str) -> dict:
    """Get video metadata using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_format", "-show_streams", video_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")
    
    import json as json_mod
    data = json_mod.loads(result.stdout)
    
    video_stream = None
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            video_stream = stream
            break
    
    if not video_stream:
        raise RuntimeError("No video stream found")
    
    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))
    
    # Parse frame rate (can be "30/1" or "29.97")
    fps_str = video_stream.get("r_frame_rate", "30/1")
    if "/" in fps_str:
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) > 0 else 30.0
    else:
        fps = float(fps_str)
    
    # Get duration
    duration = float(data.get("format", {}).get("duration", 0))
    total_frames = int(duration * fps) if duration > 0 else 0
    
    return {
        "width": width,
        "height": height,
        "fps": fps,
        "duration": duration,
        "total_frames": total_frames
    }


def _extract_frame_ffmpeg(video_path: str, timestamp_sec: float, width: int, height: int) -> np.ndarray | None:
    """Extract a single frame at a specific timestamp using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-ss", str(timestamp_sec),
        "-i", video_path,
        "-vframes", "1",
        "-f", "image2pipe",
        "-pix_fmt", "bgr24",
        "-vcodec", "rawvideo",
        "-"
    ]
    
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0 or len(result.stdout) == 0:
        return None
    
    if width == 0 or height == 0:
        return None
    
    expected_size = width * height * 3
    if len(result.stdout) < expected_size:
        return None
    
    frame = np.frombuffer(result.stdout[:expected_size], dtype=np.uint8)
    frame = frame.reshape((height, width, 3))
    return frame.copy()


def _can_use_opencv(video_path: str) -> bool:
    """Check if OpenCV can properly read this video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Try to actually read a frame
    ret, frame = cap.read()
    cap.release()
    
    return ret and frame is not None and width > 0 and height > 0


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

    # Check if OpenCV can read this video, otherwise use ffmpeg
    use_ffmpeg = not _can_use_opencv(video_path)
    
    if use_ffmpeg:
        print(f"[INFO] OpenCV cannot read this video, using ffmpeg fallback")
        video_info = _get_video_info_ffmpeg(video_path)
        fps = video_info["fps"]
        total_frames = video_info["total_frames"]
        width = video_info["width"]
        height = video_info["height"]
        print(f"[INFO] Video: {width}x{height}, {fps:.2f}fps, {total_frames} frames")
        cap = None
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[INFO] Using OpenCV: {width}x{height}, {fps:.2f}fps, {total_frames} frames")

    step = max(1, int(round(interval * fps)))
    frame_idxs = list(range(0, total_frames, step))[:max_frames]

    if not frame_idxs:
        if cap:
            cap.release()
        return {"present_ids": set(), "processed_frames": 0}

    matcher = FaceMatcher(presence_threshold=threshold)
    
    # dictionary: student_id -> { 'conf': float, 'bbox': list, 'frame': np.ndarray, 'fno': int }
    best_detections: dict[int, dict] = {}

    audit_root = pathlib.Path(cfg.get("audit_frames_folder", "stored_frames"))
    audit_root.mkdir(parents=True, exist_ok=True)

    for i, fno in enumerate(frame_idxs):
        # Extract frame either via OpenCV or ffmpeg
        if use_ffmpeg:
            timestamp_sec = fno / fps
            frame = _extract_frame_ffmpeg(video_path, timestamp_sec, width, height)
            ok = frame is not None
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
            ok, frame = cap.read()
        
        if not ok or frame is None:
            progress_callback(int((i + 1) / len(frame_idxs) * 100))
            continue

        results = matcher.detect_all(frame)

        for res in results:
            stu_id = res['student_id']
            conf = res['confidence']
            bbox = res['bbox']

            if stu_id is not None:
                # Update best detection if current one is better
                if stu_id not in best_detections or conf > best_detections[stu_id]['conf']:
                    best_detections[stu_id] = {
                        'conf': conf,
                        'bbox': bbox,
                        'frame': frame.copy(),
                        'fno': fno
                    }

        progress_callback(int((i + 1) / len(frame_idxs) * 100))

    # ── After scanning all frames, persist the BEST detections ───────────────
    present_ids: set[int] = set()
    saved_paths: list[str] = []

    with get_db() as db:
        sess = db.query(DBSession).filter(DBSession.id == session_id).first()
        if not sess:
            if cap:
                cap.release()
            raise RuntimeError(f"Session {session_id} not found in DB")

        # ── RESET SESSION: Clear existing results to ensure a fresh start ──────
        # 1. Delete old face crops from disk
        old_logs = db.query(AttendanceLog).filter(AttendanceLog.session_id == session_id).all()
        for l in old_logs:
            if l.frame_path and os.path.isfile(l.frame_path):
                try: os.remove(l.frame_path)
                except: pass
        
        # 2. Clear DB tables
        db.query(AttendanceLog).filter(AttendanceLog.session_id == session_id).delete()
        db.query(Attendance).filter(Attendance.session_id == session_id).delete()
        db.flush()

        # Now already_present will be empty since we just cleared it
        already_present = set()

        for stu_id, info in best_detections.items():
            conf = info['conf']
            bbox = info['bbox']
            frame = info['frame']
            fno = info['fno']

            # ── Crop Face ─────────────────────────────────────────────
            x1, y1, x2, y2 = bbox
            h, w = frame.shape[:2]
            
            # Add some padding (20%)
            pad_w = int((x2 - x1) * 0.2)
            pad_h = int((y2 - y1) * 0.2)
            
            crop_x1 = max(0, x1 - pad_w)
            crop_y1 = max(0, y1 - pad_h)
            crop_x2 = min(w, x2 + pad_w)
            crop_y2 = min(h, y2 + pad_h)
            
            face_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
            
            # Annotate crop (optional but helpful)
            # Re-calculating bbox relative to crop for display
            rel_x1, rel_y1 = x1 - crop_x1, y1 - crop_y1
            rel_x2, rel_y2 = x2 - crop_x1, y2 - crop_y1
            cv2.rectangle(face_crop, (rel_x1, rel_y1), (rel_x2, rel_y2), (0, 200, 0), 2)

            img_name = f"{uuid.uuid4().hex}.jpg"
            img_path = str(audit_root / img_name)
            cv2.imwrite(img_path, face_crop)
            saved_paths.append(img_path)

            # ── persistence ───────────────────────────────────────────
            # Add to Log
            db.add(AttendanceLog(
                session_id=session_id,
                student_id=stu_id,
                frame_path=img_path,
                frame_ts=fno / fps,
                confidence=conf,
                bbox=json.dumps(bbox),
            ))

            # Add to Attendance
            if stu_id not in already_present:
                db.add(Attendance(
                    session_id=session_id,
                    student_id=stu_id,
                    first_seen=datetime.datetime.utcnow(),
                    confidence=conf,
                ))
            
            present_ids.add(stu_id)

    if cap:
        cap.release()

    # Optional cleanup
    if delete_frames_after:
        for p in saved_paths:
            try:
                os.remove(p)
            except OSError:
                pass

    return {"present_ids": present_ids, "processed_frames": len(frame_idxs)}
