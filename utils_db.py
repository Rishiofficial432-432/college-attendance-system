# utils_db.py -------------------------------------------------------
# Safe query helpers – every function opens its own short-lived session,
# materialises the data into plain Python objects, and closes the session.
# These dicts / DataFrames are detached from SQLAlchemy and safe to use
# anywhere in Streamlit without triggering DetachedInstanceError.

import pandas as pd
import datetime
from db import get_db
from models import Student, Session as DBSession, Attendance, AttendanceLog, Course


# ── Courses ─────────────────────────────────────────────────────────────────

def get_all_courses() -> list[dict]:
    """Return all courses as plain dicts."""
    with get_db() as db:
        rows = db.query(Course).order_by(Course.code).all()
        return [{"id": c.id, "code": c.code, "name": c.name} for c in rows]


def get_course_count() -> int:
    with get_db() as db:
        return db.query(Course).count()


# ── Students ────────────────────────────────────────────────────────────────

def get_all_students() -> list[dict]:
    """Return all students as plain dicts (safe after session closes)."""
    with get_db() as db:
        rows = db.query(Student.id, Student.name).order_by(Student.name).all()
        return [{"ID": sid, "Name": name} for sid, name in rows]


def get_student_count() -> int:
    with get_db() as db:
        return db.query(Student).count()


# ── Sessions ────────────────────────────────────────────────────────────────

def get_distinct_dates() -> list:
    """Return list of distinct session dates, newest first."""
    with get_db() as db:
        rows = db.query(DBSession.date).distinct().order_by(DBSession.date.desc()).all()
        return [r[0] for r in rows]


def get_sessions_for_course(course_id: int) -> list[dict]:
    """Return all sessions for a specific course as plain dicts."""
    with get_db() as db:
        sessions = (db.query(DBSession)
                      .filter(DBSession.course_id == course_id)
                      .order_by(DBSession.date.desc(), DBSession.start_time)
                      .all())
        return [
            {
                "id":         s.id,
                "title":      s.title,
                "date":       s.date,
                "start_time": s.start_time,
                "end_time":   s.end_time,
                "description": s.description,
            }
            for s in sessions
        ]


def get_all_sessions() -> list[dict]:
    """Return all sessions as plain dicts, newest date first."""
    with get_db() as db:
        sessions = db.query(DBSession).order_by(DBSession.date.desc()).all()
        return [
            {
                "id":         s.id,
                "title":      s.title,
                "date":       str(s.date),
                "start_time": str(s.start_time),
                "end_time":   str(s.end_time),
            }
            for s in sessions
        ]


# ── Attendance ──────────────────────────────────────────────────────────────

def get_attendance_dataframe(session_id: int) -> pd.DataFrame:
    """
    Build an attendance DataFrame for a session.
    All DB access happens inside the session; returns a plain DataFrame.
    """
    with get_db() as db:
        all_students = db.query(Student).order_by(Student.name).all()
        present_rows = (
            db.query(Attendance)
            .filter(Attendance.session_id == session_id)
            .all()
        )
        present_ids   = {p.student_id for p in present_rows}
        first_seen_map = {p.student_id: p.first_seen for p in present_rows}
        conf_map       = {p.student_id: p.confidence  for p in present_rows}

        data = []
        for stu in all_students:
            if stu.id in present_ids:
                status = "Present"
                seen   = first_seen_map[stu.id].strftime("%H:%M:%S") \
                         if first_seen_map.get(stu.id) else ""
                conf   = f"{conf_map.get(stu.id, 0):.3f}"
            else:
                status = "Absent"
                seen   = ""
                conf   = ""
            data.append({
                "ID":         stu.id,
                "Name":       stu.name,
                "Status":     status,
                "First Seen": seen,
                "Confidence": conf,
            })
        return pd.DataFrame(data, columns=["ID", "Name", "Status", "First Seen", "Confidence"])



# ── Audit log ───────────────────────────────────────────────────────────────

def get_audit_logs(session_id: int) -> list[dict]:
    """
    Return audit-log rows for a session as plain dicts (frame_path, ts, conf,
    student_name). Safe to iterate in Streamlit without a live session.
    """
    with get_db() as db:
        # Eager-load student name in the same query to avoid lazy-load issues
        logs = (
            db.query(AttendanceLog)
            .filter(AttendanceLog.session_id == session_id)
            .order_by(AttendanceLog.frame_ts)
            .all()
        )
        # Fetch student names in the same session
        student_names = {s.id: s.name for s in db.query(Student).all()}

        return [
            {
                "id":           l.id,
                "frame_ts":     l.frame_ts,
                "frame_path":   l.frame_path,
                "confidence":   l.confidence,
                "student_id":   l.student_id,
                "student_name": student_names.get(l.student_id, "Unknown")
                                if l.student_id else "No match",
            }
            for l in logs
        ]


def get_audit_log_count(session_id: int) -> int:
    with get_db() as db:
        return db.query(AttendanceLog).filter(AttendanceLog.session_id == session_id).count()


# ── Delete helpers ───────────────────────────────────────────────────────────

def delete_student(student_id: int, dataset_base_dir: str = "attendance_system/dataset") -> str:
    """
    Remove a student from the DB (and their attendance records) and
    delete their photo folder on disk.  Returns the student name.
    """
    import shutil, os
    with get_db() as db:
        stu = db.query(Student).filter(Student.id == student_id).first()
        if not stu:
            return "Unknown"
        name = stu.name
        # Delete attendance rows
        db.query(Attendance).filter(Attendance.student_id == student_id).delete()
        # Nullify audit-log references (keep the frame images)
        db.query(AttendanceLog).filter(AttendanceLog.student_id == student_id)\
                               .update({"student_id": None})
        db.delete(stu)

    # Delete photo folder on disk
    folder = os.path.join(dataset_base_dir, name)
    if os.path.isdir(folder):
        shutil.rmtree(folder)

    return name


def delete_audit_logs(session_id: int, delete_images: bool = True) -> int:
    """
    Delete all audit-log rows for a session.
    If delete_images=True, also remove the frame JPEG files from disk.
    Returns the number of rows deleted.
    """
    import os
    with get_db() as db:
        logs = db.query(AttendanceLog)\
                  .filter(AttendanceLog.session_id == session_id)\
                  .all()
        paths = [l.frame_path for l in logs]
        count = len(logs)
        db.query(AttendanceLog).filter(AttendanceLog.session_id == session_id).delete()

    if delete_images:
        for p in paths:
            try:
                os.remove(p)
            except OSError:
                pass

    return count

