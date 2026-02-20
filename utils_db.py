# utils_db.py -------------------------------------------------------
# Safe query helpers – every function opens its own short-lived session,
# materialises the data into plain Python objects, and closes the session.
# These dicts / DataFrames are detached from SQLAlchemy and safe to use
# anywhere in Streamlit without triggering DetachedInstanceError.

import pandas as pd
import datetime
from db import get_db
from models import Student, Session as DBSession, Attendance, AttendanceLog


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


def get_sessions_for_date(date_obj) -> list[dict]:
    """Return sessions on a given date as plain dicts."""
    with get_db() as db:
        sessions = (db.query(DBSession)
                      .filter(DBSession.date == date_obj)
                      .order_by(DBSession.start_time)
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
        return pd.DataFrame(data)


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
