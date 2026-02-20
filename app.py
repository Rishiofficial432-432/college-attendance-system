# app.py -------------------------------------------------------------
import streamlit as st
import os
import uuid
import datetime
import json
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

# ── Local imports ──────────────────────────────────────────────────────────────
from db import init_db, get_db
from models import Student, Session as DBSession, Attendance, AttendanceLog
from config import cfg

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="College Attendance System",
    page_icon="🎓",
    layout="wide",
)

# ── One-time DB initialisation ────────────────────────────────────────────────
init_db()

# ── Cache heavy objects (loaded once per server process) ──────────────────────
@st.cache_resource
def get_matcher():
    from recogniser.insightface_wrapper import FaceMatcher
    return FaceMatcher(presence_threshold=cfg.get("presence_threshold", 0.45))


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background: linear-gradient(180deg,#1a1a2e,#16213e); }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    .metric-card { background:#1e293b; border-radius:12px; padding:1.2rem;
                   text-align:center; border:1px solid #334155; }
    .metric-num  { font-size:2.5rem; font-weight:700; color:#38bdf8; }
    .metric-lbl  { font-size:.9rem;  color:#94a3b8; margin-top:.3rem; }
    .present-badge { background:#16a34a20; color:#4ade80; border:1px solid #22c55e;
                     border-radius:6px; padding:2px 10px; font-size:.82rem; }
    .absent-badge  { background:#dc262620; color:#f87171; border:1px solid #ef4444;
                     border-radius:6px; padding:2px 10px; font-size:.82rem; }
    h1 { color:#e2e8f0 !important; }
    h2, h3 { color:#cbd5e1 !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=60)
st.sidebar.title("Attendance System")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["📊 Dashboard", "👤 Enrol Students", "🗓️ Create Session",
     "📽️ Upload Video", "📋 Audit Log"],
)

# ── Sidebar: enrolled students count ────────────────────────────────────────
with get_db() as d:
    enrolled_count = d.query(Student).count()
st.sidebar.markdown(f"**Enrolled students:** {enrolled_count}")

# ─────────────────────────────────────────────────────────────────────────────
# 1️⃣  ENROL STUDENTS
# ─────────────────────────────────────────────────────────────────────────────
if page == "👤 Enrol Students":
    st.title("👤 Enrol a New Student")
    st.write("Upload **one clear, well-lit portrait** of the student. The ArcFace embedding is computed and stored in SQLite.")

    col_a, col_b = st.columns(2)
    with col_a:
        name = st.text_input("Full name", placeholder="e.g. Rishi Sharma")
    with col_b:
        st.markdown("<br>", unsafe_allow_html=True)

    img_file = st.file_uploader("Portrait photo (JPEG / PNG)", type=["jpg", "jpeg", "png"])

    if img_file:
        preview_img = Image.open(img_file).convert("RGB")
        st.image(preview_img, caption="Preview", width=300)
        img_file.seek(0)

    if st.button("✅ Enrol Student", type="primary", disabled=not (name and img_file)):
        np_arr = np.frombuffer(img_file.read(), np.uint8)
        bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        with st.spinner("Loading face model & computing embedding…"):
            matcher = get_matcher()
            emb = matcher.get_embedding(bgr)

        if emb is None:
            st.error("❌ No face detected. Please upload a clearer portrait.")
        else:
            with get_db() as db:
                existing = db.query(Student).filter(Student.name == name).first()
                if existing:
                    st.warning(f"⚠️ **{name}** is already enrolled. Updating embedding…")
                    existing.embedding = emb.tobytes()
                else:
                    db.add(Student(name=name, embedding=emb.tobytes()))
            matcher.reload()
            st.success(f"✅ **{name}** enrolled successfully!")
            st.rerun()

    # ── Show enrolled list ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Currently Enrolled")
    with get_db() as db:
        students = db.query(Student).order_by(Student.name).all()

    if students:
        rows = [{"ID": s.id, "Name": s.name} for s in students]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No students enrolled yet.")

# ─────────────────────────────────────────────────────────────────────────────
# 2️⃣  CREATE SESSION
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🗓️ Create Session":
    st.title("🗓️ Create a Class Session")

    with st.form("session_form"):
        title = st.text_input("Session title *", placeholder="e.g. ML Lecture – Week 6")
        col1, col2, col3 = st.columns(3)
        with col1:
            date = st.date_input("Date *", datetime.date.today())
        with col2:
            start = st.time_input("Start time *", datetime.time(9, 0))
        with col3:
            end = st.time_input("End time *", datetime.time(10, 0))
        descr = st.text_area("Description (optional)")
        submitted = st.form_submit_button("➕ Create Session", type="primary")

    if submitted:
        if not title:
            st.error("Title is required.")
        elif end <= start:
            st.error("End time must be after start time.")
        else:
            with get_db() as db:
                new_sess = DBSession(
                    title=title, date=date,
                    start_time=start, end_time=end,
                    description=descr,
                )
                db.add(new_sess)
                db.flush()
                sess_id = new_sess.id
            st.success(f"✅ Session **{title}** created (ID = {sess_id}).")

    # ── Session list ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("All Sessions")
    with get_db() as db:
        sessions = db.query(DBSession).order_by(DBSession.date.desc()).all()
        rows = [{"ID": s.id, "Title": s.title, "Date": str(s.date),
                 "Start": str(s.start_time), "End": str(s.end_time)} for s in sessions]
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No sessions yet.")

# ─────────────────────────────────────────────────────────────────────────────
# 3️⃣  UPLOAD VIDEO & PROCESS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📽️ Upload Video":
    st.title("📽️ Process a Recorded Class Video")
    st.write("Upload the lecture video. Frames are sampled every **N seconds**, "
             "faces are matched against enrolled students, and attendance is recorded.")

    # ── Session picker ────────────────────────────────────────────────────────
    with get_db() as db:
        sessions = db.query(DBSession).order_by(DBSession.date.desc()).all()
        session_map = {f"{s.title}  [{s.date}]  (ID {s.id})": s.id for s in sessions}

    if not session_map:
        st.warning("⚠️ No sessions found. Please create a session first.")
        st.stop()

    selected_label = st.selectbox("Select session", list(session_map.keys()))
    session_id = session_map[selected_label]

    # ── Config tweaks ─────────────────────────────────────────────────────────
    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        interval = st.slider("Frame sampling interval (seconds)", 1, 30,
                             int(cfg.get("frame_interval_sec", 5)))
    with col_cfg2:
        threshold = st.slider("Match confidence threshold", 0.30, 0.80,
                              float(cfg.get("presence_threshold", 0.45)), 0.05)

    del_frames = st.checkbox("🗑️ Delete audit frames after processing (saves disk space)", value=False)

    # ── File upload ───────────────────────────────────────────────────────────
    uploaded = st.file_uploader("Upload video (MP4 / AVI / MOV)", type=["mp4", "avi", "mov"])

    if uploaded:
        upload_dir = Path(cfg.get("upload_folder", "uploaded_videos"))
        upload_dir.mkdir(parents=True, exist_ok=True)
        ext = Path(uploaded.name).suffix
        video_path = upload_dir / f"{uuid.uuid4().hex}{ext}"

        with open(video_path, "wb") as f:
            f.write(uploaded.read())
        st.info(f"Video saved → `{video_path.name}`")

        if st.button("🚀 Run Attendance Detection", type="primary"):
            # Temporarily override config values from UI sliders
            cfg["frame_interval_sec"] = interval
            cfg["presence_threshold"] = threshold

            progress_bar = st.progress(0)
            status_txt = st.empty()

            def on_progress(pct: int):
                progress_bar.progress(pct)
                status_txt.text(f"Processing… {pct}%")

            with st.spinner("Analysing video frames…"):
                from recogniser.video_processor import process_video
                result = process_video(
                    str(video_path),
                    session_id,
                    progress_callback=on_progress,
                    delete_frames_after=del_frames,
                )

            status_txt.empty()
            progress_bar.progress(100)

            n_present = len(result["present_ids"])
            n_frames  = result["processed_frames"]

            st.success(f"✅ Done!  Sampled **{n_frames}** frames  •  "
                       f"**{n_present}** student(s) marked present.")
            st.balloons()

            # Quick result summary
            if result["present_ids"]:
                with get_db() as db:
                    names = [db.query(Student).get(sid).name
                             for sid in result["present_ids"]]
                st.markdown("**Present:** " + ", ".join(sorted(names)))

# ─────────────────────────────────────────────────────────────────────────────
# 4️⃣  DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Dashboard":
    st.title("📊 Attendance Dashboard")

    # ── Date picker ───────────────────────────────────────────────────────────
    with get_db() as db:
        raw_dates = db.query(DBSession.date).distinct().order_by(DBSession.date.desc()).all()
    date_list = [d[0] for d in raw_dates]

    if not date_list:
        st.info("No sessions recorded yet. Create a session and process a video first.")
        st.stop()

    chosen_date = st.selectbox("Select date", date_list,
                               format_func=lambda d: str(d))

    # ── Session picker ────────────────────────────────────────────────────────
    with get_db() as db:
        day_sessions = (db.query(DBSession)
                          .filter(DBSession.date == chosen_date)
                          .order_by(DBSession.start_time)
                          .all())
        sess_map = {f"{s.title} ({s.start_time}–{s.end_time})": s.id
                    for s in day_sessions}

    if not sess_map:
        st.info("No sessions on this date.")
        st.stop()

    sel_label = st.selectbox("Session", list(sess_map.keys()))
    sess_id   = sess_map[sel_label]

    # ── Fetch data ───────────────────────────────────────────────────────────
    with get_db() as db:
        all_students   = db.query(Student).order_by(Student.name).all()
        present_rows   = (db.query(Attendance)
                            .filter(Attendance.session_id == sess_id)
                            .all())
        present_ids    = {r.student_id for r in present_rows}
        first_seen_map = {r.student_id: r.first_seen for r in present_rows}
        conf_map       = {r.student_id: r.confidence  for r in present_rows}

    n_total   = len(all_students)
    n_present = len(present_ids)
    n_absent  = n_total - n_present

    # ── Summary metrics ───────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="metric-card"><div class="metric-num">{n_total}</div>'
                f'<div class="metric-lbl">Total Students</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="metric-card"><div class="metric-num" style="color:#4ade80">{n_present}</div>'
                f'<div class="metric-lbl">Present</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="metric-card"><div class="metric-num" style="color:#f87171">{n_absent}</div>'
                f'<div class="metric-lbl">Absent</div></div>', unsafe_allow_html=True)
    pct = round(n_present / n_total * 100) if n_total else 0
    c4.markdown(f'<div class="metric-card"><div class="metric-num" style="color:#facc15">{pct}%</div>'
                f'<div class="metric-lbl">Attendance %</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Attendance table ───────────────────────────────────────────────────────
    rows = []
    for stu in all_students:
        if stu.id in present_ids:
            status = "Present"
            seen   = first_seen_map[stu.id].strftime("%H:%M:%S") if first_seen_map.get(stu.id) else ""
            conf   = f"{conf_map.get(stu.id, 0):.3f}"
        else:
            status = "Absent"
            seen   = ""
            conf   = ""
        rows.append({"ID": stu.id, "Name": stu.name,
                     "Status": status, "First Seen": seen, "Confidence": conf})

    df = pd.DataFrame(rows)

    search = st.text_input("🔍 Search by name", "")
    if search:
        df = df[df["Name"].str.contains(search, case=False)]

    # colour status column
    def colour_status(val):
        if val == "Present":
            return "color: #4ade80; font-weight:600"
        elif val == "Absent":
            return "color: #f87171; font-weight:600"
        return ""

    styled = df.style.applymap(colour_status, subset=["Status"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── CSV download ──────────────────────────────────────────────────────────
    csv_bytes = df.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Export CSV",
        data=csv_bytes,
        file_name=f"attendance_session_{sess_id}.csv",
        mime="text/csv",
    )

# ─────────────────────────────────────────────────────────────────────────────
# 5️⃣  AUDIT LOG
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📋 Audit Log":
    st.title("📋 Audit Log – Sampled Frames")
    st.write("Every frame sampled during video processing is stored here for review.")

    # Session selector
    with get_db() as db:
        sessions = db.query(DBSession).order_by(DBSession.date.desc()).all()
        sess_map = {f"[{s.date}] {s.title} (ID {s.id})": s.id for s in sessions}

    if not sess_map:
        st.info("No sessions yet.")
        st.stop()

    sel = st.selectbox("Session", list(sess_map.keys()))
    sess_id = sess_map[sel]

    with get_db() as db:
        logs     = (db.query(AttendanceLog)
                      .filter(AttendanceLog.session_id == sess_id)
                      .order_by(AttendanceLog.frame_ts)
                      .all())
        log_data = [
            {
                "id":         l.id,
                "frame_ts":   l.frame_ts,
                "student_id": l.student_id,
                "confidence": l.confidence,
                "frame_path": l.frame_path,
            }
            for l in logs
        ]
        # also fetch student names in same session
        student_names = {s.id: s.name for s in db.query(Student).all()}

    if not log_data:
        st.info("No frames logged for this session yet. Process a video first.")
        st.stop()

    st.write(f"**{len(log_data)}** frames logged.")

    only_detections = st.checkbox("Show only frames with a recognised student", value=False)
    if only_detections:
        log_data = [l for l in log_data if l["student_id"] is not None]

    # Display in expandable cards
    for log in log_data:
        ts   = log["frame_ts"]
        sid  = log["student_id"]
        name = student_names.get(sid, "Unknown") if sid else "No match"
        conf = log["confidence"]

        label = f"⏱ {ts:.1f}s — {name} (conf={conf:.3f})"
        with st.expander(label):
            col_img, col_info = st.columns([2, 1])
            with col_img:
                fp = log["frame_path"]
                if os.path.isfile(fp):
                    st.image(Image.open(fp), use_container_width=True)
                else:
                    st.warning("Frame image not found on disk (may have been deleted).")
            with col_info:
                st.markdown(f"**Student:** {name}")
                st.markdown(f"**Confidence:** {conf:.4f}")
                st.markdown(f"**Timestamp:** {ts:.2f} s")
                st.markdown(f"**Frame path:** `{os.path.basename(fp)}`")
