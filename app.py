# app.py -------------------------------------------------------------
import streamlit as st
import os
import uuid
import datetime
import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from PIL import Image

# ── Local imports ──────────────────────────────────────────────────────────────
from db import init_db, get_db
from models import Student, Session as DBSession
from config import cfg
import utils_db as udb

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="College Attendance System",
    page_icon="🎓",
    layout="wide",
)

init_db()

# ── Cache heavy model ──────────────────────────────────────────────────────────
@st.cache_resource
def get_matcher():
    from recogniser.insightface_wrapper import FaceMatcher
    return FaceMatcher(presence_threshold=cfg.get("presence_threshold", 0.45))


# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background: linear-gradient(180deg,#1a1a2e,#16213e); }
    [data-testid="stSidebar"] * { color: #e0e0e0 !important; }
    .metric-card { background:#1e293b; border-radius:12px; padding:1.2rem;
                   text-align:center; border:1px solid #334155; }
    .metric-num  { font-size:2.5rem; font-weight:700; color:#38bdf8; }
    .metric-lbl  { font-size:.9rem;  color:#94a3b8; margin-top:.3rem; }
    .danger-zone { border:1px solid #ef444460; border-radius:10px;
                   padding:1rem; background:#1a0a0a; margin-top:1rem; }
    h1 { color:#e2e8f0 !important; }
    h2, h3 { color:#cbd5e1 !important; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/fluency/96/graduation-cap.png", width=60)
st.sidebar.title("Attendance System")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Dashboard", "👤 Enrol Students", "🗓️ Create Session",
     "📽️ Upload Video", "📋 Audit Log"],
)
st.sidebar.markdown(f"**Enrolled students:** {udb.get_student_count()}")

# ─────────────────────────────────────────────────────────────────────────────
# 1️⃣  ENROL STUDENTS
# ─────────────────────────────────────────────────────────────────────────────
if page == "👤 Enrol Students":
    st.title("👤 Enrol a New Student")
    st.write("Upload a clear, well-lit portrait. The ArcFace embedding is computed and stored.")

    name     = st.text_input("Full name", placeholder="e.g. Rishi Sharma")
    img_file = st.file_uploader("Portrait photo (JPEG / PNG)", type=["jpg", "jpeg", "png"])

    if img_file:
        st.image(Image.open(img_file).convert("RGB"), caption="Preview", width=280)
        img_file.seek(0)

    if st.button("✅ Enrol Student", type="primary", disabled=not (name and img_file)):
        np_arr = np.frombuffer(img_file.read(), np.uint8)
        bgr    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        with st.spinner("Computing face embedding…"):
            matcher = get_matcher()
            emb     = matcher.get_embedding(bgr)
        if emb is None:
            st.error("❌ No face detected. Please upload a clearer portrait.")
        else:
            with get_db() as db:
                existing = db.query(Student).filter(Student.name == name).first()
                if existing:
                    existing.embedding = emb.tobytes()
                    st.warning(f"⚠️ **{name}** already existed — embedding updated.")
                else:
                    db.add(Student(name=name, embedding=emb.tobytes()))
                    st.success(f"✅ **{name}** enrolled!")
            matcher.reload()
            st.rerun()

    # ── Enrolled list + per-student delete ───────────────────────────────────
    st.markdown("---")
    st.subheader("Currently Enrolled")
    students = udb.get_all_students()

    if students:
        for stu in students:
            col_name, col_del = st.columns([5, 1])
            col_name.markdown(f"**{stu['Name']}** (ID: {stu['ID']})")
            if col_del.button("🗑️", key=f"del_stu_{stu['ID']}",
                               help=f"Delete {stu['Name']} and all their photos"):
                deleted = udb.delete_student(stu["ID"])
                st.warning(f"🗑️ **{deleted}** removed from DB and disk.")
                st.rerun()
    else:
        st.info("No students enrolled yet.")

    # ── Danger zone: delete ALL students ─────────────────────────────────────
    with st.expander("⚠️ Danger Zone – Delete ALL Students"):
        st.markdown('<div class="danger-zone">', unsafe_allow_html=True)
        st.warning("This will delete **every enrolled student**, all their photos, and all attendance records.")
        if st.button("🔥 Delete All Students", type="secondary"):
            ids = [s["ID"] for s in udb.get_all_students()]
            for sid in ids:
                udb.delete_student(sid)
            st.error(f"All {len(ids)} students deleted.")
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2️⃣  CREATE SESSION
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🗓️ Create Session":
    st.title("🗓️ Create a Class Session")

    with st.form("session_form"):
        title = st.text_input("Session title *", placeholder="e.g. ML Lecture – Week 6")
        col1, col2, col3 = st.columns(3)
        with col1: date  = st.date_input("Date *", datetime.date.today())
        with col2: start = st.time_input("Start time *", datetime.time(9, 0))
        with col3: end   = st.time_input("End time *",   datetime.time(10, 0))
        descr     = st.text_area("Description (optional)")
        submitted = st.form_submit_button("➕ Create Session", type="primary")

    if submitted:
        if not title:
            st.error("Title is required.")
        elif end <= start:
            st.error("End time must be after start time.")
        else:
            with get_db() as db:
                ns = DBSession(title=title, date=date, start_time=start,
                               end_time=end, description=descr)
                db.add(ns)
                db.flush()
                sid = ns.id
            st.success(f"✅ Session **{title}** created (ID = {sid}).")

    st.markdown("---")
    st.subheader("All Sessions")
    sessions = udb.get_all_sessions()
    if sessions:
        st.dataframe(pd.DataFrame(sessions), use_container_width=True, hide_index=True)
    else:
        st.info("No sessions yet.")

# ─────────────────────────────────────────────────────────────────────────────
# 3️⃣  UPLOAD VIDEO & PROCESS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📽️ Upload Video":
    st.title("📽️ Process a Recorded Class Video")

    # ── Session picker ────────────────────────────────────────────────────────
    all_sessions = udb.get_all_sessions()
    if not all_sessions:
        st.warning("⚠️ No sessions found. Create one first.")
        st.stop()

    session_map    = {f"{s['title']}  [{s['date']}]  (ID {s['id']})": s["id"]
                      for s in all_sessions}
    selected_label = st.selectbox("Select session", list(session_map.keys()))
    session_id     = session_map[selected_label]

    col_cfg1, col_cfg2 = st.columns(2)
    with col_cfg1:
        interval  = st.slider("Frame sampling interval (s)", 1, 30,
                               int(cfg.get("frame_interval_sec", 5)))
    with col_cfg2:
        threshold = st.slider("Match confidence threshold", 0.30, 0.80,
                               float(cfg.get("presence_threshold", 0.45)), 0.05)

    del_frames = st.checkbox("🗑️ Delete audit frames after processing", value=False)

    # ── File upload + video player ────────────────────────────────────────────
    uploaded = st.file_uploader("Upload video (MP4 / AVI / MOV)",
                                 type=["mp4", "avi", "mov"])

    if uploaded:
        upload_dir = Path(cfg.get("upload_folder", "uploaded_videos"))
        upload_dir.mkdir(parents=True, exist_ok=True)
        ext        = Path(uploaded.name).suffix
        video_path = upload_dir / f"{uuid.uuid4().hex}{ext}"

        with open(video_path, "wb") as f:
            f.write(uploaded.read())

        # ── Small video player ─────────────────────────────────────────────────
        st.markdown("#### 🎬 Video Preview")
        st.video(str(video_path))

        st.info(f"Saved → `{video_path.name}`")

        if "processing_done" not in st.session_state:
            st.session_state.processing_done = False
        if "last_result" not in st.session_state:
            st.session_state.last_result = None
        if "last_session_id" not in st.session_state:
            st.session_state.last_session_id = None

        if st.button("🚀 Run Attendance Detection", type="primary"):
            cfg["frame_interval_sec"] = interval
            cfg["presence_threshold"] = threshold

            progress_bar = st.progress(0)
            status_txt   = st.empty()

            def on_progress(pct: int):
                progress_bar.progress(pct)
                status_txt.text(f"Processing… {pct}%")

            with st.spinner("Analysing video frames…"):
                from recogniser.video_processor import process_video
                result = process_video(
                    str(video_path), session_id,
                    progress_callback=on_progress,
                    delete_frames_after=del_frames,
                )

            status_txt.empty()
            progress_bar.progress(100)
            st.session_state.processing_done = True
            st.session_state.last_result      = result
            st.session_state.last_session_id  = session_id
            st.balloons()

        # ── Post-processing actions (appear after processing is done) ─────────
        if st.session_state.get("processing_done") and \
                st.session_state.get("last_session_id") == session_id:

            result   = st.session_state.last_result
            n_frames = result["processed_frames"]
            n_present = len(result["present_ids"])

            st.success(f"✅ Done! Sampled **{n_frames}** frames · "
                       f"**{n_present}** student(s) marked present.")

            # Build the attendance dataframe for this session
            df_result = udb.get_attendance_dataframe(session_id)

            # Row 1: CSV download  |  View audit log toggle
            col_dl, col_audit = st.columns(2)

            with col_dl:
                csv_bytes = df_result.to_csv(index=False).encode()
                st.download_button(
                    label="⬇️ Download Attendance CSV",
                    data=csv_bytes,
                    file_name=f"attendance_session_{session_id}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            with col_audit:
                if st.button("🔍 View Audit Log", use_container_width=True):
                    st.session_state["show_audit"] = \
                        not st.session_state.get("show_audit", False)

            # Quick results table
            st.dataframe(df_result, use_container_width=True, hide_index=True)

            # ── Inline audit log viewer ────────────────────────────────────────
            if st.session_state.get("show_audit", False):
                st.markdown("---")
                st.subheader("📋 Audit Log – Sampled Frames")

                logs = udb.get_audit_logs(session_id)
                if not logs:
                    st.info("No frames logged yet.")
                else:
                    only_hits = st.checkbox("Only frames with a recognised student",
                                            key="ov_only_hits", value=False)
                    if only_hits:
                        logs = [l for l in logs if l["student_id"] is not None]

                    for log in logs:
                        ts   = log["frame_ts"]
                        name = log["student_name"]
                        conf = log["confidence"]
                        with st.expander(f"⏱ {ts:.1f}s — {name} (conf={conf:.3f})"):
                            fp = log["frame_path"]
                            col_i, col_m = st.columns([2, 1])
                            with col_i:
                                if os.path.isfile(fp):
                                    st.image(Image.open(fp), width=680)
                                else:
                                    st.warning("Frame deleted from disk.")
                            with col_m:
                                st.markdown(f"**Student:** {name}")
                                st.markdown(f"**Confidence:** {conf:.4f}")
                                st.markdown(f"**Timestamp:** {ts:.2f} s")

# ─────────────────────────────────────────────────────────────────────────────
# 4️⃣  DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Dashboard":
    st.title("📊 Attendance Dashboard")

    date_list = udb.get_distinct_dates()
    if not date_list:
        st.info("No sessions recorded yet.")
        st.stop()

    chosen_date  = st.selectbox("Select date", date_list, format_func=lambda d: str(d))
    day_sessions = udb.get_sessions_for_date(chosen_date)
    if not day_sessions:
        st.info("No sessions on this date.")
        st.stop()

    sess_map  = {f"{s['title']} ({s['start_time']}–{s['end_time']})": s["id"]
                 for s in day_sessions}
    sel_label = st.selectbox("Session", list(sess_map.keys()))
    sess_id   = sess_map[sel_label]

    df        = udb.get_attendance_dataframe(sess_id)
    n_total   = len(df)
    n_present = int((df["Status"] == "Present").sum())
    n_absent  = n_total - n_present
    pct       = round(n_present / n_total * 100) if n_total else 0

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    for col, num, lbl, colour in [
        (c1, n_total,   "Total",   "#38bdf8"),
        (c2, n_present, "Present", "#4ade80"),
        (c3, n_absent,  "Absent",  "#f87171"),
        (c4, f"{pct}%", "Attendance %", "#facc15"),
    ]:
        col.markdown(f'<div class="metric-card"><div class="metric-num" style="color:{colour}">'
                     f'{num}</div><div class="metric-lbl">{lbl}</div></div>',
                     unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    search  = st.text_input("🔍 Search by name", "")
    view_df = df[df["Name"].str.contains(search, case=False)] if search else df

    def colour_status(val):
        if val == "Present": return "color:#4ade80; font-weight:600"
        if val == "Absent":  return "color:#f87171; font-weight:600"
        return ""

    styled = view_df.style.map(colour_status, subset=["Status"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Export CSV",
        data=df.to_csv(index=False).encode(),
        file_name=f"attendance_session_{sess_id}.csv",
        mime="text/csv",
    )

# ─────────────────────────────────────────────────────────────────────────────
# 5️⃣  AUDIT LOG
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📋 Audit Log":
    st.title("📋 Audit Log – Sampled Frames")

    all_sessions = udb.get_all_sessions()
    if not all_sessions:
        st.info("No sessions yet.")
        st.stop()

    sess_map = {f"[{s['date']}] {s['title']} (ID {s['id']})": s["id"]
                for s in all_sessions}
    sel      = st.selectbox("Session", list(sess_map.keys()))
    sess_id  = sess_map[sel]

    log_count = udb.get_audit_log_count(sess_id)
    st.write(f"**{log_count}** frames logged for this session.")

    # ── Delete audit log button ───────────────────────────────────────────────
    col_del_opt, col_del_btn = st.columns([3, 1])
    with col_del_opt:
        del_images = st.checkbox("Also delete frame images from disk", value=True)
    with col_del_btn:
        if st.button("🗑️ Delete Audit Log", use_container_width=True):
            n = udb.delete_audit_logs(sess_id, delete_images=del_images)
            st.warning(f"🗑️ Deleted **{n}** log entries"
                       + (" and their frame images." if del_images else "."))
            st.rerun()

    st.markdown("---")

    logs = udb.get_audit_logs(sess_id)
    if not logs:
        st.info("No frames logged for this session.")
        st.stop()

    only_hits = st.checkbox("Show only recognised-student frames", value=False)
    if only_hits:
        logs = [l for l in logs if l["student_id"] is not None]

    for log in logs:
        ts   = log["frame_ts"]
        name = log["student_name"]
        conf = log["confidence"]

        with st.expander(f"⏱ {ts:.1f}s — {name} (conf={conf:.3f})"):
            col_img, col_info = st.columns([2, 1])
            with col_img:
                fp = log["frame_path"]
                if os.path.isfile(fp):
                    # Bounding box is already drawn on the saved frame
                    st.image(Image.open(fp), width=700)
                else:
                    st.warning("Frame image not found on disk (may have been deleted).")
            with col_info:
                st.markdown(f"**Student:** {name}")
                st.markdown(f"**Confidence:** {conf:.4f}")
                st.markdown(f"**Timestamp:** {ts:.2f} s")
                st.markdown(f"**File:** `{os.path.basename(fp)}`")
