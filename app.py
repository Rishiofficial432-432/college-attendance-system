import streamlit as st
import cv2
import pandas as pd
import numpy as np
import os
from PIL import Image
from face_attendance.pipeline import FaceAttendanceSystem

# -----------------------------------------------------------------------
# Page config
# -----------------------------------------------------------------------
st.set_page_config(
    page_title="Face Attendance System",
    page_icon="🎓",
    layout="wide"
)

ATTENDANCE_FILE = 'attendance.csv'
DATASET_DIR = 'attendance_system/dataset'

# -----------------------------------------------------------------------
# Load system once (downloads InsightFace model on first run ~200 MB)
# -----------------------------------------------------------------------
@st.cache_resource
def get_system():
    return FaceAttendanceSystem(attendance_file=ATTENDANCE_FILE)

with st.spinner("Loading InsightFace model (first run downloads ~200 MB)..."):
    system = get_system()

# -----------------------------------------------------------------------
# Sidebar navigation
# -----------------------------------------------------------------------
st.sidebar.title("🎓 Face Attendance")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Dashboard", "👤 Enroll Student", "📸 Mark Attendance", "🗂️ Batch Attendance"]
)

enrolled = system.matcher.get_enrolled_names()
if enrolled:
    st.sidebar.success(f"**Enrolled ({len(enrolled)}):** {', '.join(enrolled)}")
else:
    st.sidebar.warning("No students enrolled yet.")

# -----------------------------------------------------------------------
# Helper: convert uploaded file → OpenCV BGR image
# -----------------------------------------------------------------------
def uploaded_to_bgr(uploaded_file):
    pil_img = Image.open(uploaded_file).convert("RGB")
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# -----------------------------------------------------------------------
# PAGE: Dashboard
# -----------------------------------------------------------------------
if page == "📊 Dashboard":
    st.title("📊 Attendance Dashboard")

    if os.path.exists(ATTENDANCE_FILE):
        df = pd.read_csv(ATTENDANCE_FILE, names=['Name', 'Date', 'Time'])
        df = df[df['Name'] != 'Name']  # drop header row if present

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", len(df))
        col2.metric("Unique Students", df['Name'].nunique() if not df.empty else 0)
        col3.metric("Enrolled", len(enrolled))

        if not df.empty:
            st.subheader("Recent Attendance")
            st.dataframe(df.tail(20), use_container_width=True)

            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download CSV", csv, "attendance.csv", "text/csv")
        else:
            st.info("No attendance records yet. Upload a photo on the 'Mark Attendance' page.")
    else:
        st.info("No attendance records yet.")

# -----------------------------------------------------------------------
# PAGE: Enroll Student
# -----------------------------------------------------------------------
elif page == "👤 Enroll Student":
    st.title("👤 Enroll New Student")
    st.write("Upload **one or more clear face photos** of the student. The system averages the embeddings for a robust profile.")

    col1, col2 = st.columns(2)
    with col1:
        student_name = st.text_input("Student Name")
    with col2:
        person_id = st.number_input("Student ID (unique integer)", min_value=1, step=1, value=1)

    uploaded_files = st.file_uploader(
        "Upload face photo(s) — JPG / PNG",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if st.button("✅ Enroll Student", disabled=not (student_name and uploaded_files)):
        embeddings = []
        preview_cols = st.columns(min(len(uploaded_files), 5))

        for i, uf in enumerate(uploaded_files):
            img_bgr = uploaded_to_bgr(uf)
            faces = system.app.get(img_bgr)

            # Show preview with detection box
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            if faces:
                face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                x1, y1, x2, y2 = map(int, face.bbox[:4])
                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 200, 0), 3)
                embeddings.append(face.embedding.astype(np.float32))
                caption = "✅ Face found"
            else:
                caption = "❌ No face"

            if i < 5:
                preview_cols[i].image(img_rgb, caption=caption, use_container_width=True)

        if embeddings:
            mean_emb = np.mean(np.stack(embeddings, axis=0), axis=0).astype(np.float32)
            system.matcher.enroll(int(person_id), student_name, mean_emb)
            system.is_trained = True

            # Save crops to dataset dir for future gallery rebuild
            person_dir = os.path.join(DATASET_DIR, student_name)
            os.makedirs(person_dir, exist_ok=True)
            for idx, uf in enumerate(uploaded_files):
                img_bgr = uploaded_to_bgr(uf)
                cv2.imwrite(os.path.join(person_dir, f"{idx}.jpg"), img_bgr)

            st.success(f"✅ **{student_name}** enrolled from {len(embeddings)} photo(s)!")
            st.rerun()
        else:
            st.error("No faces detected in any of the uploaded photos. Please try clearer images.")

# -----------------------------------------------------------------------
# PAGE: Mark Attendance (single photo)
# -----------------------------------------------------------------------
elif page == "📸 Mark Attendance":
    st.title("📸 Mark Attendance from Photo")

    if not system.is_trained:
        st.warning("⚠️ No students enrolled yet. Please enroll students first.")
    else:
        st.write("Upload a photo (e.g. a classroom snapshot). All detected faces will be matched and attendance marked.")

        uploaded = st.file_uploader("Upload photo — JPG / PNG", type=["jpg", "jpeg", "png"])

        write_to_db = st.checkbox("Write to attendance record", value=True)

        if uploaded:
            img_bgr = uploaded_to_bgr(uploaded)
            img_rgb = cv2.cvtColor(img_bgr.copy(), cv2.COLOR_BGR2RGB)

            faces = system.app.get(img_bgr)

            results = []
            for face in faces:
                x1, y1, x2, y2 = map(int, face.bbox[:4])
                emb = face.embedding.astype(np.float32)
                pid, name, sim = system.matcher.predict(emb)

                if system.matcher.is_match(sim):
                    color = (0, 200, 0)
                    label = f"{name} ({sim:.2f})"
                    if write_to_db:
                        system.attendance.mark(name)
                    results.append({"Name": name, "Confidence": f"{sim:.3f}", "Status": "✅ Present"})
                else:
                    color = (200, 0, 0)
                    label = f"Unknown ({sim:.2f})"
                    results.append({"Name": "Unknown", "Confidence": f"{sim:.3f}", "Status": "❌ Unknown"})

                cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 3)
                cv2.putText(img_rgb, label, (x1, max(y1 - 10, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(img_rgb, caption="Detection Result", use_container_width=True)
            with col2:
                if results:
                    st.subheader(f"Detected {len(faces)} face(s)")
                    st.dataframe(pd.DataFrame(results), use_container_width=True)
                    marked = [r['Name'] for r in results if r['Status'] == '✅ Present']
                    if marked and write_to_db:
                        st.success(f"Marked present: **{', '.join(marked)}**")
                else:
                    st.info("No faces detected in this photo.")

# -----------------------------------------------------------------------
# PAGE: Batch Attendance (multiple photos)
# -----------------------------------------------------------------------
elif page == "🗂️ Batch Attendance":
    st.title("�️ Batch Attendance from Multiple Photos")

    if not system.is_trained:
        st.warning("⚠️ No students enrolled yet. Please enroll students first.")
    else:
        st.write("Upload multiple photos (e.g. 10 classroom snapshots). The system marks a student **present if seen in any photo**.")

        uploaded_files = st.file_uploader(
            "Upload photos — JPG / PNG",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )

        threshold = st.slider("Confidence threshold", 0.30, 0.80, 0.45, 0.05)
        write_to_db = st.checkbox("Write to attendance record", value=True)

        if uploaded_files and st.button(f"🔍 Process {len(uploaded_files)} Photo(s)"):
            already_marked = set()
            all_results = []

            progress = st.progress(0)
            status_text = st.empty()

            for i, uf in enumerate(uploaded_files):
                status_text.text(f"Processing photo {i+1}/{len(uploaded_files)}: {uf.name}")
                img_bgr = uploaded_to_bgr(uf)
                faces = system.app.get(img_bgr)

                for face in faces:
                    emb = face.embedding.astype(np.float32)
                    pid, name, sim = system.matcher.predict(emb)

                    if sim >= threshold and pid is not None:
                        status = "✅ Present"
                        if name not in already_marked:
                            already_marked.add(name)
                            if write_to_db:
                                system.attendance.mark(name)
                    else:
                        status = "❌ Unknown"
                        name = "Unknown"

                    all_results.append({
                        "Photo": uf.name,
                        "Name": name,
                        "Confidence": f"{sim:.3f}",
                        "Status": status
                    })

                progress.progress((i + 1) / len(uploaded_files))

            status_text.empty()
            progress.empty()

            st.subheader("Results")
            if all_results:
                df_results = pd.DataFrame(all_results)
                st.dataframe(df_results, use_container_width=True)

                if already_marked:
                    st.success(f"✅ **Marked present ({len(already_marked)}):** {', '.join(sorted(already_marked))}")
                else:
                    st.warning("No known students detected in any photo.")
            else:
                st.info("No faces detected in any of the uploaded photos.")
