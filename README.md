# 🎓 AI-Powered College Attendance System

A modern, automated attendance tracking solution powered by Computer Vision. This system uses face recognition to streamline attendance marking, replacing manual registers with a seamless digital workflow.

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🌟 Key Features

*   **👤 Biometric Enrollment**: Enroll students by capturing high-quality facial embeddings using ArcFace/InsightFace.
*   **📂 Course & Session Management**: Organize attendance by specific courses and individual class sessions.
*   **📽️ Video-Based Attendance**: Process recorded class videos to automatically detect and mark present students.
*   **📊 Insightful Dashboard**: Real-time analytics and attendance percentages for every session.
*   **📋 Audit Logs**: Transparent verification system with sampled frames showing exactly when a student was recognized.
*   **📥 Data Export**: Export attendance records directly to CSV for administrative use.

## 🛠️ Tech Stack

*   **Frontend**: Streamlit (Modern Web UI)
*   **Computer Vision**: InsightFace (ArcFace), OpenCV, ONNX Runtime
*   **Database**: SQLite with SQLAlchemy ORM
*   **Data Handling**: Pandas, NumPy
*   **Config Management**: PyYAML

## 🚀 Getting Started

### Prerequisites

*   Python 3.9 or higher
*   A webcam or recorded video files for attendance

### Installation

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Rishiofficial432-432/college-attendance-system.git
    cd college-attendance-system
    ```

2.  **Set Up Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

### Running the App

```bash
streamlit run app.py
```

## 📖 Usage Guide

1.  **Manage Courses**: Create your course categories (e.g., CS101 - Machine Learning).
2.  **Enrol Students**: Upload a clear photo of each student to generate their unique biometric embedding.
3.  **Create Session**: Define a class session for a specific date and time.
4.  **Upload Video**: Upload the recorded class video. The system will sample frames and identify students automatically.
5.  **Review & Export**: Check the Dashboard or Audit Log to verify results and download the CSV report.

## 🔒 Privacy & Security

This project is designed with privacy in mind. Face images are stored locally and are excluded from version control via `.gitignore`. Only mathematical embeddings are used for recognition.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue for feature requests.

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.