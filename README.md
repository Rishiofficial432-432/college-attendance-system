# 🎓 College Attendance System

A generic and automated attendance system using Face Recognition technology. This project aims to replace manual attendance marking with a seamless, contactless, and efficient solution using Computer Vision.

## 🚀 Features

- **Face Enrollment**: Capture and store student face data for training.
- **Automated Detection (Upcoming)**: Real-time face detection from a camera feed.
- **Attendance Logging (Upcoming)**: Mark attendance automatically when a registered face is recognized.
- **CSV Export (Upcoming)**: Export attendance records for administration.

## 🛠️ Tech Stack

- **Python 3.x**
- **OpenCV**: For image processing and face detection.
- **Numpy**: For numerical operations.
- **Haar Cascades**: Pre-trained models for face detection.

## 📂 Project Structure

```
COLLEGE-ATTENDANCE-SYSTEM/
│
├── attendance_system/
│   ├── dataset/              # Stores captured face images
│   │   └── person_1/         # Images for a specific user
│   │
│   └── enroll_faces.py       # Script to capture and save user faces
│
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## ⚙️ prerequisites

Ensure you have Python installed on your system. You can download it from [python.org](https://www.python.org/).

## 📥 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Rishiofficial432-432/college-attendance-system.git
    cd college-attendance-system
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## 🖥️ Usage

### 1. Enroll a New Student
To capture face data for a new student, run the **enrollment script**.

1.  Open `attendance_system/enroll_faces.py`.
2.  Update the `PERSON_ID` variable with the student's unique ID or Name:
    ```python
    PERSON_ID = "101_JohnDoe"
    ```
3.  Run the script:
    ```bash
    python attendance_system/enroll_faces.py
    ```
4.  Look straight at the webcam. The script will automatically capture **30 images** and save them in the `dataset/` folder.
5.  Press `ESC` to exit manually if needed.

### 2. (Next Steps) Train & Recognize
*Current version supports enrollment. Future updates will include the training model and recognition script.*

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repo and submit a pull request.

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).