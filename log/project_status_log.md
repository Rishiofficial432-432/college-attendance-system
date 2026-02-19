# Face Attendance Project Log
**Last Updated**: 2026-02-16

## Project Overview
The project is a **Face Attendance System** built using Python and Streamlit. It automates attendance marking using computer vision techniques.

## Key Components

### 1. Application (`app.py`)
Main entry point for the Streamlit application. It provides a web interface with four main sections:
- **Dashboard**: Displays attendance statistics, records, and allows downloading data as CSV.
- **Enroll Student**: Captures face images from the webcam for training.
- **Train Model**: Processes captured images to train the face recognition model.
- **Live Attendance**: Runs real-time face recognition on a video feed (webcam or RTSP) to mark attendance.

### 2. Pipeline (`face_attendance/pipeline.py`)
Core recognition pipeline with the following capabilities:
- **Face Detection**: Using Haar Cascades via `detector.py`
- **Feature Extraction**: LBP (Local Binary Patterns) via `extractor.py`
- **Dimensionality Reduction**: PCA + LDA via `reducer.py`
- **Classification**: SVM-based matching via `matcher.py`
- **Attendance Logging**: CSV-based tracking via `attendance.py`

**NEW - Batch Processing:**
- `recognize_image(image, write_to_db)`: Process single image
- `process_images_dir(img_dir, dry_run)`: Process directory of images

### 3. CLI (`main.py`)
Command-line interface supporting three modes:
- `train`: Train models on enrolled faces
- `run`: Live video recognition (webcam/RTSP)
- `batch`: **NEW** - Process directory of static images

### 4. Demo Script (`batch_attendance.py`)
Standalone script for end-to-end workflow:
- Optional enrollment via webcam
- Automatic training
- Batch image processing
- Results export

### 5. Dependencies
Based on `requirements.txt`, the project relies on:
- `ultralytics` (YOLO for detection)
- `scikit-image` (LBP features)
- `scikit-learn` (Machine Learning models - PCA, LDA, SVM)
- `pandas` (Data handling)
- `streamlit` (Web Interface)
- `opencv-python` (Image processing)

## Implementation History

### Session 1 (Earlier)
- Initial codebase analysis
- Successfully ran application using `streamlit run app.py`
- Verified application accessible at `http://localhost:8501`
- Created initial project log

### Session 2 (2026-02-16)
- **Added batch image processing capabilities**
  - Implemented `recognize_image` method in `pipeline.py`
  - Implemented `process_images_dir` method in `pipeline.py`
  - Updated `main.py` to support `batch` mode with `--images` and `--dry-run` flags
  - Created `batch_attendance.py` standalone demo script
- **Documentation**
  - Created implementation plan
  - Created comprehensive walkthrough
  - Updated project log

## Usage Examples

### Streamlit UI
```bash
streamlit run app.py
```

### CLI - Live Recognition
```bash
python main.py run --source 0
```

### CLI - Batch Processing
```bash
# Train first
python main.py train

# Process images
python main.py batch --images ./test_photos

# Dry run (no DB changes)
python main.py batch --images ./test_photos --dry-run
```

### Standalone Demo
```bash
python batch_attendance.py --enroll alice bob --photos ./attendance_photos
```

## Current Status
✅ Streamlit UI fully functional  
✅ Live video recognition operational  
✅ **Batch image processing implemented and ready for testing**  
✅ Documentation complete  

## Notes
- Default confidence threshold: 0.6
- Cooldown period: 60 seconds (prevents duplicate entries)
- Supported image formats: jpg, jpeg, png, bmp, tif, tiff
- Face selection: Largest detected face in image
