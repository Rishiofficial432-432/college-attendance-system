import cv2
import os

# -------- CONFIG --------
PERSON_ID = "person_1"      # change for each student
SAVE_DIR = f"dataset/{PERSON_ID}"
IMG_SIZE = 128
NUM_IMAGES = 30
# ------------------------

# Create folder if not exists
os.makedirs(SAVE_DIR, exist_ok=True)

# Load Haar Cascade for face detection (simple & fast for enrollment)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

cap = cv2.VideoCapture(0)  # webcam
count = 0

print("[INFO] Starting face enrollment...")
print("[INFO] Look straight at the camera")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))

        count += 1
        file_path = os.path.join(SAVE_DIR, f"{count}.jpg")
        cv2.imwrite(file_path, face)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, f"Saved: {count}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        if count >= NUM_IMAGES:
            break

    cv2.imshow("Enrollment", frame)

    if cv2.waitKey(1) & 0xFF == 27 or count >= NUM_IMAGES:
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Enrollment completed.")
