from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "AI server running"}

# Load face detection model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -------------------- DETECT FACE --------------------
@app.post("/detect-face")
async def detect_face(file: UploadFile = File(...)):
    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"status": "error", "message": "Invalid image"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    return {
        "faces_detected": len(faces),
        "status": "verified" if len(faces) > 0 else "no_face"
    }

# -------------------- REGISTER FACE --------------------
@app.post("/register-face")
async def register_face(file: UploadFile = File(...)):
    contents = await file.read()

    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return {"status": "no_face"}

    (x, y, w, h) = faces[0]
    face = gray[y:y+h, x:x+w]

    face = cv2.resize(face, (100, 100))

    try:
        # Save to Render-safe directory
        success = cv2.imwrite("/tmp/known_face.jpg", face)

        if not success:
            return {"error": "Failed to save face"}

        return {"status": "face_registered"}

    except Exception as e:
        return {"error": str(e)}

# -------------------- VERIFY FACE --------------------
@app.post("/verify-face")
async def verify_face(file: UploadFile = File(...)):

    # Load stored face
    known_face = cv2.imread("/tmp/known_face.jpg", cv2.IMREAD_GRAYSCALE)

    if known_face is None:
        return {
            "error": "No face registered. Please register first."
        }

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return {"status": "no_face"}

    (x, y, w, h) = faces[0]
    face = gray[y:y+h, x:x+w]

    face = cv2.resize(face, (100, 100))

    if known_face.shape != face.shape:
        return {
            "error": "Face size mismatch. Try same image."
        }

    try:
        diff = np.mean((known_face - face) ** 2)
        similarity = 1 / (1 + diff)

        return {
            "match": similarity > 0.5,
            "confidence": float(similarity)
        }

    except Exception as e:
        return {
            "error": str(e)
    }
