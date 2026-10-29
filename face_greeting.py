"""
Personalized greeting from a laptop webcam using face recognition.

Models (OpenCV Zoo, downloaded to ./models on first run, ~40 MB total):
    YuNet  - face detection     (face_detection_yunet_2023mar.onnx)
    SFace  - face recognition   (face_recognition_sface_2021dec.onnx)
Both run through OpenCV's built-in cv2.FaceDetectorYN / cv2.FaceRecognizerSF,
so nothing beyond opencv-python and numpy is required.

Known people live in a folder per person:
    faces/
        Alice/  img1.jpg  img2.jpg ...
        Bob/    img1.jpg ...
You can also enroll live: look at the camera, press 'e', type your name in
the terminal. A few photos per person (different angles/lighting) help.

Install:
    pip install opencv-python numpy
    pip install pyttsx3            # optional, only for --speak

Run:
    python face_greeting.py
    python face_greeting.py --speak --cooldown 60
    python face_greeting.py --threshold 0.40     # stricter matching

Keys:  q / Esc = quit   e = enroll the largest visible face   r = reload faces/
"""

import argparse
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

MODEL_DIR = Path(__file__).parent / "models"
FACES_DIR = Path(__file__).parent / "faces"
ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
DETECTOR = ("face_detection_yunet_2023mar.onnx", f"{ZOO}/face_detection_yunet")
RECOGNIZER = ("face_recognition_sface_2021dec.onnx", f"{ZOO}/face_recognition_sface")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# BGR colours
GREEN = (60, 200, 60)      # recognised
ORANGE = (40, 140, 240)    # unknown


def get_model(name: str, base_url: str) -> str:
    path = MODEL_DIR / name
    if not path.exists():
        MODEL_DIR.mkdir(exist_ok=True)
        print(f"Downloading {name} ...")
        urllib.request.urlretrieve(f"{base_url}/{name}", path)
    return str(path)


class Speaker:
    """Non-blocking text-to-speech (pyttsx3) so the video loop never stalls."""

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._busy = threading.Lock()
        if enabled:
            try:
                import pyttsx3  # noqa: F401
            except ImportError:
                print("pyttsx3 not installed; greetings will be on-screen only.")
                self.enabled = False

    def say(self, text: str):
        if not self.enabled or self._busy.locked():
            return
        threading.Thread(target=self._run, args=(text,), daemon=True).start()

    def _run(self, text: str):
        import pyttsx3
        with self._busy:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()


class FaceGreeter:
    def __init__(self, threshold: float, det_conf: float):
        self.detector = cv2.FaceDetectorYN.create(
            get_model(*DETECTOR), "", (320, 320), det_conf, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(get_model(*RECOGNIZER), "")
        self.threshold = threshold        # SFace cosine similarity; 0.363 is the paper's value
        self.known: dict[str, list[np.ndarray]] = {}

    def detect(self, img):
        h, w = img.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(img)
        return [] if faces is None else list(faces)

    def embed(self, img, face):
        aligned = self.recognizer.alignCrop(img, face)
        return self.recognizer.feature(aligned)

    def load_known(self):
        self.known.clear()
        FACES_DIR.mkdir(exist_ok=True)
        for person_dir in sorted(p for p in FACES_DIR.iterdir() if p.is_dir()):
            for img_path in person_dir.iterdir():
                if img_path.suffix.lower() not in IMG_EXTS:
                    continue
                img = cv2.imread(str(img_path))
                faces = self.detect(img) if img is not None else []
                if not faces:
                    print(f"  no face found in {img_path}, skipping")
                    continue
                largest = max(faces, key=lambda f: f[2] * f[3])
                self.known.setdefault(person_dir.name, []).append(self.embed(img, largest))
        print("Known people:", {k: len(v) for k, v in self.known.items()} or "none yet")

    def identify(self, img, face):
        """Return (name or None, best similarity score)."""
        feat = self.embed(img, face)
        best_name, best_score = None, -1.0
        for name, feats in self.known.items():
            score = max(self.recognizer.match(feat, f, cv2.FaceRecognizerSF_FR_COSINE)
                        for f in feats)
            if score > best_score:
                best_name, best_score = name, score
        return (best_name if best_score >= self.threshold else None), best_score

    def enroll(self, img, face, name: str):
        person_dir = FACES_DIR / name
        person_dir.mkdir(parents=True, exist_ok=True)
        # Save a padded crop so it can be re-detected on reload
        x, y, w, h = map(int, face[:4])
        pad = int(0.4 * max(w, h))
        H, W = img.shape[:2]
        crop = img[max(0, y - pad):min(H, y + h + pad), max(0, x - pad):min(W, x + w + pad)]
        cv2.imwrite(str(person_dir / f"{time.strftime('%Y%m%d_%H%M%S')}.jpg"), crop)
        self.known.setdefault(name, []).append(self.embed(img, face))


def greeting_for(name: str) -> str:
    hour = datetime.now().hour
    part = "morning" if hour < 12 else "afternoon" if hour < 17 else "evening"
    return f"Good {part}, {name}!"


def draw_face(img, face, label, colour):
    x, y, w, h = map(int, face[:4])
    cv2.rectangle(img, (x, y), (x + w, y + h), colour, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    y_text = max(y, th + 8)
    cv2.rectangle(img, (x, y_text - th - 8), (x + tw + 8, y_text), colour, -1)
    cv2.putText(img, label, (x + 4, y_text - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def draw_banner(img, text: str):
    w = img.shape[1]
    cv2.rectangle(img, (0, 0), (w, 44), (60, 60, 60), -1)
    cv2.putText(img, text, (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                (255, 255, 255), 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description="Webcam face recognition with personalized greetings")
    ap.add_argument("--camera", type=int, default=0, help="camera index")
    ap.add_argument("--threshold", type=float, default=0.363,
                    help="cosine similarity needed to accept a match (higher = stricter)")
    ap.add_argument("--det-conf", type=float, default=0.8, help="face detection confidence")
    ap.add_argument("--cooldown", type=float, default=30.0,
                    help="seconds before the same person is greeted again")
    ap.add_argument("--speak", action="store_true", help="say greetings out loud (pyttsx3)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    greeter = FaceGreeter(args.threshold, args.det_conf)
    greeter.load_known()
    speaker = Speaker(args.speak)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")

    win = "Face Greeting  (q: quit, e: enroll, r: reload)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    last_greeted: dict[str, float] = {}
    banner, banner_until = "", 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame grab failed; exiting.")
            break
        frame = cv2.flip(frame, 1)
        clean = frame.copy()       # unannotated copy for enrollment
        faces = greeter.detect(frame)
        now = time.time()

        for face in faces:
            name, score = greeter.identify(frame, face)
            if name:
                draw_face(frame, face, f"{name} {score:.2f}", GREEN)
                if now - last_greeted.get(name, 0) > args.cooldown:
                    last_greeted[name] = now
                    banner, banner_until = greeting_for(name), now + 4
                    print(banner)
                    speaker.say(banner)
            else:
                draw_face(frame, face, "Unknown - press e", ORANGE)

        if now < banner_until:
            draw_banner(frame, banner)

        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("r"):
            greeter.load_known()
        if key == ord("e"):
            if not faces:
                print("No face visible to enroll.")
                continue
            largest = max(faces, key=lambda f: f[2] * f[3])
            name = input("Name for this face (blank to cancel): ").strip()
            if name:
                greeter.enroll(clean, largest, name)
                last_greeted.pop(name, None)   # greet them right away
                print(f"Enrolled {name}.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
