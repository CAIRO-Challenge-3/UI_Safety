"""
Real-time PPE detection from a laptop webcam.

Model: YOLOv8n fine-tuned for construction-site PPE
       (Hugging Face: mancalazure/construction-ppe-safety, best.pt, ~6 MB,
        weights CC BY 4.0, trained on Roboflow "construction-site-safety").
Classes come from the model itself (typically Hardhat, Mask, Safety Vest,
their NO-* violation counterparts, Person, Safety Cone, machinery, vehicle).

Install:
    pip install ultralytics opencv-python huggingface_hub

Run:
    python ppe_webcam.py                      # default webcam, auto device
    python ppe_webcam.py --conf 0.4 --imgsz 416
    python ppe_webcam.py --weights path/to/your_ppe.pt
    python ppe_webcam.py --device cpu         # force CPU

Keys:  q / Esc = quit   s = save screenshot   p = pause
"""

import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

DEFAULT_REPO = "mancalazure/construction-ppe-safety"
DEFAULT_FILE = "best.pt"

# BGR colours
GREEN = (60, 200, 60)      # PPE present
RED = (40, 40, 230)        # violation (NO-*)
BLUE = (230, 160, 40)      # person
GREY = (170, 170, 170)     # everything else (cones, vehicles, ...)


def get_weights(weights_arg: str | None) -> str:
    """Use a local .pt if given, otherwise download the default from the HF Hub."""
    if weights_arg:
        if not Path(weights_arg).exists():
            raise FileNotFoundError(f"Weights not found: {weights_arg}")
        return weights_arg
    from huggingface_hub import hf_hub_download
    print(f"Downloading {DEFAULT_REPO}/{DEFAULT_FILE} (cached after first run)...")
    return hf_hub_download(repo_id=DEFAULT_REPO, filename=DEFAULT_FILE)


def colour_for(name: str):
    n = name.lower()
    if n.startswith("no-") or n.startswith("no_") or n.startswith("no "):
        return RED
    if n == "person":
        return BLUE
    if n in {"hardhat", "helmet", "mask", "safety vest", "vest", "gloves", "goggles"}:
        return GREEN
    return GREY


def draw_box(img, xyxy, label, colour):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y_text = max(y1, th + 6)
    cv2.rectangle(img, (x1, y_text - th - 6), (x1 + tw + 6, y_text), colour, -1)
    cv2.putText(img, label, (x1 + 3, y_text - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_banner(img, violations: dict, fps: float):
    h, w = img.shape[:2]
    ok = not violations
    bg = (40, 120, 40) if ok else (30, 30, 180)
    cv2.rectangle(img, (0, 0), (w, 34), bg, -1)
    text = "PPE OK" if ok else "VIOLATION: " + ", ".join(
        f"{k} x{v}" for k, v in violations.items())
    cv2.putText(img, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)
    fps_txt = f"{fps:5.1f} FPS"
    (tw, _), _ = cv2.getTextSize(fps_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(img, fps_txt, (w - tw - 10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description="Webcam PPE detection (YOLOv8n)")
    ap.add_argument("--weights", default=None, help="local .pt (default: download from HF)")
    ap.add_argument("--camera", type=int, default=0, help="camera index")
    ap.add_argument("--conf", type=float, default=0.35, help="confidence threshold")
    ap.add_argument("--imgsz", type=int, default=640, help="inference size (416/320 = faster)")
    ap.add_argument("--device", default=None, help="'cpu', '0' (CUDA), 'mps' (Apple)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    args = ap.parse_args()

    model = YOLO(get_weights(args.weights))
    names = model.names
    print("Classes:", list(names.values()))

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {args.camera}")

    win = "PPE Detection  (q: quit, s: screenshot, p: pause)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    fps, t_prev, paused, frame = 0.0, time.time(), False, None
    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                print("Frame grab failed; exiting.")
                break
            frame = cv2.flip(frame, 1)  # mirror view feels natural on a laptop

            res = model.predict(frame, conf=args.conf, imgsz=args.imgsz,
                                device=args.device, verbose=False)[0]

            violations = {}
            for box in res.boxes:
                cls_name = names[int(box.cls)]
                conf = float(box.conf)
                colour = colour_for(cls_name)
                draw_box(frame, box.xyxy[0].tolist(), f"{cls_name} {conf:.2f}", colour)
                if colour == RED:
                    violations[cls_name] = violations.get(cls_name, 0) + 1

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-6))  # smoothed
            t_prev = now
            draw_banner(frame, violations, fps)

        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("s"):
            fn = f"ppe_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(fn, frame)
            print("Saved", fn)
        if key == ord("p"):
            paused = not paused
            t_prev = time.time()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()