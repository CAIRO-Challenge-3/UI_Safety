"""
Simple PPE detection on Pepper's top camera.

Runs on your laptop (not on the robot): grabs frames from Pepper over the
network via NAOqi, runs a YOLO PPE model, shows the result, and makes Pepper
say a warning when someone is missing equipment.

Requirements (Python 3):
    pip install qi ultralytics opencv-python numpy

Works with NAOqi 2.5 (Pepper <= 1.8 with classic NAOqi). Pepper on NAOqi 2.9
(Android) does not expose ALVideoDevice remotely - use the QiSDK there.

Usage:
    python pepper_ppe.py --ip 192.168.1.42 --weights ppe.pt
"""

import argparse
import time

import cv2
import numpy as np
import qi
from ultralytics import YOLO

# NAOqi constants
TOP_CAMERA = 0
RES_VGA = 2            # 640x480
COLORSPACE_BGR = 13    # matches OpenCV's channel order
FPS = 10

# Class names that mean "PPE missing". Adjust to the labels of YOUR model
# (print model.names to see them).
VIOLATION_CLASSES = {"NO-Hardhat", "NO-Safety Vest", "NO-Mask", "no_helmet", "no_vest"}


def connect(ip, port):
    session = qi.Session()
    session.connect(f"tcp://{ip}:{port}")
    return session


def grab_frame(video, handle):
    img = video.getImageRemote(handle)
    if img is None:
        return None
    width, height, data = img[0], img[1], img[6]
    frame = np.frombuffer(bytes(data), dtype=np.uint8).reshape((height, width, 3))
    return frame.copy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", required=True, help="Pepper's IP address")
    parser.add_argument("--port", type=int, default=9559)
    parser.add_argument("--weights", required=True, help="YOLO model trained on PPE classes")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--speak", action="store_true", help="Let Pepper say warnings")
    args = parser.parse_args()

    model = YOLO(args.weights)
    print("Model classes:", model.names)

    session = connect(args.ip, args.port)
    video = session.service("ALVideoDevice")
    tts = session.service("ALTextToSpeech") if args.speak else None

    handle = video.subscribeCamera("ppe_detector", TOP_CAMERA, RES_VGA, COLORSPACE_BGR, FPS)
    last_warning = 0.0
    cooldown_s = 5.0

    try:
        while True:
            frame = grab_frame(video, handle)
            if frame is None:
                continue

            result = model(frame, conf=args.conf, verbose=False)[0]
            labels = [model.names[int(c)] for c in result.boxes.cls]
            violations = [l for l in labels if l in VIOLATION_CLASSES]

            annotated = result.plot()
            status = "VIOLATION: " + ", ".join(sorted(set(violations))) if violations else "OK"
            color = (0, 0, 255) if violations else (0, 200, 0)
            cv2.putText(annotated, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.imshow("Pepper PPE", annotated)

            if violations and tts and time.time() - last_warning > cooldown_s:
                tts.say("Please wear your safety equipment.")
                last_warning = time.time()

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        video.unsubscribe(handle)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()