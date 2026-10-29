# -*- coding: utf-8 -*-
"""
Personalized greeting running ON Pepper (NAOqi 2.5, Python 2.7).

The laptop version (face_greeting.py) uses cv2.FaceDetectorYN / FaceRecognizerSF,
which need OpenCV >= 4.5.4; Pepper only has 4.1.2. So this version uses
Pepper's built-in face recognition (ALFaceDetection): no models to copy, and
faces are learned live in front of the robot.

Enroll (stand in front of Pepper, one person in view):
    python face_greeting_pepper.py --enroll Alice
Manage:
    python face_greeting_pepper.py --list
    python face_greeting_pepper.py --forget Alice
Run:
    python face_greeting_pepper.py
Live view with face boxes and names (same viewer as the PPE script):
    http://<pepper-ip>:8081/
"""

from __future__ import division, print_function

import argparse
import signal
import time
from datetime import datetime

import cv2
import numpy as np
from naoqi import ALProxy

from ppe_pepper_onboard import (COLORSPACE_BGR, FrameBuffer, RES_VGA, TOP_CAMERA,
                                grab_frame, start_stream_server)

GREEN = (60, 200, 60)      # recognised
ORANGE = (40, 140, 240)    # unknown

# Pepper top camera field of view, used only if ALVideoDevice can't convert angles.
HFOV, VFOV = np.radians(57.2), np.radians(44.3)


def greeting_for(name):
    hour = datetime.now().hour
    part = "morning" if hour < 12 else "afternoon" if hour < 17 else "evening"
    return "Good %s, %s!" % (part, name)


def read_faces(memory, last_stamp):
    """Return (stamp, [(label, score, alpha, beta, sizeX, sizeY), ...]) for a new
    FaceDetected value, or (last_stamp, None) if nothing new arrived."""
    value = memory.getData("FaceDetected")
    if not value or len(value) < 2:
        return last_stamp, []
    stamp = tuple(value[0])
    if stamp == last_stamp:
        return last_stamp, None
    faces = []
    for face_info in value[1][:-1]:          # last entry is Time_Filtered_Reco_Info
        shape, extra = face_info[0], face_info[1]
        _, alpha, beta, size_x, size_y = shape[:5]
        faces.append((extra[2], extra[1], alpha, beta, size_x, size_y))
    return stamp, faces


def face_box(video, face, width, height):
    """Camera angles -> pixel box (x, y, w, h)."""
    _, _, alpha, beta, size_x, size_y = face
    try:
        x, y, w, h = video.getImageInfoFromAngularInfo(TOP_CAMERA, [alpha, beta, size_x, size_y])
    except Exception:
        x, y = 0.5 - alpha / HFOV, 0.5 + beta / VFOV
        w, h = size_x / HFOV, size_y / VFOV
    return (int((x - w / 2) * width), int((y - h / 2) * height), int(w * width), int(h * height))


def draw(frame, boxes, banner):
    font = cv2.FONT_HERSHEY_SIMPLEX
    for label, (x, y, w, h) in boxes:
        colour = GREEN if label != "Unknown" else ORANGE
        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
        (tw, th), _ = cv2.getTextSize(label, font, 0.6, 1)
        ty = max(y, th + 8)
        cv2.rectangle(frame, (x, ty - th - 8), (x + tw + 8, ty), colour, -1)
        cv2.putText(frame, label, (x + 4, ty - 5), font, 0.6, (255, 255, 255), 1)
    if banner:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 36), (60, 60, 60), -1)
        cv2.putText(frame, banner, (10, 25), font, 0.8, (255, 255, 255), 2)
    return frame


def enroll(face_detection, memory, name, timeout):
    print("Look at Pepper's eyes, %s. Waiting for a single face..." % name)
    deadline, stamp, last_msg = time.time() + timeout, None, None
    while time.time() < deadline:
        stamp, faces = read_faces(memory, stamp)
        msg = None
        if faces and len(faces) == 1:
            if face_detection.learnFace(name):
                print("Learned %s." % name)
                return True
            msg = "Pepper couldn't learn the face, keep still and try again."
        elif faces:
            msg = "%d faces in view, only %s should be in front of Pepper." % (len(faces), name)
        if msg and msg != last_msg:
            print(msg)
            last_msg = msg
        time.sleep(0.5)
    print("Gave up after %ds without a usable face." % timeout)
    return False


def main():
    ap = argparse.ArgumentParser(description="Personalized greeting on Pepper")
    ap.add_argument("--ip", default="127.0.0.1", help="NAOqi address (localhost on the robot)")
    ap.add_argument("--port", type=int, default=9559)
    ap.add_argument("--enroll", metavar="NAME", help="learn the face in front of Pepper, then exit")
    ap.add_argument("--forget", metavar="NAME", help="remove a learned person, then exit")
    ap.add_argument("--list", action="store_true", help="list learned people, then exit")
    ap.add_argument("--cooldown", type=float, default=60.0,
                    help="seconds a person must be out of view before being greeted again")
    ap.add_argument("--min-score", type=float, default=0.0,
                    help="ignore recognitions below this score (Pepper reports 0..1)")
    ap.add_argument("--no-speak", action="store_true", help="don't let Pepper speak")
    ap.add_argument("--stream-port", type=int, default=8081, help="MJPEG viewer port, 0 = off")
    args = ap.parse_args()

    face_detection = ALProxy("ALFaceDetection", args.ip, args.port)
    memory = ALProxy("ALMemory", args.ip, args.port)

    if args.list:
        print("Learned people:", face_detection.getLearnedFacesList() or "none yet")
        return
    if args.forget:
        print("Forgot %s." % args.forget if face_detection.forgetPerson(args.forget)
              else "%s was not learned." % args.forget)
        return

    face_detection.setRecognitionEnabled(True)
    face_detection.subscribe("face_greeting", 500, 0.0)   # period ms, precision

    def on_term(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_term)

    if args.enroll:
        try:
            enroll(face_detection, memory, args.enroll, timeout=30)
        finally:
            face_detection.unsubscribe("face_greeting")
        return

    print("Learned people:", face_detection.getLearnedFacesList() or "none yet")
    tts = None if args.no_speak else ALProxy("ALAnimatedSpeech", args.ip, args.port)

    video = handle = server = None
    frames_out = FrameBuffer()
    if args.stream_port:
        video = ALProxy("ALVideoDevice", args.ip, args.port)
        handle = video.subscribeCamera("face_greeting_view", TOP_CAMERA, RES_VGA, COLORSPACE_BGR, 5)
        server = start_stream_server(frames_out, args.stream_port)
        print("Live view: http://<pepper-ip>:%d/" % args.stream_port)

    last_seen = {}
    banner, banner_until = "", 0.0
    stamp, faces = None, []
    try:
        while True:
            stamp, new_faces = read_faces(memory, stamp)
            if new_faces is not None:
                faces = new_faces
            now = time.time()

            for label, score, _, _, _, _ in (new_faces or []):
                if not label or score < args.min_score:
                    continue
                # Greet once per visit: someone who stays in view is not greeted
                # again; they have to be gone for --cooldown seconds first.
                away_for = now - last_seen.get(label, 0)
                last_seen[label] = now
                if away_for > args.cooldown:
                    banner, banner_until = greeting_for(label), now + 4
                    print("%s (score %.2f)" % (banner, score))
                    if tts:
                        tts.post.say(banner)

            if video:
                frame = grab_frame(video, handle)
                if frame is not None:
                    h, w = frame.shape[:2]
                    boxes = [(f[0] or "Unknown", face_box(video, f, w, h)) for f in faces]
                    frames_out.put(draw(frame, boxes, banner if now < banner_until else ""))
            else:
                time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        face_detection.unsubscribe("face_greeting")
        if video:
            video.unsubscribe(handle)
        if server:
            server.shutdown()
        print("Stopped.")


if __name__ == "__main__":
    main()
