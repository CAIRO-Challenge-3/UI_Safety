# -*- coding: utf-8 -*-
"""
PPE detection running ON Pepper (NAOqi 2.5, Python 2.7, 32-bit Atom).

Grabs frames from Pepper's top camera, runs a YOLO helmet model (exported by
export_pepper_onnx.py) through OpenCV's DNN module, and makes Pepper say a warning when someone
isn't wearing a helmet (demo scope: helmet / no helmet only). Pepper has no
display, so the annotated video (helmet boxes + "HELMET DETECTED" / "NO HELMET"
banner) is served as an MJPEG stream you open in a browser:
    http://<pepper-ip>:8080/

It can also annotate a single image file instead of the camera (--image), which
works on any machine with OpenCV >= 4.1.2, no robot needed.

Requirements on Pepper (installed to ~/.local, see README steps):
    numpy==1.16.6  opencv-python-headless==4.1.2.30

Export the model on your laptop first (a stock `yolo export` won't load in
OpenCV 4.1.2); it writes a folder of ONNX pieces + manifest.json:
    python export_pepper_onnx.py --weights models/helmet-26s.pt --out models/helmet_pepper

Usage (on Pepper):
    python ppe_pepper_onboard.py --model helmet_pepper            # live + stream
    python ppe_pepper_onboard.py --model helmet_pepper --image in.jpg --out out.jpg
"""

from __future__ import division, print_function

import argparse
import json
import os
import signal
import threading
import time
import traceback

import cv2
import numpy as np

try:
    from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
    from SocketServer import ThreadingMixIn
except ImportError:  # Python 3 (testing on a laptop)
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from socketserver import ThreadingMixIn

# NAOqi constants
TOP_CAMERA = 0
RES_VGA = 2            # 640x480
COLORSPACE_BGR = 13    # matches OpenCV's channel order
FPS = 5

# Demo scope: helmets only. Class names come from the model's manifest.json;
# these cover helmet-26s ("helmet" / "face" = bare head) and the older PPE model.
HELMET_LABELS = {"helmet", "Hardhat"}
NO_HELMET_LABELS = {"face", "head", "NO-Hardhat"}
SHOW_CLASSES = HELMET_LABELS | NO_HELMET_LABELS
VIOLATION_CLASSES = NO_HELMET_LABELS
WARNING_TEXT = "Please wear your helmet."
THANKS_TEXT = "Thank you for wearing your helmet."


def grab_frame(video, handle):
    img = video.getImageRemote(handle)
    if img is None:
        return None
    width, height, data = img[0], img[1], img[6]
    return np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3)).copy()


def letterbox(frame, size):
    """Resize keeping aspect ratio and pad to size x size (as Ultralytics does)."""
    h, w = frame.shape[:2]
    scale = min(size / float(w), size / float(h))
    nw, nh = int(round(w * scale)), int(round(h * scale))
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    left, top = (size - nw) // 2, (size - nh) // 2
    canvas[top:top + nh, left:left + nw] = cv2.resize(frame, (nw, nh))
    return canvas, scale, left, top


def softmax(x, axis):
    # ONNX opset < 13 semantics: flatten to 2D at `axis`, softmax over the second dim.
    flat = x.reshape(int(np.prod(x.shape[:axis])), -1)
    flat = np.exp(flat - flat.max(axis=1, keepdims=True))
    return (flat / flat.sum(axis=1, keepdims=True)).reshape(x.shape)


def run_numpy_op(op, tensors):
    """The few ONNX ops of an attention block, which OpenCV 4.1.2 can't run."""
    def get(name):
        if name in op["consts"]:
            return np.array(op["consts"][name], dtype=np.float32 if op["op"] == "Mul" else np.int64)
        return tensors[name]

    ins, attrs, kind = [get(n) for n in op["inputs"]], op["attrs"], op["op"]
    if kind == "Split":
        sizes = attrs.get("split") or [ins[0].shape[attrs.get("axis", 0)] // len(op["outputs"])] * len(op["outputs"])
        outs = np.split(ins[0], np.cumsum(sizes)[:-1], axis=attrs.get("axis", 0))
    elif kind == "Mul":
        outs = [ins[0] * ins[1]]
    elif kind == "Transpose":
        outs = [np.ascontiguousarray(ins[0].transpose(attrs["perm"]))]
    elif kind == "MatMul":
        outs = [np.matmul(ins[0], ins[1])]
    elif kind == "Softmax":
        outs = [softmax(ins[0], attrs.get("axis", 1))]
    elif kind == "Reshape":
        shape = [ins[0].shape[i] if s == 0 else int(s) for i, s in enumerate(ins[1])]
        outs = [np.ascontiguousarray(ins[0].reshape(shape))]
    else:
        raise ValueError("unsupported numpy op " + kind)
    for name, value in zip(op["outputs"], outs):
        tensors[name] = value.astype(np.float32)


def native_str(obj):
    """json gives unicode on Python 2; OpenCV's Python 2 bindings want str."""
    if isinstance(obj, dict):
        return dict((native_str(k), native_str(v)) for k, v in obj.items())
    if isinstance(obj, list):
        return [native_str(v) for v in obj]
    if type(obj).__name__ == "unicode":
        return obj.encode("utf-8")
    return obj


class YoloPipeline(object):
    """Runs a model folder from export_pepper_onnx.py: OpenCV segments + numpy attention."""

    def __init__(self, folder):
        with open(os.path.join(folder, "manifest.json")) as f:
            self.manifest = native_str(json.load(f))
        self.names = self.manifest["names"]
        self.size = self.manifest["imgsz"]
        self.steps = self.manifest["steps"]
        self.nets = {i: cv2.dnn.readNetFromONNX(os.path.join(folder, s["file"]))
                     for i, s in enumerate(self.steps) if s["kind"] == "opencv"}

    def forward(self, blob):
        tensors = {"images": blob}
        for i, step in enumerate(self.steps):
            if step["kind"] == "numpy":
                for op in step["ops"]:
                    run_numpy_op(op, tensors)
                continue
            net = self.nets[i]
            for name in step["inputs"]:
                net.setInput(tensors[name], name)
            for name, value in zip(step["outputs"], net.forward(step["forward"])):
                tensors[name] = value
        return tensors


def decode_head(tensors, manifest):
    """Decode the raw per-stride head maps (box regression + class logits).

    Returns xyxy boxes in letterboxed-input pixels and per-class scores.
    """
    reg_max = manifest["reg_max"]
    all_boxes, all_scores = [], []
    for level in manifest["head"]:
        box_map, cls_map, stride = tensors[level["box"]], tensors[level["cls"]], level["stride"]
        h, w = box_map.shape[2:]
        if reg_max > 1:
            # Distribution focal loss (YOLOv8/11): softmax over bins, then expectation.
            dfl = softmax(box_map[0].reshape(4, reg_max, h * w).transpose(0, 2, 1), 2)
            ltrb = (dfl * np.arange(reg_max, dtype=np.float32)).sum(axis=2)
        else:
            ltrb = box_map[0].reshape(4, h * w)       # YOLO26: distances directly
        ys, xs = np.mgrid[0:h, 0:w]
        ax, ay = xs.ravel() + 0.5, ys.ravel() + 0.5
        all_boxes.append(np.stack([ax - ltrb[0], ay - ltrb[1], ax + ltrb[2], ay + ltrb[3]], 1) * stride)
        all_scores.append(1 / (1 + np.exp(-cls_map[0].reshape(-1, h * w).T)))
    return np.concatenate(all_boxes), np.concatenate(all_scores)


def detect(model, frame, conf, nms):
    size = model.size
    img, scale, dx, dy = letterbox(frame, size)
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (size, size), swapRB=True, crop=False)
    xyxy, scores = decode_head(model.forward(blob), model.manifest)

    cls_ids = scores.argmax(axis=1)
    best = scores.max(axis=1)
    keep = best > conf
    xyxy, cls_ids, best = xyxy[keep], cls_ids[keep], best[keep]

    boxes = []
    for x1, y1, x2, y2 in xyxy:
        boxes.append([int((x1 - dx) / scale), int((y1 - dy) / scale),
                      int((x2 - x1) / scale), int((y2 - y1) / scale)])

    if not boxes:
        return []
    idx = np.array(cv2.dnn.NMSBoxes(boxes, best.tolist(), conf, nms)).flatten()
    detections = [(model.names[int(cls_ids[i])], float(best[i]), boxes[i]) for i in idx]
    return [d for d in detections if d[0] in SHOW_CLASSES]


def helmet_status(detections):
    """(banner text, BGR colour) for the helmet-only demo."""
    labels = set(d[0] for d in detections)
    if labels & NO_HELMET_LABELS:
        return "NO HELMET", (0, 0, 200)
    if labels & HELMET_LABELS:
        return "HELMET DETECTED", (0, 160, 0)
    return "No person in view", (90, 90, 90)


def draw(frame, detections, fps=None):
    """Overlay helmet boxes and a 'helmet detected or not' banner on the frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    for label, score, (x, y, w, h) in detections:
        ok = label in HELMET_LABELS
        color = (0, 200, 0) if ok else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        text = "Helmet" if ok else "No helmet"
        (tw, th), _ = cv2.getTextSize(text, font, 0.5, 1)
        ty = max(y, th + 6)
        cv2.rectangle(frame, (x, ty - th - 6), (x + tw + 4, ty), color, -1)
        cv2.putText(frame, text, (x + 2, ty - 4), font, 0.5, (255, 255, 255), 1)

    status, color = helmet_status(detections)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), color, -1)
    cv2.putText(frame, status, (10, 29), font, 0.9, (255, 255, 255), 2)
    if fps is not None:
        cv2.putText(frame, "%.1f det/s" % fps, (frame.shape[1] - 100, 27), font, 0.5, (255, 255, 255), 1)
    return frame


class FrameBuffer(object):
    """Latest annotated JPEG, shared between the detection loop and HTTP clients."""

    def __init__(self):
        self.jpeg = None
        self.cond = threading.Condition()

    def put(self, frame):
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ok:
            with self.cond:
                self.jpeg = buf.tobytes()
                self.cond.notify_all()

    def wait(self, timeout=5.0):
        with self.cond:
            self.cond.wait(timeout)
            return self.jpeg


PAGE = (b"<html><head><title>Pepper PPE</title></head>"
        b"<body style='margin:0;background:#111;text-align:center'>"
        b"<img src='/stream' style='max-width:100%;height:auto'></body></html>")


def start_stream_server(frames, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(PAGE)
            elif self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        jpeg = frames.wait()
                        if jpeg is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                        self.wfile.write(jpeg)
                        self.wfile.write(b"\r\n")
                except Exception:  # browser closed the tab
                    pass
            else:
                self.send_error(404)

    class Server(ThreadingMixIn, HTTPServer):
        daemon_threads = True
        allow_reuse_address = True

    server = Server(("0.0.0.0", port), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.daemon = True
    thread.start()
    return server


class AsyncDetector(object):
    """Runs detection on the newest frame in a background thread so the live view
    keeps its frame rate while the (slow) model works. OpenCV releases the GIL."""

    def __init__(self, model, conf, nms):
        self.model, self.conf, self.nms = model, conf, nms
        self.lock = threading.Lock()
        self.frame = None
        self.result = (0, [], 0.0)      # (sequence no., detections, seconds per detection)
        self.error = None
        self.running = True
        thread = threading.Thread(target=self._run)
        thread.daemon = True
        thread.start()

    def submit(self, frame):
        with self.lock:
            self.frame = frame

    def latest(self):
        with self.lock:
            return self.result

    def _run(self):
        seq = 0
        while self.running:
            with self.lock:
                frame, self.frame = self.frame, None
            if frame is None:
                time.sleep(0.01)
                continue
            try:
                start = time.time()
                detections = detect(self.model, frame, self.conf, self.nms)
            except Exception:
                self.error = traceback.format_exc()
                return
            seq += 1
            with self.lock:
                self.result = (seq, detections, time.time() - start)


def annotate_image(model, args):
    frame = cv2.imread(args.image)
    if frame is None:
        raise SystemExit("Could not read %s" % args.image)
    detections = detect(model, frame, args.conf, args.nms)
    for label, score, box in detections:
        print("%-15s %.2f %s" % (label, score, box))
    out = args.out or os.path.splitext(args.image)[0] + "_ppe.jpg"
    cv2.imwrite(out, draw(frame, detections))
    print("Wrote %s" % out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="model folder from export_pepper_onnx.py")
    parser.add_argument("--ip", default="127.0.0.1", help="NAOqi address (localhost on the robot)")
    parser.add_argument("--port", type=int, default=9559)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--nms", type=float, default=0.45)
    parser.add_argument("--cooldown", type=float, default=8.0, help="seconds between warnings")
    parser.add_argument("--max-warnings", type=int, default=2,
                        help="warnings per violation, then stay quiet until the scene is clear")
    parser.add_argument("--clear-after", type=float, default=10.0,
                        help="seconds with nobody in view before warnings re-arm")
    parser.add_argument("--no-speak", action="store_true", help="don't let Pepper speak")
    parser.add_argument("--snapshots", default=None, help="folder to save violation images")
    parser.add_argument("--stream-port", type=int, default=8080, help="MJPEG viewer port, 0 = off")
    parser.add_argument("--image", default=None, help="annotate this image file instead of the camera")
    parser.add_argument("--out", default=None, help="output path for --image (default: <name>_ppe.jpg)")
    parser.add_argument("--threads", type=int, default=4, help="OpenCV threads (Pepper's Atom has 4 cores)")
    parser.add_argument("--view-fps", type=float, default=8.0, help="frame rate of the live view")
    args = parser.parse_args()

    if not hasattr(cv2, "dnn") or not hasattr(cv2.dnn, "readNetFromONNX"):
        raise SystemExit("OpenCV %s has no ONNX support (loaded from %s). "
                         "Put ~/.local/lib/python2.7/site-packages first on PYTHONPATH."
                         % (cv2.__version__, cv2.__file__))

    cv2.setNumThreads(args.threads)
    model = YoloPipeline(args.model)
    print("Loaded %s (classes %s, imgsz %d) with OpenCV %s, %d threads"
          % (args.model, model.names, model.size, cv2.__version__, cv2.getNumThreads()))

    if args.image:
        annotate_image(model, args)
        return

    from naoqi import ALProxy  # only needed for the live camera

    video = ALProxy("ALVideoDevice", args.ip, args.port)
    tts = None if args.no_speak else ALProxy("ALTextToSpeech", args.ip, args.port)
    if args.snapshots and not os.path.isdir(args.snapshots):
        os.makedirs(args.snapshots)

    frames_out = FrameBuffer()
    server = None
    if args.stream_port:
        server = start_stream_server(frames_out, args.stream_port)
        print("Live view: http://<pepper-ip>:%d/" % args.stream_port)

    # Turn `kill <pid>` into a clean shutdown so the camera gets unsubscribed.
    def on_term(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, on_term)

    handle = video.subscribeCamera("ppe_onboard", TOP_CAMERA, RES_VGA, COLORSPACE_BGR, FPS)
    detector = AsyncDetector(model, args.conf, args.nms)
    last_warning = 0.0
    last_person_seen = 0.0
    warnings_given = 0
    helmet_confirmed = False
    last_seq = 0
    next_frame = 0.0

    try:
        while True:
            # Pace the live view so it doesn't steal CPU from the detector.
            time.sleep(max(0.0, next_frame - time.time()))
            next_frame = time.time() + 1.0 / args.view_fps
            frame = grab_frame(video, handle)
            if frame is None:
                continue
            detector.submit(frame)
            if detector.error:
                raise SystemExit("Detection thread crashed:\n" + detector.error)

            # The video runs at --view-fps; boxes come from the latest finished
            # detection, so they can trail a moving person by one detection.
            seq, detections, det_seconds = detector.latest()
            det_fps = 1.0 / det_seconds if det_seconds else None
            annotated = draw(frame, detections, det_fps)
            frames_out.put(annotated)

            if seq == last_seq:          # no new detection result since last frame
                continue
            last_seq = seq
            violations = sorted(set(d[0] for d in detections if d[0] in VIOLATION_CLASSES))
            labels = ", ".join("%s %.2f" % (d[0], d[1]) for d in detections) or "-"
            print("[%.1f det/s] %s | %s" % (det_fps or 0, helmet_status(detections)[0], labels))

            # Warn at most --max-warnings times per visit, and never again once a
            # helmet has been seen (a helmeted head flickering to "no helmet" for a
            # frame must not trigger it). A visit ends after --clear-after seconds
            # with nobody detected; that re-arms everything for the next person.
            now = time.time()
            if detections:
                if now - last_person_seen > args.clear_after:
                    warnings_given, helmet_confirmed = 0, False
                last_person_seen = now
            if any(d[0] in HELMET_LABELS for d in detections) and not helmet_confirmed:
                helmet_confirmed = True
                print("Helmet detected - warnings off until the scene is clear.")
                if warnings_given and tts:       # only thank someone who was warned
                    tts.post.say(THANKS_TEXT)
            if (violations and not helmet_confirmed and warnings_given < args.max_warnings
                    and now - last_warning > args.cooldown):
                last_warning = now
                warnings_given += 1
                if tts:
                    tts.post.say(WARNING_TEXT)
                if args.snapshots:
                    name = time.strftime("violation_%Y%m%d_%H%M%S.jpg")
                    cv2.imwrite(os.path.join(args.snapshots, name), annotated)
    except KeyboardInterrupt:
        pass
    finally:
        detector.running = False
        video.unsubscribe(handle)
        if server:
            server.shutdown()
        print("Stopped.")


if __name__ == "__main__":
    main()
