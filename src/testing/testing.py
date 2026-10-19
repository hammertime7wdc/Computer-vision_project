import argparse
import sys
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# --- config -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = PROJECT_ROOT / "models" / "best.pt"
CONF = 0.8                           # min confidence to count a sign (classifier: softmax score)
SMOOTH = 4                           # same sign for N frames before a command fires

# camera source: a number = USB/built-in webcam (or a phone shown as a webcam by DroidCam / Camo / iVCam / Iriun),
# or a stream URL from a phone app on the same Wi-Fi, e.g.
#   Android "IP Webcam"  ->  "http://192.168.1.23:8080/video"
#   DroidCam (Wi-Fi)     ->  "http://192.168.1.23:4747/video"
# You can also pass it on the command line:  python testing.py "http://192.168.1.23:8080/video"
SOURCE = 0
ROTATE = 0                           # 0, 90, 180 or 270 (clockwise). A phone held upright needs 90 or 270. Key r cycles it.
MAX_WIDTH = 1280                     # bigger frames (phone 1080p / 4K) are shrunk to this width so scanning stays fast
NONE = "none"                        # the "no command sign" class of the classifier

# class name -> command. Both naming schemes are listed so either model works.
CMD = {
    "stop": "S",
    "no_left_turn": "L", "turn_left": "L",
    "no_right_turn": "R", "turn_right": "R",
    "go_ahead": "F",
    "no_entry": "N",
}

# multi-sign scan (classifier only): the classifier is run on many square windows and every
# confident window is a sign. Window size is a fraction of the frame's short side.
SCALES = (1.0, 0.7, 0.5)             # small windows find small / far signs, big ones find close signs
OVERLAP = 0.5                        # 0.5 = windows slide by half their size
NMS_IOU = 0.3                        # same class windows overlapping more than this are merged

WIN = "test (q quit, f fullscreen, m scan/centre, r rotate)"
# -----------------------------------------------------------------------------

model = None


def put(img, text, org, color, scale=1.0):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def nms(hits):
    """hits: list of (name, conf, (x1, y1, x2, y2)). Keep the best box per sign, per class."""
    keep = []
    for h in sorted(hits, key=lambda t: -t[1]):
        if all(h[0] != k[0] or iou(h[2], k[2]) < NMS_IOU for k in keep):
            keep.append(h)
    return keep


def make_windows(w, h):
    """Square windows (x1, y1, x2, y2) covering the frame at every scale."""
    wins = []
    for sc in SCALES:
        s = max(32, int(min(w, h) * sc))
        step = max(1, int(s * (1 - OVERLAP)))
        xs = list(range(0, w - s + 1, step))
        ys = list(range(0, h - s + 1, step))
        if xs[-1] != w - s:
            xs.append(w - s)
        if ys[-1] != h - s:
            ys.append(h - s)
        wins += [(x, y, x + s, y + s) for y in ys for x in xs]
    return wins


def draw_hits(img, hits):
    for name, p, (x1, y1, x2, y2) in hits:
        color = (0, 255, 0) if name in CMD else (0, 165, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        put(img, f"{name} {p:.2f}" + (f" -> {CMD[name]}" if name in CMD else ""),
            (x1 + 6, max(y1 + 28, 28)), color, 0.8)


def detect(frame):
    """Detector weights: real boxes on the full frame, any number of signs."""
    r = model.predict(frame, conf=CONF, verbose=False)[0]
    hits = [(model.names[int(b.cls[0])], float(b.conf[0]), tuple(int(v) for v in b.xyxy[0]))
            for b in r.boxes]
    out = frame.copy()
    draw_hits(out, hits)
    return out, hits


def classify_scan(frame):
    """Classifier weights, multi-sign: run the classifier on all windows, keep confident non-none ones."""
    h, w = frame.shape[:2]
    wins = make_windows(w, h)
    results = model.predict([frame[y1:y2, x1:x2] for x1, y1, x2, y2 in wins], verbose=False)
    hits = []
    for box, r in zip(wins, results):
        probs = r.probs.data.cpu().numpy()
        top = int(np.argmax(probs))
        name, p = model.names[top], float(probs[top])
        if name != NONE and p >= CONF:
            hits.append((name, p, box))
    hits = nms(hits)
    out = frame.copy()
    draw_hits(out, hits)
    put(out, f"scan: {len(wins)} windows", (20, 90), (200, 200, 200), 0.7)
    return out, hits


def classify_centre(frame):
    """Classifier weights, single sign: centre square only, with a bar per class."""
    h, w = frame.shape[:2]
    s = min(h, w)
    x0 = (w - s) // 2
    probs = model.predict(frame[:s, x0:x0 + s], verbose=False)[0].probs.data.cpu().numpy()
    out = frame.copy()
    cv2.rectangle(out, (x0, 0), (x0 + s - 1, s - 1), (255, 255, 255), 2)
    for i, p in enumerate(probs):
        y = 120 + i * 34
        cv2.rectangle(out, (20, y), (20 + int(300 * float(p)), y + 22), (0, 200, 255), -1)
        cv2.putText(out, f"{model.names[i]} {float(p):.2f}", (330, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    top = int(np.argmax(probs))
    name, p = model.names[top], float(probs[top])
    hits = [(name, p, (x0, 0, x0 + s, s))] if (name != NONE and p >= CONF) else []
    return out, hits


def pick_command_sign(hits):
    """Several signs in view: the robot follows the largest command sign (the closest one)."""
    cands = [h for h in hits if h[0] in CMD]
    if not cands:
        return None
    return max(cands, key=lambda h: (h[2][2] - h[2][0]) * (h[2][3] - h[2][1]))


ROT_CODE = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def prepare(frame, rotate):
    """Rotate (phone held upright) and shrink big phone frames."""
    if rotate in ROT_CODE:
        frame = cv2.rotate(frame, ROT_CODE[rotate])
    h, w = frame.shape[:2]
    if w > MAX_WIDTH:
        frame = cv2.resize(frame, (MAX_WIDTH, int(h * MAX_WIDTH / w)), interpolation=cv2.INTER_AREA)
    return frame


class Stream:
    """Reads the camera in a background thread and always hands out the NEWEST frame.
    Without this a network camera buffers old frames and the picture lags more and more
    because the scan is slower than the camera."""

    def __init__(self, cap):
        self.cap = cap
        self.frame, self.new, self.alive = None, False, True
        self.lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        fails = 0
        while self.alive:
            ok, f = self.cap.read()
            if not ok:
                fails += 1
                if fails > 50:           # ~2.5 s of nothing: the stream is gone
                    self.alive = False
                time.sleep(0.05)
                continue
            fails = 0
            with self.lock:
                self.frame, self.new = f, True

    def read(self, timeout=5.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.lock:
                if self.new:
                    self.new = False
                    return True, self.frame
            if not self.alive:
                break
            time.sleep(0.003)
        return False, None

    def release(self):
        self.alive = False
        time.sleep(0.1)
        self.cap.release()


class LocalCam:
    """Plain read() wrapper so local webcams and streams look the same to main()."""

    def __init__(self, cap):
        self.cap = cap

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


def open_source(src):
    """Open a webcam index or a phone stream URL. Returns an object with read() / release()."""
    if isinstance(src, str) and "://" in src:
        cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open stream {src}\n"
                "- phone and PC must be on the same Wi-Fi, and the phone app must be running/streaming\n"
                "- Android IP Webcam: http://<phone-ip>:8080/video   DroidCam: http://<phone-ip>:4747/video\n"
                "- open the same URL in a browser first to check it works")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return Stream(cap)
    index = int(src)
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF] if sys.platform == "win32" else [cv2.CAP_ANY]
    for be in backends:
        cap = cv2.VideoCapture(index, be)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            for _ in range(10):          # warm up, first frames can be black
                cap.read()
            return LocalCam(cap)
        cap.release()
    raise RuntimeError(f"Camera {index} did not open - close other apps using it, or try another index "
                       "(python testing.py --list shows which ones work)")


def list_cameras(n=6):
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF] if sys.platform == "win32" else [cv2.CAP_ANY]
    for i in range(n):
        for be in backends:
            cap = cv2.VideoCapture(i, be)
            ok = cap.isOpened() and cap.read()[0]
            cap.release()
            if ok:
                print(f"camera index {i}: works")
                break
        else:
            print(f"camera index {i}: no")


def main():
    global model
    ap = argparse.ArgumentParser()
    ap.add_argument("source", nargs="?", default=None,
                    help="webcam index or phone stream URL (default: SOURCE in the file)")
    ap.add_argument("--weights", default=str(WEIGHTS),
                    help="path to a YOLO model file (default: models/best.pt)")
    ap.add_argument("--rotate", type=int, default=None, choices=[0, 90, 180, 270])
    ap.add_argument("--list", action="store_true", help="show which webcam indexes work and exit")
    args = ap.parse_args()
    if args.list:
        list_cameras()
        return
    src = args.source if args.source is not None else SOURCE
    rotate = ROTATE if args.rotate is None else args.rotate
    weights = Path(args.weights)
    if not weights.is_absolute():
        weights = PROJECT_ROOT / weights
    if not weights.exists():
        raise FileNotFoundError(
            f"Model weights not found: {weights}. "
            "Choose a file in models/ with --weights."
        )
    print("Using weights:", weights)

    model = YOLO(weights)
    is_cls = model.task == "classify"
    print("Task:", model.task, "| classes:", model.names)
    unknown = [c for c in CMD if c not in model.names.values()]
    if len(unknown) == len(CMD):
        print("WARNING: none of the CMD class names exist in this model - no command will ever fire")

    cap = open_source(src)
    print("Camera source:", src)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    fullscreen = True
    scan = True                          # classifier: True = multi-sign scan, False = centre square

    hist = deque(maxlen=SMOOTH)
    last = None                          # last command that fired
    t_prev = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("No frame from the camera (stream lost?)")
                break
            frame = prepare(frame, rotate)
            fn = detect if not is_cls else (classify_scan if scan else classify_centre)
            out, hits = fn(frame)

            target = pick_command_sign(hits)
            label = target[0] if target else None
            hist.append(label)
            if len(hist) == hist.maxlen:
                if hist[0] in CMD and len(set(hist)) == 1:
                    if CMD[hist[0]] != last:
                        last = CMD[hist[0]]
                        print("CMD", last, f"({hist[0]}, {len(hits)} sign(s) in view)")
                elif all(h is None for h in hist):
                    last = None          # sign gone -> the next sign can trigger again

            now = time.perf_counter()
            fps, t_prev = 1.0 / max(now - t_prev, 1e-6), now
            put(out, f"signs: {len(hits)}   following: {label or '-'}   fps {fps:.0f}", (20, 35), (255, 255, 255), 0.9)
            put(out, f"cmd: {last}", (20, 65), (0, 255, 0) if last else (180, 180, 180), 0.8)
            cv2.imshow(WIN, out)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                rotate = (rotate + 90) % 360
                hist.clear()
            elif key == ord("m"):
                scan = not scan
                hist.clear()
            elif key == ord("f"):
                fullscreen = not fullscreen
                cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()