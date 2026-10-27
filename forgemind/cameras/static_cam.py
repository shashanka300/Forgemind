"""Static bench camera client: a USB webcam (or RTSP/HTTP stream, or a video
file for rehearsals) -> ForgeMind.

    forgemind static-cam --device 0 --fps 1
    forgemind static-cam --device rtsp://192.168.1.50:8554/live --server http://192.168.1.10:8770
    forgemind static-cam --device bench-recording.mp4 --loop      # replay a recording live

The server does the change gating; this just sends the newest frame every
1/fps seconds. A reader thread drains the camera continuously so the frame
sent is the current one (OpenCV buffers several frames, which otherwise
arrive seconds late). Video files play at their own speed, so a recorded
bench session can stand in for the live camera. Runs on the machine the
webcam is plugged into (no browser, so no HTTPS needed). An old phone can be
the static camera instead: open the phone link with `&role=static`.
"""
import os
import threading
import time
import urllib.error
import urllib.request

FRAME_PX = 1280


class LatestFrame:
    """Reads a camera/stream/file in the background; keeps only the newest frame."""

    def __init__(self, src, loop=False):
        import cv2
        self.cv2 = cv2
        self.src = src
        self.is_file = isinstance(src, str) and os.path.isfile(src)
        self.loop = loop
        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            raise SystemExit(f"could not open camera {src}")
        if not self.is_file:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.frame, self.ended, self.failures = None, False, 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name="camera-reader")
        self.thread.start()

    def _run(self):
        t0, n = time.time(), 0
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                if self.is_file and self.loop:
                    self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, 0)
                    t0, n = time.time(), 0
                    continue
                if self.is_file:
                    self.ended = True
                    return
                self.failures += 1
                time.sleep(0.2)
                continue
            with self._lock:
                self.frame = frame
            if self.is_file:                      # play at the file's own speed
                n += 1
                wait = t0 + n / self.fps - time.time()
                if wait > 0:
                    time.sleep(wait)

    def latest(self):
        with self._lock:
            return None if self.frame is None else self.frame.copy()

    def close(self):
        self._stop.set()
        self.thread.join(1)
        self.cap.release()


def run(server, token, device=0, fps=1.0, show=False, loop=False):
    import cv2
    src = int(device) if str(device).isdigit() else device
    cam = LatestFrame(src, loop=loop)
    url = server.rstrip("/") + "/api/frame?camera=static"
    headers = {"Content-Type": "image/jpeg"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    period = 1.0 / max(0.1, fps)
    kind = "file" if cam.is_file else "camera"
    print(f"static {kind} {device} -> {url} at {fps} fps (Ctrl+C to stop)")
    sent = 0
    try:
        while not cam.ended:
            t = time.time()
            frame = cam.latest()
            if frame is None:
                time.sleep(0.1)
                continue
            h, w = frame.shape[:2]
            if max(h, w) > FRAME_PX:
                s = FRAME_PX / max(h, w)
                frame = cv2.resize(frame, (int(w * s), int(h * s)))
            ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            req = urllib.request.Request(url, data=jpg.tobytes(), headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=15) as r:
                    body = r.read().decode()
                sent += 1
                if '"events": [{' in body or '"events":[{' in body:
                    print(f"[{time.strftime('%H:%M:%S')}] {body[:300]}")
            except urllib.error.HTTPError as e:
                print(f"server said {e.code}: {e.read()[:200]!r}")
            except urllib.error.URLError as e:
                print(f"server unreachable: {e.reason}")
            if show:
                cv2.imshow("ForgeMind static camera", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            time.sleep(max(0.0, period - (time.time() - t)))
    except KeyboardInterrupt:
        pass
    finally:
        cam.close()
        print(f"sent {sent} frames" + (" (end of file)" if cam.ended else ""))
    return sent
