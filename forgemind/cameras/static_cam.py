"""Static bench camera client: a USB webcam (or RTSP/HTTP stream) -> ForgeMind.

    forgemind static-cam --device 0 --fps 1
    forgemind static-cam --device rtsp://192.168.1.50:8554/live --server http://192.168.1.10:8770

The server does the change gating; this just sends a JPEG every 1/fps
seconds. Runs on the machine the webcam is plugged into (no browser, so no
HTTPS needed). An old phone can be the static camera instead: open the
phone link with `&role=static`.
"""
import time
import urllib.error
import urllib.request

FRAME_PX = 1280


def run(server, token, device=0, fps=1.0, show=False):
    import cv2
    src = int(device) if str(device).isdigit() else device
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera {device}")
    url = server.rstrip("/") + "/api/frame?camera=static"
    headers = {"Content-Type": "image/jpeg"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    period = 1.0 / max(0.1, fps)
    print(f"static camera {device} -> {url} at {fps} fps (Ctrl+C to stop)")
    sent = 0
    try:
        while True:
            t = time.time()
            ok, frame = cap.read()
            if not ok:
                print("camera read failed; retrying")
                time.sleep(1)
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
        cap.release()
        print(f"sent {sent} frames")
