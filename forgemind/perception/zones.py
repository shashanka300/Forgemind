"""Zones from printed AprilTag (36h11) markers.

The phone knows where it is from the most recent marker it saw; the static
camera's zone is fixed. No SLAM: a marker per place is enough for "the key is
on the shelf". Print the markers with `forgemind tags`.
"""
import time

import numpy as np

ZONE_TIMEOUT_S = 300


def _detector():
    import cv2
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    return cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())


class ZoneTracker:
    def __init__(self, zones: dict[int, str]):
        self.zones = zones
        self._det = None
        self.zone = None
        self.since = 0.0

    def detect_markers(self, image):
        import cv2
        if self._det is None:
            self._det = _detector()
        gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        _, ids, _ = self._det.detectMarkers(gray)
        return [int(i) for i in ids.flatten()] if ids is not None else []

    def update(self, image, ts=None):
        """Detect markers; return (zone, changed)."""
        ts = ts or time.time()
        found = [self.zones[i] for i in self.detect_markers(image) if i in self.zones]
        prev = self.current(ts)
        if found:
            self.zone, self.since = found[0], ts
        return self.current(ts), self.current(ts) != prev

    def set(self, zone, ts=None):
        self.zone, self.since = zone, ts or time.time()

    def current(self, ts=None):
        if self.zone and (ts or time.time()) - self.since <= ZONE_TIMEOUT_S:
            return self.zone
        return None


def make_tag_sheet(zones: dict[int, str], out_path, px=600):
    """A printable PNG with one labelled marker per zone."""
    import cv2
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    tiles = []
    for mid, name in sorted(zones.items()):
        m = cv2.aruco.generateImageMarker(d, mid, px)
        tile = np.full((px + 200, px + 120), 255, np.uint8)
        tile[60:60 + px, 60:60 + px] = m
        cv2.putText(tile, f"{name}  (id {mid})", (60, px + 140), cv2.FONT_HERSHEY_SIMPLEX,
                    1.6, 0, 3, cv2.LINE_AA)
        tiles.append(tile)
    cols = 2
    rows = [np.hstack(tiles[i:i + cols] + [np.full_like(tiles[0], 255)] * (cols - len(tiles[i:i + cols])))
            for i in range(0, len(tiles), cols)]
    cv2.imwrite(str(out_path), np.vstack(rows))
    return out_path
