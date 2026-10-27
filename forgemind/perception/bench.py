"""The bench region: which part of the bench camera's view is the bench.

    FM_BENCH_ROI   x0,y0,x1,y1 (0..1 of the full frame) kept; default whole frame
    FM_BENCH_MASK  JSON list of [x0,y0,x1,y1] (0..1 of the full frame) painted out,
                   e.g. an RGB keyboard whose cycling lights trip the change gate

Bench frames are cropped to the ROI and the masks are painted flat grey as
they arrive, so everything downstream (change gate, segmenter, keyframes,
frames sent to the model API) sees only the bench. Boxes are relative to the
cropped image.

`forgemind bench-region <image or video>` suggests both from a sample frame.
"""
import json

from PIL import Image, ImageDraw

MASK_FILL = (128, 128, 128)
# Things on a desk that are not "the bench" for ForgeMind's purposes.
MASK_CONCEPTS = ("keyboard", "laptop", "computer monitor")
BENCH_CONCEPTS = ("desk", "table")


def parse_roi(text):
    if not text:
        return None
    vals = [float(v) for v in text.split(",")]
    if len(vals) != 4 or not (0 <= vals[0] < vals[2] <= 1 and 0 <= vals[1] < vals[3] <= 1):
        raise ValueError(f"FM_BENCH_ROI must be x0,y0,x1,y1 within 0..1, got {text!r}")
    return tuple(vals)


def parse_masks(text):
    if not text:
        return []
    masks = json.loads(text)
    for m in masks:
        if len(m) != 4:
            raise ValueError(f"FM_BENCH_MASK entries are [x0,y0,x1,y1], got {m!r}")
    return [tuple(float(v) for v in m) for m in masks]


def apply(image, roi=None, masks=()):
    """Crop to the ROI and paint the masks out. Returns a new image."""
    if not roi and not masks:
        return image
    W, H = image.size
    x0, y0, x1, y1 = roi or (0.0, 0.0, 1.0, 1.0)
    out = image.crop((int(x0 * W), int(y0 * H), int(x1 * W), int(y1 * H)))
    if masks:
        d = ImageDraw.Draw(out)
        rw, rh = x1 - x0, y1 - y0
        for m in masks:
            # full-frame coords -> crop coords, clipped to the crop
            a = max(0.0, (m[0] - x0) / rw), max(0.0, (m[1] - y0) / rh)
            b = min(1.0, (m[2] - x0) / rw), min(1.0, (m[3] - y0) / rh)
            if a[0] < b[0] and a[1] < b[1]:
                d.rectangle((a[0] * out.width, a[1] * out.height,
                             b[0] * out.width, b[1] * out.height), fill=MASK_FILL)
    return out


def suggest(image, segmenter, object_concepts=(), min_score=0.5, pad=0.01, with_roi=False,
            protect=0.3):
    """(roi, masks, dropped) from a sample frame.

    Masks: keyboard/laptop/monitor boxes, except any that would cover more than
    `protect` of a detected bench object (a laptop with boards lying on it is
    part of the bench). ROI: the desk's box, only if `with_roi` (useful for a
    camera that also sees the room or the user's face).
    """
    asked = list(MASK_CONCEPTS) + list(object_concepts) + (list(BENCH_CONCEPTS) if with_roi else [])
    dets = [d for d in segmenter.segment(image, asked) if d.score >= min_score]
    objects = [d.box for d in dets if d.concept in object_concepts
               and d.concept not in MASK_CONCEPTS]
    roi = None
    desks = [d for d in dets if d.concept in BENCH_CONCEPTS]
    if with_roi and desks:
        d = max(desks, key=lambda d: d.area)
        if d.area < 0.9:
            b = d.box
            roi = tuple(round(v, 3) for v in (max(0, b[0] - pad), max(0, b[1] - pad),
                                              min(1, b[2] + pad), min(1, b[3] + pad)))
    masks, dropped = [], []
    for d in dets:
        if d.concept not in MASK_CONCEPTS:
            continue
        m = tuple(round(v, 3) for v in (max(0, d.box[0] - pad), max(0, d.box[1] - pad),
                                        min(1, d.box[2] + pad), min(1, d.box[3] + pad)))
        hides = [o for o in objects if _covered(o, m) > protect]
        (dropped if hides else masks).append((d.concept, m) if hides else m)
    return roi, masks, dropped


def _covered(box, mask):
    area = max(1e-9, (box[2] - box[0]) * (box[3] - box[1]))
    w = min(box[2], mask[2]) - max(box[0], mask[0])
    h = min(box[3], mask[3]) - max(box[1], mask[1])
    return w * h / area if w > 0 and h > 0 else 0.0


def preview(image, roi, masks, out_path):
    """Full frame with the ROI outlined and masks shaded, next to what the pipeline sees."""
    W, H = image.size
    left = image.convert("RGB").copy()
    d = ImageDraw.Draw(left, "RGBA")
    for m in masks:
        d.rectangle((m[0] * W, m[1] * H, m[2] * W, m[3] * H), fill=(255, 0, 0, 90),
                    outline=(255, 0, 0, 255), width=4)
    if roi:
        d.rectangle((roi[0] * W, roi[1] * H, roi[2] * W, roi[3] * H),
                    outline=(0, 255, 0, 255), width=6)
    right = apply(image.convert("RGB"), roi, masks)
    scale = H / right.height
    right = right.resize((max(1, int(right.width * scale)), H))
    sheet = Image.new("RGB", (W + right.width + 20, H), (20, 20, 20))
    sheet.paste(left, (0, 0))
    sheet.paste(right, (W + 20, 0))
    sheet.thumbnail((2400, 2400))
    sheet.save(out_path, quality=88)
    return out_path
