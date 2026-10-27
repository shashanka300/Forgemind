"""ForgeMind command line.

    forgemind serve [--zrok]          start the server (phone link printed)
    forgemind static-cam --device 0   stream a USB webcam as the bench camera
    forgemind tags                    print-ready AprilTag sheet for your zones
    forgemind check                   show providers; test the model API + tool calling
"""
import argparse
import json
import sys
from urllib.parse import quote


def lan_ip():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def cmd_serve(args):
    import uvicorn

    from forgemind.config import Settings
    from forgemind.runtime import Runtime
    from forgemind.server.app import create_app
    over = {}
    if args.host:
        over["host"] = args.host
    if args.port:
        over["port"] = args.port
    if args.mock:
        over["llm_provider"] = "mock"
    s = Settings.from_env(**over)
    print("loading models (first run downloads CLIP, ~600 MB)...")
    rt = Runtime(s)
    print(json.dumps(rt.describe(), indent=2))
    suffix = f"/?t={quote(s.token)}" if s.token else "/"
    print(f"\n  on this computer:  http://localhost:{s.port}{suffix}")
    if s.host == "0.0.0.0":
        print(f"  same Wi-Fi:        http://{lan_ip()}:{s.port}{suffix}")
        print("  (browsers only allow the live camera over HTTPS or localhost; use --zrok for the phone)")
    print(f"  bench webcam:      forgemind static-cam --server http://localhost:{s.port} --token {s.token}\n")
    tunnel = None
    if args.zrok:
        from forgemind.server.tunnel import start_zrok
        tunnel = start_zrok(s.port, s.token)
    try:
        uvicorn.run(create_app(rt), host=s.host, port=s.port, log_level="warning")
    finally:
        if tunnel:
            tunnel.terminate()
        rt.close()


def cmd_static_cam(args):
    from forgemind.cameras.static_cam import run
    run(args.server, args.token, args.device, args.fps, args.show)


def cmd_tags(args):
    from forgemind.config import Settings
    from forgemind.perception.zones import make_tag_sheet
    s = Settings.from_env()
    out = make_tag_sheet(s.zones, args.out)
    print(f"wrote {out}: " + ", ".join(f"id {k} = {v}" for k, v in sorted(s.zones.items())))
    print("Print at 100%, stick each tag where the phone will see it in that zone.")


def cmd_check(args):
    from forgemind.agent.tools import schemas
    from forgemind.config import Settings
    from forgemind.providers import build_llm
    s = Settings.from_env()
    print(f"llm provider: {s.llm_provider}   segmenter: {s.segmenter}   embedder: {s.embedder}")
    if s.segmenter == "remote_sam":
        from PIL import Image

        from forgemind.providers.segmenters import RemoteSamSegmenter
        seg = RemoteSamSegmenter(s.sam_url, s.sam_token, s.sam_min_score, timeout=180)
        print(f"calling hosted SAM at {s.sam_url} (first call may cold-start)...")
        dets = seg.segment(Image.new("RGB", (640, 480), (120, 120, 120)), s.concepts)
        if seg.last_error:
            print(f"SAM endpoint FAILED: {seg.last_error}")
            sys.exit(1)
        print(f"SAM endpoint OK: {len(dets)} detections on a blank test frame, {seg.last_ms} ms")
    llm = build_llm(s)
    print(f"agent model: {llm.model_for('agent')}   worker model: {llm.model_for('worker')}")
    if s.llm_provider == "mock":
        print("mock mode - nothing to call. Set FM_LLM_PROVIDER=meta and MODEL_API_KEY in .env.")
        return
    try:
        models = llm.list_models()
        print(f"models visible to this key ({len(models)}): {', '.join(models[:20])}")
    except Exception as e:
        print(f"listing models failed: {e}")
    for role in ("agent", "worker"):
        try:
            res = llm.chat([{"role": "user", "content": "Where did I leave my multimeter?"}],
                           tools=schemas(), model=llm.model_for(role))
            calls = [(c.name, c.arguments) for c in res.tool_calls]
            print(f"[{role}] {llm.model_for(role)}: tool calls={calls} text={res.content!r:.120}")
            print(f"[{role}] usage: {res.usage}")
        except Exception as e:
            print(f"[{role}] {llm.model_for(role)} FAILED: {e}")
            sys.exit(1)
    print("OK: the model API answers and returns tool calls.")


def cmd_calibrate(args):
    from forgemind.calibrate import pick_embedder, run
    from forgemind.config import Settings
    s = Settings.from_env()
    if args.segmenter:
        s.segmenter = args.segmenter
    out = args.out or str(s.data_dir / "calibration.json")
    s.data_dir.mkdir(parents=True, exist_ok=True)
    rep = run(s, args.photos, embedders=args.embedders.split(","), target=args.precision,
              crop=not args.no_crop, out=out)
    best = pick_embedder(rep["results"])
    print(f"\nbest: {best['embedder']} -> FM_EMBEDDER={best['embedder']} "
          + " ".join(f"{k}={v}" for k, v in best["recommend"].items()))
    if best["embedder"] in ("dinov2", "dinov3"):
        print("note: DINO is image-only; text search (where_is by description) still needs clip/pe.")


def cmd_replay(args):
    from forgemind.config import Settings
    from forgemind.replay import run
    over = {"token": "off"}
    if args.concepts:
        over["concepts"] = [c.strip() for c in args.concepts.split(",") if c.strip()]
    if args.match is not None:
        over["match_min_sim"] = args.match
    s = Settings.from_env(**over)
    s.data_dir.mkdir(parents=True, exist_ok=True)
    run(s, args.video, camera=args.camera, fps=args.fps, out_dir=args.out)


def cmd_bench_region(args):
    import json
    import re
    from pathlib import Path

    from PIL import Image

    from forgemind.config import Settings
    from forgemind.perception import bench
    from forgemind.providers import build_segmenter
    s = Settings.from_env()
    if args.segmenter:
        s.segmenter = args.segmenter
    if args.concepts:
        s.concepts = [c.strip() for c in args.concepts.split(",") if c.strip()]
    src = Path(args.source)
    if src.suffix.lower() in (".mp4", ".mov", ".mkv", ".avi", ".webm"):
        from forgemind.replay import frames
        img = next((im for t, im in frames(src, 1.0) if t >= args.at), None)
        if img is None:
            raise SystemExit(f"no frame at {args.at}s in {src}")
    else:
        img = Image.open(src).convert("RGB")
    roi, masks, dropped = bench.suggest(img, build_segmenter(s, None), s.object_concepts,
                                        with_roi=args.roi)
    for concept, m in dropped:
        print(f"not masking the {concept} at {list(m)}: bench objects lie on it")
    roi_txt = ",".join(str(v) for v in roi) if roi else ""
    mask_txt = json.dumps([list(m) for m in masks]) if masks else ""
    out = Path(args.out or s.data_dir / "bench-region.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    bench.preview(img, roi, masks, out)
    print(f"FM_BENCH_ROI={roi_txt}")
    print(f"FM_BENCH_MASK={mask_txt}")
    print(f"preview (green = bench, red = ignored; right = what the pipeline sees): {out}")
    if args.write:
        env = Path(".env")
        text = env.read_text(encoding="utf-8") if env.exists() else ""
        for key, val in (("FM_BENCH_ROI", roi_txt), ("FM_BENCH_MASK", mask_txt)):
            line = f"{key}={val}"
            text, n = re.subn(rf"(?m)^{key}=.*$", line.replace("\\", "\\\\"), text)
            if not n:
                text += ("" if text.endswith("\n") or not text else "\n") + line + "\n"
        env.write_text(text, encoding="utf-8")
        print("written to .env")


def cmd_concepts(args):
    from forgemind.concepts import probe
    from forgemind.config import Settings
    from forgemind.providers import build_segmenter
    s = Settings.from_env()
    if args.segmenter:
        s.segmenter = args.segmenter
    phrases = [p.strip() for p in (args.phrases or ",".join(s.concepts)).split(",") if p.strip()]
    out = args.out or str(s.data_dir / "concepts")
    probe(build_segmenter(s, None), args.sources, phrases, args.min_score, out)
    print(f"annotated frames: {out}")


def cmd_agent_eval(args):
    import tempfile

    from forgemind.agent_eval import run
    from forgemind.config import Settings
    s = Settings.from_env(token="off", data_dir=tempfile.mkdtemp(prefix="fm-eval-"),
                          vector_backend="numpy", embedder="hash")
    if args.model:
        s.agent_model = args.model
    run(s, out=args.out, frames=args.frames)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="forgemind", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("serve", help="start the server")
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.add_argument("--zrok", action="store_true", help="public HTTPS link for the phone")
    p.add_argument("--mock", action="store_true", help="force the mock model")
    p.set_defaults(fn=cmd_serve)
    p = sub.add_parser("static-cam", help="stream a USB webcam as the bench camera")
    p.add_argument("--server", default="http://localhost:8770")
    p.add_argument("--token", default="", help="the t= value from the printed link")
    p.add_argument("--device", default="0", help="camera index or stream URL")
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--show", action="store_true", help="preview window (q quits)")
    p.set_defaults(fn=cmd_static_cam)
    p = sub.add_parser("tags", help="printable AprilTag sheet for the zones")
    p.add_argument("--out", default="zone-tags.png")
    p.set_defaults(fn=cmd_tags)
    p = sub.add_parser("calibrate", help="embedder crop test: pick embedder + thresholds")
    p.add_argument("photos", help="folder with one subfolder per object (see forgemind/calibrate.py)")
    p.add_argument("--embedders", default="clip,pe,dinov2")
    p.add_argument("--precision", type=float, default=0.90, help="target identification precision")
    p.add_argument("--segmenter", default="sam3", help="crops photos to the object (none = whole photo)")
    p.add_argument("--no-crop", action="store_true", help="photos are already crops")
    p.add_argument("--out", help="JSON report path (default: <data dir>/calibration.json)")
    p.set_defaults(fn=cmd_calibrate)
    p = sub.add_parser("replay", help="run a recorded video through perception + memory")
    p.add_argument("video")
    p.add_argument("--camera", choices=["static", "mobile"], default="static")
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--concepts", help="comma-separated, overrides FM_CONCEPTS")
    p.add_argument("--match", type=float, help="override FM_MATCH_MIN_SIM")
    p.add_argument("--out", help="output folder (default: <data dir>/replay/<video name>)")
    p.set_defaults(fn=cmd_replay)
    p = sub.add_parser("bench-region", help="suggest FM_BENCH_ROI / FM_BENCH_MASK from a frame")
    p.add_argument("source", help="sample image, or a video from the bench camera")
    p.add_argument("--at", type=float, default=0.0, help="seconds into the video")
    p.add_argument("--segmenter", help="override FM_SEGMENTER (default from .env)")
    p.add_argument("--out", help="preview image path")
    p.add_argument("--roi", action="store_true",
                   help="also crop to the desk (for a camera that sees the room or you)")
    p.add_argument("--concepts", help="bench objects to protect from masks (default FM_CONCEPTS)")
    p.add_argument("--write", action="store_true", help="write the values into .env")
    p.set_defaults(fn=cmd_bench_region)
    p = sub.add_parser("concepts", help="compare SAM phrasings on sample frames")
    p.add_argument("sources", nargs="+", help="images, or video.mp4@seconds")
    p.add_argument("--phrases", help="comma-separated (default: FM_CONCEPTS)")
    p.add_argument("--min-score", type=float, default=0.4)
    p.add_argument("--segmenter", help="override FM_SEGMENTER")
    p.add_argument("--out", help="folder for annotated frames")
    p.set_defaults(fn=cmd_concepts)
    p = sub.add_parser("agent-eval", help="scripted turns against the real model: tools, latency, cost")
    p.add_argument("--model", help="override FM_AGENT_MODEL for this run")
    p.add_argument("--frames", action="store_true", help="attach bench + phone frames like live use")
    p.add_argument("--out", help="JSON results path")
    p.set_defaults(fn=cmd_agent_eval)
    p = sub.add_parser("check", help="test the configured model API")
    p.set_defaults(fn=cmd_check)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
