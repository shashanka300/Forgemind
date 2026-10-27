"""Calibration maths on synthetic photos (no models, no segmenter)."""
import numpy as np
from PIL import Image

from forgemind.calibrate import (auc, camera_of, concept_of, identification, load_photos,
                                 pair_threshold, run, threshold_for_precision)
from forgemind.providers.embedders import HashEmbedder


def test_names_and_auc():
    assert concept_of("hex_key-2.5mm") == "hex key" and concept_of("multimeter") == "multimeter"
    assert camera_of("bench_1.jpg") == "bench" and camera_of("Phone-2.png") == "phone"
    assert auc([0.9, 0.8], [0.1, 0.2]) == 1.0 and auc([0.5], [0.5]) == 0.5


def test_thresholds():
    res = [(0.95, True), (0.9, True), (0.85, False), (0.8, True), (0.7, False)]
    t = threshold_for_precision(res, 0.9)
    assert t["threshold"] == 0.9 and t["precision"] == 1.0 and t["recall"] == 0.4
    p = pair_threshold([0.9, 0.85, 0.6], [0.7, 0.5], 0.9)
    assert p == {"threshold": 0.85, "precision": 1.0, "recall": 0.667}


def test_identification_prefers_same_object():
    items = [{"object": "a", "camera": "phone"}, {"object": "b", "camera": "phone"},
             {"object": "a", "camera": "bench"}]
    vecs = [np.array([1.0, 0]), np.array([0, 1.0]), np.array([0.9, 0.1])]
    assert identification(vecs, items) == [(0.9, True)]


def test_run_end_to_end_on_colour_objects(tmp_path, settings):
    colours = {"block-red": (220, 30, 30), "block-dark_red": (190, 20, 25),
               "cup-blue": (30, 60, 220)}
    rng = np.random.default_rng(0)
    for name, rgb in colours.items():
        d = tmp_path / "photos" / name
        d.mkdir(parents=True)
        for cam in ("phone", "bench"):
            for k in range(2):
                jitter = np.clip(np.array(rgb) + rng.integers(-8, 8, 3), 0, 255).astype(np.uint8)
                Image.new("RGB", (64, 64), tuple(int(x) for x in jitter)).save(d / f"{cam}_{k}.png")
    assert len(load_photos(tmp_path / "photos")) == 12
    import forgemind.calibrate as cal
    cal.EMBEDDERS["hash"] = lambda s: HashEmbedder()
    rep = run(settings, tmp_path / "photos", embedders=["hash"], crop=False,
              out=tmp_path / "rep.json", log=lambda *a: None)
    r = rep["results"][0]
    assert r["pairs"] == {"same_object": 18, "different": 48, "look_alike": 16}
    assert r["identification"]["queries"] == 6 and r["auc"] is not None
    assert (tmp_path / "rep.json").exists()


def test_pick_embedder_prefers_text_capable_unless_clear_win():
    from forgemind.calibrate import pick_embedder

    def r(name, recall, look):
        return {"embedder": name, "identification": {"at_target": {"recall": recall}},
                "auc_look_alike": look}
    assert pick_embedder([r("pe", 0.889, 0.775), r("dinov3", 0.944, 0.587)])["embedder"] == "pe"
    assert pick_embedder([r("pe", 0.5, 0.7), r("dinov3", 0.9, 0.6)])["embedder"] == "dinov3"
    assert pick_embedder([r("clip", 0.6, 0.68), r("pe", 0.889, 0.775)])["embedder"] == "pe"
