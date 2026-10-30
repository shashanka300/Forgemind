"""Score System 1 deciders on the hand-written test turns.

    python eval/system1_eval.py --laya F:/models/laya-forgemind [--base] [--baseline]

Fast-path rule: act without Spark only if the tool isn't "chat", tool confidence >= T and
the two-actions score < M. T and M are chosen on the generated validation split (fewest
fast turns that are wrong, then most covered), then applied unchanged to the test turns.
A fast action is wrong if the tool is not acceptable for the turn, or the turn is compound
(the second request would be dropped) or chat.
"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault("USE_TF", "0")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from laya_zero_shot_data import DATA, AMBIG, COMPOUND, SHORT
from system1_train import QUESTIONS

TOOLS = list(SHORT)


def test_set():
    rows = [(u, {lab}, False) for lab, us in DATA.items() for u in us]
    rows += [(u, ok, False) for u, ok in AMBIG]
    rows += [(u, set(), True) for u in COMPOUND]
    return rows


def valid_set(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    return [(r["text"], set() if r["compound"] else {r["label"]}, r["compound"]) for r in rows]


class LayaDecider:
    def __init__(self, path):
        import laya
        self.agent = laya.load(path, device="cuda")
        self.name = "laya " + os.path.basename(path.rstrip("/"))

    def decide(self, text):
        a = self.agent.predict(text, QUESTIONS)["answers"]
        return a["tool"]["choice"], a["tool"]["answer_confidence"], a["multi"]["noul"]


class EmbedBaseline:
    """Softmax regression on PE text embeddings (tool), logistic (multi)."""

    def __init__(self, train_path):
        import torch
        from forgemind.providers.embedders import PerceptionEncoderEmbedder
        self.torch, self.emb = torch, PerceptionEncoderEmbedder()
        rows = [json.loads(l) for l in open(train_path, encoding="utf-8")]
        X = self._x([r["text"] for r in rows])
        y_multi = torch.tensor([float(r["compound"]) for r in rows])
        single = [i for i, r in enumerate(rows) if not r["compound"]]
        y_tool = torch.tensor([TOOLS.index(rows[i]["label"]) for i in single])
        self.W_tool = self._fit(X[single], y_tool, len(TOOLS))
        self.W_multi = self._fit(X, y_multi.long(), 2)
        self.name = "PE-text softmax baseline"

    def _x(self, texts):
        t = self.torch
        v = t.tensor([self.emb.embed_text(s) for s in texts], dtype=t.float32)
        return t.nn.functional.normalize(v, dim=1)

    def _fit(self, X, y, k, steps=400):
        t = self.torch
        lin = t.nn.Linear(X.shape[1], k)
        opt = t.optim.Adam(lin.parameters(), lr=0.05, weight_decay=1e-4)
        for _ in range(steps):
            opt.zero_grad()
            loss = t.nn.functional.cross_entropy(lin(X * 10), y)
            loss.backward()
            opt.step()
        return lin

    def decide(self, text):
        t = self.torch
        with t.no_grad():
            x = self._x([text]) * 10
            p = t.softmax(self.W_tool(x), 1)[0]
            m = t.softmax(self.W_multi(x), 1)[0, 1].item()
        i = int(p.argmax())
        return TOOLS[i], float(p[i]), m


def outcomes(decider, rows):
    out, ms = [], []
    for text, ok, compound in rows:
        t0 = time.perf_counter()
        tool, conf, multi = decider.decide(text)
        ms.append((time.perf_counter() - t0) * 1000)
        out.append(dict(text=text, ok=ok, compound=compound, tool=tool, conf=conf, multi=multi))
    return out, sorted(ms)[len(ms) // 2]


def fast(o, T, M):
    return o["tool"] != "chat" and o["conf"] >= T and o["multi"] < M


def wrong(o):
    return o["compound"] or o["tool"] not in o["ok"] or "chat" in o["ok"] and len(o["ok"]) == 1


def score(outs, T, M):
    acted = [o for o in outs if fast(o, T, M)]
    bad = [o for o in acted if wrong(o)]
    eligible = [o for o in outs if not o["compound"] and o["ok"] - {"chat"}]
    covered = [o for o in acted if not wrong(o)]
    return len(bad), len(covered), len(eligible), bad


def pick(outs):
    best = None
    for T in (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98, 0.99):
        for M in (0.1, 0.2, 0.3, 0.5):
            nbad, ncov, _, _ = score(outs, T, M)
            key = (nbad, -ncov)
            if best is None or key < best[0]:
                best = (key, T, M)
    return best[1], best[2]


def report(decider, valid_path):
    v, _ = outcomes(decider, valid_set(valid_path))
    T, M = pick(v)
    vb, vc, ve, _ = score(v, T, M)
    t, ms = outcomes(decider, test_set())
    top1 = sum(o["tool"] in o["ok"] for o in t if not o["compound"]) / sum(not o["compound"] for o in t)
    tb, tc, te, bad = score(t, T, M)
    multi_hit = sum(o["multi"] >= M for o in t if o["compound"])
    print(f"\n== {decider.name}  ({ms:.0f} ms median)")
    print(f"  cut-off from validation: tool conf >= {T}, two-actions < {M}  "
          f"(validation: {vc}/{ve} fast, {vb} wrong)")
    print(f"  TEST: top-1 tool {top1:.2f}; fast path covers {tc}/{te} tool turns, {tb} wrong fast actions; "
          f"compound caught {multi_hit}/{len(COMPOUND)}")
    for o in bad:
        print(f"    WRONG {o['conf']:.2f} m={o['multi']:.2f} {o['tool']:16s} | {o['text']}")
    missed = [o for o in t if not o["compound"] and o["tool"] not in o["ok"]]
    for o in missed[:25]:
        print(f"    miss  {o['conf']:.2f} {o['tool']:16s} want {'/'.join(sorted(o['ok']))} | {o['text']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--laya", help="fine-tuned checkpoint dir")
    ap.add_argument("--base", action="store_true", help="also score the base checkpoint")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--train", default="eval/decisions_train.jsonl")
    ap.add_argument("--valid", default="eval/decisions_valid.jsonl")
    a = ap.parse_args()
    if a.base:
        report(LayaDecider("convaiinnovations/laya"), a.valid)
    if a.laya:
        report(LayaDecider(a.laya), a.valid)
    if a.baseline:
        # trained on the same rows Laya trained on (valid rows excluded)
        valid = {json.loads(l)["text"] for l in open(a.valid, encoding="utf-8")}
        tmp = a.train.replace(".jsonl", ".minus-valid.jsonl")
        with open(tmp, "w", encoding="utf-8") as f:
            f.writelines(l for l in open(a.train, encoding="utf-8") if json.loads(l)["text"] not in valid)
        report(EmbedBaseline(tmp), a.valid)


if __name__ == "__main__":
    main()
