"""Fine-tune Laya for ForgeMind's System 1 decisions, and train a cheap baseline.

    python eval/system1_train.py --data eval/decisions_train.jsonl --out F:/models/laya-forgemind

Questions per turn (the same ones the runtime will ask):
  tool  - choice over the 14 agent tools + "chat" (answer without a tool)
  multi - yes/no: two or more different actions in one message
Compound rows only label `multi`; single rows label both. 10% of the generated rows are kept
aside (`--valid`) for choosing the confidence cut-off; the hand-written turns stay the test.
"""
import argparse
import json
import os
import random
import sys

os.environ.setdefault("USE_TF", "0")
sys.path.insert(0, os.path.dirname(__file__))
from laya_zero_shot_data import SHORT

TOOL_Q = {"type": "choice", "instructions": "Which action should the workshop assistant take for what the user just said?",
          "criteria": SHORT}
MULTI_Q = {"type": "noul", "instructions": "Does the user ask for two or more different actions in this message?"}
QUESTIONS = {"tool": TOOL_Q, "multi": MULTI_Q}


def laya_rows(rows):
    out = []
    for r in rows:
        if r["compound"]:
            out.append({"state": r["text"], "questions": {"multi": MULTI_Q}, "expected": {"multi": True}})
        else:
            out.append({"state": r["text"], "questions": QUESTIONS,
                        "expected": {"tool": r["label"], "multi": False}})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="eval/decisions_train.jsonl")
    ap.add_argument("--out", default="F:/models/laya-forgemind")
    ap.add_argument("--valid", default="eval/decisions_valid.jsonl")
    ap.add_argument("--epochs", type=int, default=4)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    random.Random(1).shuffle(rows)
    n_valid = len(rows) // 10
    valid, train = rows[:n_valid], rows[n_valid:]
    with open(a.valid, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in valid)
    tmp = a.out.rstrip("/") + "-train.jsonl"
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in laya_rows(train))
    print(f"train {len(train)}  valid {len(valid)}")

    from laya.train import TrainConfig, finetune, resolve_checkpoint_dir
    base = resolve_checkpoint_dir("convaiinnovations/laya")
    cfg = TrainConfig(epochs=a.epochs, shuffle_options=("choice",), log_every=20)
    summary = finetune(tmp, base, a.out, cfg, device="cuda")
    print(json.dumps(summary, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
