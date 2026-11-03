# Submission draft

Draft text for the Meta Global AI Developer Hackathon form. Replace the `<…>` placeholders
and adjust field names to the official form (the rules page needs a login). Check the
official rules on prior work (this builds on the author's earlier gallery-memory app) and
on video length before submitting.

- **Project:** ForgeMind
- **Track:** Indie developer · **Theme:** Push the Frontier
- **Video:** https://youtu.be/WTOR9wyb4Ns · **Repo:** https://github.com/shashanka300/Forgemind (make public if required) · **Licence:** AGPL-3.0

## One-liner

A workshop assistant with eyes, ears and memory: a camera over your bench and the phone in your
hand feed a memory that lasts across days. You talk to it hands-free; it recognises the things you
teach it, remembers where they and your places are, keeps your experiment's steps and measurements,
and writes them up. Today *you* are the robot's body; the same requests can drive a robot later.

## Problem

People doing hands-on work lose time to things that go missing, to reconstructing where they left
off, to records nobody writes while working, and to hazards nobody watches when they step away.
Assistants only know what you type or the one photo you upload.

## What it does

- **Teach your things:** hold it up, say "these are my clippers"; recognised from then on by both
  cameras, including objects with no common name.
- **Where is it / what do you see:** answers from memory and from the live view, honestly saying
  when it can't see something.
- **Two cameras:** "switch to the desk camera" puts the mounted camera on your phone.
- **Remembers places and facts:** "my ML books are on the second shelf".
- **Work sessions:** steps, measurements and a cited report from what you say while working.
- **Corrections by voice** and a **safety watch** for an iron left on the bench.

## Models

| Model | Role |
|---|---|
| Muse Spark 1.3 (Model API) | The agent with tool calling, describing what's in view, reading displays, reports |
| Muse Voice Transcribe (Model API) | Hands-free speech to text |
| SAM 3.1 (open weights, local) | Finds objects by name on both cameras |
| Perception Encoder (open weights, local) | Object identity across cameras |
| Muse Glimmer 30B (open weights, local) | Generated training data for the fast path; simulated users |
| Also used (not Meta) | YOLOE (taught objects, AGPL-3.0), a fine-tuned Laya decision model (fast path), Kokoro (voice) |

## How it's built

Everything that looks at every frame runs on one local RTX 5090 (SAM 3.1, YOLOE, Perception
Encoder, ~230 ms per frame); Spark and Voice Transcribe run on the Meta Model API. Simple turns
take a local fast path (fine-tuned Laya + rules, under 0.2 s) whose writes Spark re-checks in the
background. FastAPI server, SQLite memory, a phone web page over a zrok tunnel or direct Wi-Fi,
a USB webcam over the bench. Validated tool calls, a source for every fact, guards against false
"done" claims and the assistant hearing itself, and a spend ledger with a hard cap.

## Results (docs/EVAL.md, docs/HANDOFF.md)

- Agent: 19/19 correct tools on scripted turns including real phrases from phone tests, median 1.8 s
- Fast path: 0.98 on 86 held-out phrases; teach / where / measure answered in 0.03–0.2 s
- Taught objects on replayed phone frames: 19/19 sampled labels correct, ~230 ms per frame
- Honest limit: live Spark turns take 3–10 s (model reasoning and API latency)
- API spend for building and testing: about $5

## What's next

A robot base or arm behind the same "look at / highlight / read this" requests; faster turns
(streaming answers, more of the conversation on the local fast path); sessions as training data
for robot policies.

## Team

`<name>`, solo.
