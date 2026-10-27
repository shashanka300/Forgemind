# Submission draft

Draft text for the Meta Global AI Developer Hackathon form. Replace the `<…>` placeholders
and adjust field names to the official form (the rules page needs a login). Check the
official rules on prior work (this builds on the author's earlier gallery-memory app) and
on video length before submitting.

- **Project:** ForgeMind
- **Track:** Indie developer · **Theme:** Push the Frontier
- **Video:** `<link>` · **Repo:** `<https://github.com/shashanka300/Forgemind>` (make public if required)

## One-liner

A workshop assistant with eyes, ears and memory: a bench camera and the phone in your hand
feed a persistent world memory, and an agent on Meta's Muse models finds your tools,
resumes yesterday's experiment, reads your multimeter and warns you when the iron is left
on. Today *you* are the robot's body; the same requests can drive a robot later.

## Problem

People doing hands-on work lose time to things that go missing, to reconstructing where
they left off, to records nobody writes while working, and to hazards nobody watches when
they step away. Assistants only know what you type or the one photo you upload.

## What it does

- **Where is it?** Finds objects across two cameras with zone, camera and time; highlights
  them live when the phone arrives; asks you to look when its memory is stale and remembers
  when something wasn't there.
- **Teach your things:** "this is my soldering iron" becomes a personal identity both
  cameras recognise.
- **Work continuity:** sessions with steps, settings and measurements; resumes yesterday's
  job with what's done and what's next.
- **Reads instruments:** a close-up of the multimeter, read by Muse Spark's vision and logged
  as an instrument reading; refuses OL, blank or ambiguous displays.
- **Safety watch:** iron on the bench and nobody there → phone alert, saying where you are.
- **Reports with provenance:** every fact cites its memory event; numbers not in the record
  are removed; missing values are flagged.
- **Hands-free voice** with an instant acknowledgement while the agent works.

## Meta models

| Model | Role |
|---|---|
| Muse Spark 1.3 (Model API) | The agent (tool calling, `reasoning_effort=low`), reading displays from photos, report summaries, extracting your goals and preferences |
| Muse Voice Transcribe (Model API) | Hands-free streaming speech-to-text (end-of-speech detection) |
| SAM 3.1 (open weights, local; API for on-demand checks) | Concept segmentation on both cameras: 14 concepts in ~170 ms per frame via batched grounding |
| Perception Encoder (open weights, local) | Object identity across cameras and text search over memory |
| Muse Glimmer 30B (open weights, local) | Simulated users in the evaluation; tested as a fallback agent |

## How it's built

Local RTX 5090 for everything that runs per frame (SAM 3.1, Perception Encoder); the Meta
Model API for reasoning, vision reading, reports and speech. FastAPI server, SQLite event
store + Qdrant Edge vectors, a phone web app over zrok HTTPS, a USB webcam for the bench.
Validated tool calls, provenance on every fact, tools that report missing/stale data
instead of trusting the model, a guard against false "done" claims, and a spend ledger with
a hard budget cap.

## Results (docs/EVAL.md)

- All six acceptance checks pass in automated tests; #2, #3 and #6 also against the real API
- Agent: 15/15 correct tools on scripted turns, median 2.2 s per answer, ~$0.004 per turn
- Simulated users (Glimmer as the person): 4/4 goals reached, 0 false claims
- Perception: ~270 ms per bench frame end to end (2.9x faster than the first version)
- Cross-camera identity (Assembly101, 5 cameras): 94% top-1, 89% recall at 90% precision
- Total API spend for building and evaluating: about $1.2

## What's next

A mobile robot base or arm behind the same `go_look_at` / `highlight` / `read_closeup`
requests; the egocentric sessions as training data for robot policies; a full-duplex voice
front end (MiniCPM-o 4.5 locally) once tool calling lands in duplex models.

## Team

`<name>`, solo.
