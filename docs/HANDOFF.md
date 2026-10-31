# Handoff

Where the build stands and what to do next. Update this at the end of each work session.

**Last updated:** Oct 31, 2026, end of session. Next: live test of the "listening restored" build (never run live yet), demo script only. Fixed link https://shashanka-forgemind.shares.zrok.io/ , token in `.env`. Start with launch config `forgemind-live` + `static-cam --device 0` + `desk-mic`; all off while building. **Deadline:** Nov 5, 7:30 AM IST (Nov 4, 9 PM ET).
**Branch:** `main`. Check `git status -sb` for unpushed commits.

## Status

| Piece | State | Verified how |
|---|---|---|
| Full loop in mock mode (cameras → memory → agent → phone) | Works | 37 tests; phone page loads, chat round trip in browser |
| Object registry + teach identity (F-OBJ-1..5) | Existed in skeleton; teaching improved | Teach now takes up to 3 views from the last 4 s of phone frames, asks SAM for the named concept if the frame's detections missed it, and says when it fell back to the centre of the view. Tests only (colour-block fixtures) |
| Replay test (`forgemind replay`) | **Built + run** | Video → real pipeline (gate, SAM 3.1, PE, registry), mock model, no API. Fixed cam (overhead, 70 s): 16 entities / 19 false "disappeared" before the fixes → **9 entities, all real; 1 correct move (LCD); 1 correct disappearance (headset band picked up)**. Handheld walk-through as phone frames: 27 entities in a cluttered room, ~4 duplicates (a steel bottle split in 3, 1-sighting fragments) |
| Hand-aware perception | **Built** | New objects need 2 frames in a row (`FM_CONFIRM_NEW`); on the bench an object overlapping a hand/person box never creates an entity; a covered object isn't "missing"; "gone" needs 3 misses and 10 s; move linking: a same-concept object the bench saw go missing within 120 s matches at `match_min_sim - 0.15` |
| Voice, Option A (Day 5) | **Built** | Hands-free (`/ws/asr?mode=hands_free`, Voice Transcribe ENDPOINTING): verified live, each sentence becomes a turn ~0.4–1 s after speech ends. Phone streams silence while it talks (no self-transcription), auto-reconnects, tap Hold to interrupt. Instant spoken ack on the phone ("Let me check." / "Okay." / "One moment.") with the answer queued behind it. Not yet tried on a real phone. Duplex models investigated: NVIDIA VoiceChat needs an 80 GB GPU; MiniCPM-o 4.5 fits but has no tool calling (stretch goal) |
| Simulated users + rehearsal (Day 9) | **4/4 personas, rehearsal passes** | `forgemind sim-users` (Glimmer as the user, Spark as the agent, judged in code). Found and fixed: compound requests ("I measured 3.3 V. Please write up a report.") dropped the second request because tool summaries ended the turn (`is_compound` keeps the loop going; final reply = the tools' summaries); repeated readings within 60 s are one measurement; reports are written in the background (`FM_REPORT_ASYNC`). Full-system rehearsal with all real models over HTTP/WebSocket: teach, where_is, session, real close-up read, HTML report, $0.056 |
| Integration (Day 9) | **Built** | Reports render as phone HTML (`/reports/<name>`, `?raw=1` for Markdown), UTF-8 file I/O (cp1252 broke on Δ/→). Keyframes not referenced by memory are pruned every 10 min. Bench camera client keeps only the newest frame (reader thread), replays video files at real speed (`--loop`); verified streaming the fixed-cam recording into the live server. Docs: `docs/EVAL.md` scorecard, `docs/DEMO.md` script + setup checklist, README rewritten with an architecture diagram |
| Performance + budget (Day 9) | **Built** | SAM: all concepts in one batched grounding pass (~170 ms vs ~575 ms for 14 concepts, identical detections) + batched crop embeddings: bench frame ~780 → ~270 ms. Spend ledger (`providers/budget.py`): every Meta LLM call, SAM API image and Voice Transcribe second recorded; warnings at 1/3 and 2/3 of the cap (phone card), `BudgetExceeded` at `FM_BUDGET_USD` (fallback model takes over if configured); `forgemind spend`; tests use an isolated ledger |
| Reports + notes + inspector (Day 8) | **Acceptance #6 passes** | Citation check (F-OUT-5) in `reports.checked_narrative`: per sentence, cited [E#] must exist and every number must be in a cited fact (counts of steps/measurements/settings allowed); one rewrite request, then failing sentences are removed with a note. Real Spark summary: 0 problems, states the missing input voltage explicitly. User notes (F-EXP-9, `memory/notes.py`): background extraction on turns matching a signal regex (want/need/problem/keeps/always/prefer…), worker model JSON; shown in the situation summary and in resume ("You mentioned: …"); `FM_NOTES` toggle. Real Spark: goal/obstacle/preference notes correct, ~3.5 s off the reply path. Memory tab lists notes with Forget (`/api/notes`). Eval 15/15, median 2.1 s |
| Safety + active perception (Day 7) | **Acceptance #4, #5 pass in tests** | #4: iron on the bench, user leaves, phone in another zone: alert fires at the unattended threshold (0 s late in the simulated timeline; loop ticks every 5 s, target 10 s) and says "Your phone is in the <zone>." Unchanged (gate-skipped) bench frames keep their objects/presence seen. #5: `where_is` on a stale sighting sends a look request for its zone ("Could you point the phone at the bench so I can check it's still there?"); if the phone sees the zone but not the object for 3 frames → `not_found` event + "I don't see the X on the bench. It must have moved."; later where_is adds "But when you checked the bench …, it wasn't there." If it is there → seek highlights it and memory is fresh again. Guard: replies claiming an action with no tool call are retried once, then "Sorry, I didn't actually record that." |
| Muse Glimmer local (Option A item 4) | **Tested, not used for live turns** | `F:\models\glimmer\Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf` on llama.cpp b11934 CUDA 13.4 (`F:\llama.cpp\b11934\llama-server.exe -m <gguf> -c 32768 -ngl 99 -fa on --jinja --port 8081 --alias glimmer`), ~18 GB VRAM, ~80 tok/s, prompt cache ~0.2 s. Agent eval: **9/15 tools, median 4.5 s, max 16 s**; writes 300–500 reasoning tokens before tool calls, and **claims actions without calling tools** ("Started …", "Logged: swap R2 …"). Spark stays primary. As F-AGT-7 fallback it would need a guard that rejects action claims with no tool call; not wired in `.env` |
| Experiment memory (Day 6) | **Acceptance #2, #3 pass** | #2: a session from yesterday resumes with "last active 1 day ago … Next up: …" (unit test + real Spark eval turn). #3: `read_closeup` with real Spark vision logs `source=instrument` and keeps the raw display text; 7 test photos: correct reading (.0060 → 0.006 V), null for OL / blank / no display / stopwatch, and two visible meters are refused as ambiguous. F-AGT-8: `missing` from `experiments.gaps()` (values logged without a number; done "measure X" steps with nothing logged) in resume/compare/report summaries; `stale` + `confidence` in where_is. "Last active" ignores resume/end bookkeeping events |
| Agent on Muse Spark (Day 5) | **Under target** | `forgemind agent-eval`: 14 scripted turns on a fixed world, real Spark. Baseline median 8.2 s / max 12.9 s / $0.149. Now **median 2.2–2.5 s, max 3.6–6.3 s, 13–14/14 tools, $0.03–0.04 per 14 turns** (`--frames` adds ~700 tokens, no measurable delay). Changes: speak a tool's summary directly unless it needs phrasing (`NEEDS_PHRASING` = inspect_scene, compare_sessions) → ~1.07 calls/turn; static system prompt (situation moved to the user turn) so the prefix caches; `FM_REASONING_EFFORT=low`; `alert` hidden from the agent (low effort called it for small talk); a measurement completes its planned "measure …" step in code; step matching stems words (wired/wire, tinned/tin). Spoken answer ≈ 0.25 s transcript + ~2.5 s agent. Wonder: not used for live turns (slower, 3–5x cost); candidate for report review (Day 8) |
| `where_is` end to end (acceptance #1) | **Built, tested** | Answers zone + camera + time; "last seen" card shows a server-rendered snapshot (`/api/entities/{id}/snapshot`, keyframe zoomed on the object, box drawn); if not in view it **seeks** for 120 s: the first phone frame containing the object pushes a live highlight + "There's your X." (event `found`). Test `tests/test_where_is.py` runs the full bench → shelf → ask → arrive scenario. Verified in the browser pane with SAM 3.1 + PE on the workbench photo (cards, snapshot, highlight). Not yet on the real phone |
| Embedder calibration (`forgemind calibrate`) | **Decided: PE** | Real cross-camera test on Assembly101 (see below). `.env` now `FM_EMBEDDER=pe`, teach 0.835, match 0.90 (provisional). Text-search floor per embedder (PE 0.19, CLIP 0.22) |
| SAM 3.1 via Model API (`FM_SEGMENTER=meta_sam`) | **Works** | Live call on the workbench photo: reply format matches the parser, `c` scores and object ids come back, boxes match local SAM. ~2.3 s per concept (3 in parallel: 3.2 s) |
| Voice Transcribe relay (`/ws/asr`, phone AudioWorklet → server → Meta) | **Works server-side** | Live: synthetic 16 kHz speech streamed through the relay; exact transcript ("2.5 millimeter", "3.3 volts"), final 0.25 s after release, first partial ~1 s in, connect+handshake ~1.3 s (phone buffers meanwhile). `endStream` shape accepted. Not yet tried with a real phone mic over zrok. The file endpoint (`/asr/transcribe`) returned "Malformed request" to a quick multipart test; unused by the app |
| LLM fallback to Glimmer (`FM_FALLBACK_PROVIDER=compat`) | Built | Stub models only; no Glimmer server yet |
| Perception Encoder embedder (`FM_EMBEDDER=pe`) | Built | Runs on the 5090: 16 crops ~60 ms, 2.7 GB VRAM. Thresholds not recalibrated |
| Local SAM 3 (`FM_SEGMENTER=sam3`) | Runs on the 5090 | Two bench photos: ~490 ms for 12 concepts, 80 ms for one, 4.1 GB VRAM. SAM 3.1 detector weights give the same detections. Misses: soldering iron read as "screwdriver" in one photo; irons, stations and pliers missed in the other |
| Muse Spark / Wonder via Model API | **Works from India** | `forgemind check` OK: key sees 9 models (wonder, spark 1.1–1.3 + contributor, sam-3.1, voice-transcribe, image). Both return a correct `where_is` tool call; ~2.5 s Spark, ~4.5 s Wonder per call (~2.3k prompt tokens) |

## Blockers

1. ~~SAM access~~ Granted for both `facebook/sam3` and `facebook/sam3.1`.
2. ~~`MODEL_API_KEY`~~ In `F:\Forgemind\.env` (git-ignored; never print it). `.env` also sets
   `FM_LLM_PROVIDER=meta`, `FM_AGENT_MODEL=muse-spark-1.3`, `FM_SEGMENTER=sam3`.
3. **Bench lighting (user).** The bench camera is the user's 4K webcam (device 0, DirectShow,
   sent at 2560 px). It works live (Oct 28: pliers and headphones detected), but the room is
   too dark for SAM to see the boards. **Lit on Oct 29**: live run finds 5 boards, copper wire, pliers, headphones, mouse; the wire cutters need teaching. Fixed: a still scene never confirmed a new object (gate skipped the confirming frame). Memory reset on Oct 29: old dev memory moved to `forgemind-data-dev.bak-20261029-2001` (git-ignored; delete when not needed). **Model weights (SAM, PE, 3.1 GB) are cached in `<data dir>/models`**: never move or wipe that folder when resetting memory, only `forgemind.sqlite*`, `visual`, `keyframes`, `reports`. Dev server (preview `forgemind`) uses `forgemind-data-dev`, `--mock` LLM. After that, re-run
   `bench-region` (only the keyboard is masked now, `FM_BENCH_MASK=[[0.205,0.571,0.793,0.911]]`;
   don't mask the white laptop the boards sit on), then a live run: `forgemind serve` +
   `forgemind static-cam --device 0 --token <t>` + phone.

## First real phone test (Oct 31)

Run over zrok (`zrok2` 2.0.9 in `C:/Users/shash/bin`, launch config `forgemind-live`; now a fixed address, `https://shashanka-forgemind.shares.zrok.io/` (`FM_ZROK_NAME`, stale share released at start) and a fixed `FM_TOKEN` in `.env`, entered on the login page; the old per-start token
each start). Speech, transcription and teaching worked; what failed, and the fixes (all in):
- Small talk re-taught whatever was in view (39 teaches), so mouse crops were stored under
  "baby's biter" and the mouse then matched as the biter. Now: teach needs the name in that
  sentence (`name_said`), refuses instead of storing the middle of the view, and drops views
  that look like another taught object (`Registry._keep_own_views`, `TeachConflict`).
  Cost: big objects SAM can't box (cooler, cabinet) can no longer be taught.
- New tool `correct_object` ("that's not X, it's Y"): `Registry.unlearn` + teach the right name.
- Fast path was wrong on real speech at 0.98 confidence. resume_session is off it; a fast action
  also needs the request's words (`INTENT`) and no pushback (`PUSHBACK`). On the 73 real
  sentences (`eval/phone_test_oct31.txt`, `eval/real_speech.py`): 1 fast turn, correct. So the
  0.98 on the hand-written test does NOT describe real speech; the words gate does the work.
- Fragments ("um", "last") get no model call and no reply (`is_fragment`, event `ignored`).
- A just-taught object counts as seen; where-is no longer matches on any shared word
  ("baby's hand" isn't "baby's biter") and says when a match is only visual.
- Every turn logs its time and path (event `turn`). Voice playback itself isn't logged.
- agent-eval now has the phone test's remarks + a correction: 19 turns, median 0.93 s, $0.056.
Not fixed: a sentence split in two by the transcriber; remarks to oneself still get a spoken
reply; duplicates of untaught phone objects (about 50 objects with no zone tags up).
Memory backups (git-ignored, deletable): `forgemind-data.bak-phonetest-20261031` (holds 1,429
camera stills from the test, including the user's home), `forgemind-data.bak-evalworld-20261030`.
Bench preview: open `/bench` on the server (the OpenCV install is headless, so no windows; `--show` prints the address).

## Second phone test fixes (Oct 31)

Phone test 2: Spark median 4.1 s, max 19 s (queue included). Causes and fixes:
- It heard itself: phone mic now stays quiet 1.2 s after speech, `strip_echo` drops 4+ word runs of
  recent replies at the start/end of a turn (never mid-sentence: that once made "what's now?").
- Cut-off and repeated sentences: `is_cut_off` waits 1.5 s; a longer sentence replaces one waiting.
- Teach uses frames from when the turn arrived; "this is my X" is fast (`teach_name`).
- `FM_SEND_FRAMES=false`: frames no longer go to Spark on every turn (they were; only meter reads now).
- Event `turn` has queue_ms/model_ms/source; `voice` has TTS ms. Agent-eval 19/19, median ~1.1 s.
- Audio: mic opened with echoCancellation off (it put Android in call mode, moving the route) and
  only the device in use speaks pushed messages. UNVERIFIED on a phone; devices not identified.
- Bench: `/bench` shows the newest frame at 3 fps; detection capped ~1/s (`STATIC_MIN_EVERY_S`).
  Phone boxes glide and fade (still ~0.5 s old: tunnel 250 ms + SAM 175 ms).
- `forgemind desk-mic` (MX Brio, `cameras/desk_mic.py`): opens only while the bench camera saw the
  user within 120 s and the phone's Desk mic button is on; replies on PC speakers; same sentence
  from two mics = one turn. NOT yet run with a real mic.
**Rule from the user: all cameras off while building; start only for a test.**

## Third phone test fixes (Oct 31)

Both mics at once mixed the conversation (23 desk / 22 phone turns; each cut sentences differently, so
replies bounced between PC and phone). Now: one mic at a time (phone mic on -> desk mic closed;
`should_listen`, and the server refuses desk turns); replies on the device whose mic is open.
Wake word (`FM_WAKE_WORD=true`): hands-free only acts on "Forge, ..." plus follow-ups for 8 s after a
reply; hold-to-talk and typing don't need it. Agent can't promise setup changes (`promises_setup_change`
replaces the reply). `find` no longer merges "measurement tape" with "paper tape".

## Recognising taught objects, demo focus (Oct 31)

Focus agreed with the user: make the demo script reliable; casual chat stays basic.
SAM only finds what it's asked for, so unlisted taught objects never got a box. Now each frame also asks
for taught concepts and "small object"; these candidates never create objects and count only at PE
similarity >= 0.90 to a taught view (`_taught_candidates`). Teach picks: the said name/concept via SAM,
then "object" by the hand, then "small object" (`pick_held`). Known miss: objects SAM has no word for and
whose name doesn't help ("baby's biter"; "baby teether" works). Smoke alarm needs 3 frames in a row.
Wake word works anywhere in a sentence. Test before handing over: `eval/replay_teach.py` (saved frames, look at
the sheet) and `eval/demo_rehearsal.py` (13/13: Spark median 1.5 s, prompt ~3.2k tokens, frame ~265 ms).
Not covered by these: real audio, read_closeup, an alert firing live. Live Spark was 3.5 s; turn events now log
prompt tokens to explain it. Researched alternatives (YOLOE visual prompts, OWLv2, SAM 3 exemplars
across images): kept SAM + PE for now, YOLOE is the first thing to try if recall stays poor.

## Listening restored (Oct 31, test 3)
Open at end of session: held objects are labelled in few frames (min-sim 0.85 vs 0.90 barely differs on replay: clippers 7/63 both, timer 11 vs 9 of 52; kept 0.90; the frames don't show how long each object stayed in view); keyframe pruning deletes frames before they can be replayed; live Spark 3.2-3.5 s vs 1.5 s in rehearsal, cause unknown (prompt ~3.7k tokens is not it); Desk mic button sits in `.controls`, the user couldn't find it: unchecked on a phone. Memory cleared; backups `forgemind-data.bak-*` (git-ignored, deletable, hold camera stills of the user's home).

The wake word and filters dropped 20 of 33 real sentences: it felt deaf. `FM_WAKE_WORD=false` (code stays), only
pure filler is dropped, echo needs 5+ words at a sentence edge within 20 s, teach falls back to the middle of the
view and shows the box instead of refusing. Check filters with real sentences (`eval/phone_test_oct31.txt`), never
ones written with the feature in mind.

## Where to pick up (Oct 30, end of session)

Nothing left to build before the real phone test; everything below needs the user:
1. `zrok enable` (one-time), then `forgemind serve --zrok` and open the printed link on the phone.
2. On the phone: Start camera, Hands-free; check the Michael voice plays (first tap unlocks
   audio) and the mic goes quiet while it talks. If audio fails the phone falls back to its
   own voice: look at `/api/tts` responses and the browser console.
3. Teach the soldering iron, multimeter and wire cutters (SAM can't name them).
4. `forgemind static-cam --device 0 --token <t>` with the desk lamp on; start the bench
   watch; run the demo script (`docs/DEMO.md`) and time a few spoken where-is questions
   (expect ~0.6-1.0 s after the last word for fast-path answers).
5. Print zone tags (`forgemind tags`), record "yesterday's" session, film, submit
   (`docs/SUBMISSION.md`; check the official rules first).
Optional if time allows: `forgemind sim-users` with System 1 on; a third Laya data round if
the phone test shows new confusions (`eval/gen_confusions.py`, retrain with
`eval/system1_train.py`, compare with `eval/system1_eval.py` before switching `.env`).

Running state: nothing is running (dev server, Glimmer and Laya all stopped). Models on
disk: `F:/models/laya-forgemind-r2` (in use), `F:/models/laya-forgemind` (round 1),
Kokoro + spaCy model in the HF cache / venv. API spend: ~$1.33 of the $150 cap.

## Decision-model redesign (Oct 30)

Idea: a local "System 1" decision model picks the tool (and object/step/zone) in ~30 ms;
Muse Spark (System 2) only for open-ended, compound or uncertain turns. Target: single-tool
turns answered ~0.3-0.5 s after speech ends instead of ~2.5 s. Perception unchanged.

**Step 1 (Laya zero-shot) done: not usable as is.** `laya` 0.4.17 installed in `.venv`
(adds transformers 5.22; SAM/PE unaffected). English checkpoint on the 5090: 1.7 GB VRAM,
19-46 ms per call. Labelled dev set of 86 turns (14 tools + chat, incl. the agent-eval
script) + 5 compound turns in `eval/laya_zero_shot.py`:
- tool choice 37-43/86 whatever the phrasing (one 15-way choice, per-tool yes/no, or
  category-then-tool); worse, it is wrong at confidence 1.00 ("what step am I on" ->
  log_step), so a confidence cut-off can't make it safe
- compound detection fails (0.08-0.49 on compound turns, up to 0.96 on single ones)
- object choice works: 11/12 (allen key -> hex key, snips -> wire cutters; missed "DMM")
Matches the model card (zero-shot 0.36 vs 0.77 fine-tuned on their typed-decisions set).
Next, if the user agrees: fine-tune on our labels (training data generated locally with
Glimmer, the 86 hand-written turns kept as the held-out test), against a cheap baseline
(classifier on text embeddings) trained on the same data.

**Step 1b (fine-tuned Laya) done.** Data: 2,258 phrasings generated locally with Glimmer
(`eval/gen_decisions.py`, 17 labels x 5 speaking styles + compound; some label noise, e.g.
questions about a value labelled log_measurement); 10% kept as validation
(`eval/decisions_valid.jsonl`). `eval/system1_train.py` -> `F:/models/laya-forgemind`
(4 epochs, RLCD, option shuffling, ~25 min on the 5090; checkpoint not in git).
`eval/system1_eval.py` scores the hand-written test (never trained on):
- tool top-1 **0.93** (zero-shot 0.43), 16 ms; PE-text softmax baseline 0.86
- Laya's `multi` misses some compound turns (0.01 on "I measured 3.3 V. Please write up a
  report."); the existing `is_compound` regex catches all 5 -> always apply it
- fast path = tool != chat, conf >= T, multi < 0.5, not is_compound:
  T 0.95: 65/80 covered, 2 wrong; T 0.96: 50/80, 1 wrong ("R1 is 4.7k" -> log_measurement);
  read-only tools only at 0.96: 14/80, 0 wrong
- calibrated confidences top out ~0.97, so T must be fine-grained
Not yet done: argument filling (object choice works zero-shot 11/12; numbers by regex),
runtime integration, targeted data for the confusions (highlight vs go_look_at, "R1 is X").

**Step 2 (fast path in the runtime) done, on in `.env` (`FM_SYSTEM1`).**
`forgemind/agent/system1.py`: Laya picks the tool; fast only for where_is, highlight,
resume_session, read_closeup, end_session, watch_bench, log_measurement, at conf >= 0.96,
two-actions < 0.5, and never when `is_compound`. Arguments: object = Laya choice over objects
in memory (+ "something else" -> Spark); session = word match on names, else most recent;
measurement = exactly one number with a unit (questions excluded). The orchestrator speaks
the tool's summary; for writes (measurement, end, watch) Spark re-decides in a background
thread: same call -> nothing; same + more (a missed request) -> runs the extra calls and
says "Also: ..."; different -> `undo` + Spark's calls + "Correction: ...". Laya loads in the
background at server start; turns during loading go to Spark. agent-eval: 15/15, 5 fast
turns at 31-53 ms, median 1.27 s (from 2.2 s), $0.055; both background checks agreed.
Tests: `tests/test_system1.py` (stub Laya).
**Round 2 (in use, `F:/models/laya-forgemind-r2`):** +650 Glimmer phrasings on the
confused boundaries (`eval/gen_confusions.py`; prompts checked to contain no test phrasings)
-> hand-written test top-1 0.98, 0 wrong fast actions at any cut-off; agent-eval 15/15 with
7 fast turns, median 1.35 s, max 2.6 s, $0.037. end_session is fast only when the session is
named (a goodbye was fast-ended and then undone).
**Voice path (Oct 30):** hands-free phones open `/ws/asr?mode=hands_free&server_turns=1`;
the relay (`TurnHooks`) prefetches a System 1 decision on every partial transcript and
answers each turn on the server, pushing `turn` (show + spoken ack) then `reply` to the
phone instead of the phone posting to /api/chat. Measured with `eval/voice_latency.py`
(real Voice Transcribe, synthetic speech): last word -> fast answer ~0.6-1.0 s, of which
~0.4-1.0 s is the transcriber's end-of-turn detection. An "early answer" before end of turn
was built and removed: the last word's partial only arrives ~70 ms before end of turn.
**Natural voice (Oct 30):** the phone's built-in speech sounded robotic and the Meta API
has no TTS, so replies are voiced locally by Kokoro-82M (`forgemind/server/tts.py`,
`POST /api/tts` -> MP3; `FM_TTS=kokoro`, `FM_TTS_VOICE=am_michael`, the user's pick of 4
samples). ~100-130 ms per reply, 10-70 KB; loads in ~6 s at server start in the background.
The phone (`speak()`) queues ack + answer, caches the three acks, unlocks audio on the first
tap, falls back to the phone voice on any error, and treats the new voice as "speaking" so
hands-free still sends silence and a tap still interrupts. Verified in the browser pane:
ack + answer played through Kokoro, no fallback.
Not done: real phone test (zrok). Round 1 checkpoint kept at
`F:/models/laya-forgemind` for comparison.

## Next steps (in order)

1. **SAM follow-ups:** the 3.1 checkpoint is now the default (`FM_SAM_CHECKPOINT=sam3.1`;
   the "missing convs.3 keys" message at load is harmless, see `Sam3Segmenter`). Tune concept names and the 0.5 cutoff on photos of
   the user's own bench ("soldering iron" vs "soldering station"). Benchmark script:
   `sam_bench.py` (scratchpad; move into the repo if kept). `Sam3Segmenter` now runs
   under bf16 autocast (required).
2. **API follow-up:** try push-to-talk from the real phone over zrok (mic permission,
   AudioWorklet at 16 kHz on the phone's browser). Agent latency is now under target
   (see Status); re-run `forgemind agent-eval` after prompt or tool changes (~$0.04).
3. **Embedder decision (Day 3): done on public data.** Assembly101 sample, one recording
   (`nusar-2021_action_both_9011-a01…`, CC BY-NC 4.0, internal testing only; downloaded to
   the session scratchpad, not the repo). At t=139.5 s, 9 parts hand-matched across 5 fixed
   cameras (C10115 + C10404 overhead = "bench", C10095/C10119/C10390 = "phone"); 45 crops,
   18 bench queries. Includes a true look-alike pair (two identical tracks).
   | embedder | top-1 | recall @ P≥0.9 (thr) | look-alike AUC | text search |
   |---|---|---|---|---|
   | clip | 0.83 | 0.61 (0.902) | 0.68 | yes |
   | **pe** | **0.94** | **0.89 (0.835)** | **0.78** | yes |
   | dinov2 | 0.83 | 0.06 (0.841) | 0.60 | no |
   | dinov3 | 0.94 | 0.94 (0.817) | 0.59 | no |
   Synthetic smoke test (same pixels) had favoured DINOv3; real cross-camera views do
   not. CLIP at the old 0.80 would accept almost anything (different objects average
   0.80). No embedder separates untaught look-alikes by appearance (PE needs 0.917 for
   12% recall): identity of look-alikes must come from place/recency or teaching.
   `pick_embedder` now prefers text-capable models unless image-only wins by >0.10 recall.
   **Replay done** (see Status): 0.90 holds stable objects together; lowering it (0.85 →
   14, 0.80 → 13 entities) barely helps, the fragments came from hands, now handled in code.
   Missed detections are vocabulary (SAM never matched "usb charger" to the white cube
   charger; blue pliers missed): tune `FM_CONCEPTS` to the demo objects. User footage so far:
   `sampledata.mp4` (handheld walk-through, narrated: names via Voice Transcribe) and
   `fixed cam.mp4` (overhead). Correction: the RGB keyboard never tripped the change gate on
   its own (0/69 frame pairs at 1 fps); the gate fired because hands moved almost all clip. Both git-excluded. Folder layout for
   `forgemind calibrate`:
   - 8–10 of the demo objects, **including look-alikes** (two screwdrivers, two hex keys).
   - One folder per object, named `<concept>-<which>` (e.g. `hex_key-2.5mm`,
     `screwdriver-red`); the part before `-` is what SAM is asked for.
   - In each: 3 phone photos (held close, different angles, like teaching) named
     `phone_1.jpg`… and 2–3 from the bench-camera position (above the bench, at its
     distance) named `bench_1.jpg`….
   - Run: `forgemind calibrate <folder> --embedders clip,pe,dinov2,dinov3`. It crops with
     SAM, prints per-embedder AUC, top-1 and the threshold for 90% precision, and writes
     `calibration.json`. Then set `FM_EMBEDDER` and both thresholds from its output.
   - **Smoke result** (synthetic: one workbench photo, 13 objects incl. 6 red pliers and 5
     screwdrivers, "bench" views = downscaled/rotated/relit crops; optimistic):
     | embedder | AUC | look-alike AUC | top-1 | recall @ P≥0.9 (thr) |
     |---|---|---|---|---|
     | clip (current) | 0.744 | 0.606 | 0.31 | 0.08 (0.898) |
     | pe | 0.863 | 0.749 | 0.58 | 0.19 (0.871) |
     | dinov2 | 0.904 | 0.746 | 0.65 | 0.27 (0.734) |
     | dinov3 | 0.897 | 0.767 | 0.77 | 0.54 (0.764) |
     Takeaways: CLIP B/32 at 0.80 would mis-identify often; DINO leads on identity but is
     image-only, so if it wins on real photos the registry needs two vectors per crop
     (DINO for identity, PE/CLIP for text search). Near-identical objects (the red
     pliers) can't be told apart from one crop by any of them: identity must lean on
     place and recency for those, and the demo should use visually distinct objects.
4. **Ego-Exo4D:** user requested access on Oct 27; credentials expected Oct 28. Then pull a
   few bike-repair takes (downscaled) for a larger cross-camera test with real tools.
5. **Glimmer server:** download the Q4 GGUF, serve with llama.cpp (`-c 32768`), point
   `FM_COMPAT_*` at it, confirm tool calls with `forgemind check`.
6. **Day 3 remaining:** bench region is built (`FM_BENCH_ROI`/`FM_BENCH_MASK`, suggested by
   `forgemind bench-region <video> [--roi] [--write]`; masks never cover detected bench
   objects; ROI crop is opt-in). Set it once the bench camera's final mount is fixed.
   **Concepts tuned** with `forgemind concepts` (6 desk frames + 8 web photos):
   - SAM 3.1 **never** answers to "soldering iron" (0 hits on 6 clear iron photos); irons
     come out as "soldering pen" (0.59–0.75; also fires on ballpoint pens, 0.81) or
     "screwdriver" (0.78–0.80). So the iron must be **taught as "soldering iron"**; the
     safety watch now matches taught names, not just detector concepts (tested).
   - "multimeter" fires on the user's digital stopwatch (0.94); no timer phrasing fires
     ("stopwatch", "kitchen timer", "timer", "digital clock", "egg timer"): teach the meter.
   - "soldering station" and "laptop" box whole scenes: dropped. Object boxes > 60% of the
     frame are now discarded; one label per object (cross-concept IoU ≥ 0.7 keeps the higher).
   - "lcd display"/"lcd screen" hit monitors and meter screens, not LCD modules; "circuit
     board" finds the modules (0.94–0.97). "wall charger" finds the cube charger (0.57),
     "copper coil" the coil (0.63); on the lit webcam bench "copper wire" works better (0.83, only the coil), now in `FM_CONCEPTS`. "sponge" (iron-stand sponge) is reliable (0.66–0.82).
   - Each concept costs ~37 ms/frame (14 concepts ≈ 780 ms on the 5090).
   Remaining: try teaching on the real phone over zrok. Then Day 4: `where_is` end to end.
   Note: "covered isn't missing" can delay a real "gone" while hands stay nearby; revisit if
   it hurts the safety watch.
7. **Added from the personal health agent paper** (PRD §7.4, Nature 2026), scheduled in §13:
   - F-AGT-8 explicit `missing` / `stale` fields from tools (Day 6, P0)
   - F-OUT-5 deterministic citation check on reports (Day 8, P0)
   - F-EXP-9 user notes memory: goals, obstacles, preferences per turn (Day 8, P1)
   - §15.1 evaluation harness with Glimmer as simulated user (Day 9)
   Do **not** add a multi-agent orchestrator to live turns: the paper's averaged 205 s/turn.

## Environment

- **Repo:** `F:\Forgemind` (Windows native, not WSL). Remote `shashanka300/Forgemind`.
- **venv:** `.venv`, Python 3.12 via uv. Installed: `.[vision,dev]`, torch 2.11 cu128,
  plus the `local` extra's packages (sam3 pinned, einops, pycocotools, psutil,
  triton-windows, open_clip_torch). Fresh setup:
  ```bash
  uv venv --python 3.12
  uv pip install -e ".[vision,dev]"
  uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
  uv pip install -e ".[local]"
  ```
- **GPU:** RTX 5090 32 GB (compute 12.0), 64 GB RAM.
- **Dev server for the browser pane:** entry `forgemind` in `E:\challeenge\.claude\launch.json`
  (port 8770, `FM_TOKEN=devtoken`, data in `forgemind-data-dev/`). Open
  `http://localhost:8770/?t=devtoken`. First start downloads CLIP (~600 MB).
- **Ignored locally** (`.git/info/exclude`): `.claude/`, `forgemind-data-dev/`.
- **Commits:** no AI attribution lines; author `shashanka300 <shashanka300@gmail.com>`.

## Budget

**Hard cap: USD 150 of Meta Model API spend for the whole engagement** (user's limit).
Spent so far: tracked by `forgemind spend` (ledger `~/.forgemind/ledger.sqlite`, opening
balance $0.90 for Oct 26-28). Hard stop `FM_BUDGET_USD=140`. Before any API-heavy run, estimate the cost first.
Prices: Spark $1.25/$4.25 per 1M tokens in/out (~$0.003 per agent call at ~2.3k prompt
tokens), Wonder $4/$20 (~3x Spark), SAM API $2.50 per 1k images (one per concept),
Voice Transcribe $0.18 per audio hour. Bulk work (evaluation, simulated users,
calibration) runs on local models: SAM 3.1, Glimmer, embedders.

## Gotchas

- New objects need a second sighting within 5 s (`CONFIRM_WITHIN_S`); posting test frames
  by hand more slowly than that never confirms anything. The change gate also skips
  identical frames: vary the frame or wait 15 s.
- Phone frames with no AprilTag in view have zone None; the registry keeps the object's
  last known zone then (it used to "move" everything to nowhere).
- Keyframes: the bench camera saves one per processed frame (up to 1/s) and nothing deletes
  them yet. Fine for the demo; add a janitor (keep only referenced ones) before long runs.
- Not built from F-UI-3: an arrow toward an object that is in the zone but off-screen.

- `.env` holds the real `MODEL_API_KEY`; `Settings.from_env` loads it, so test fixtures
  must override `meta_api_key=""` (conftest does). A test once reached the real API.
- Voice Transcribe rejects audio slower than real time ("Ingress audio slower than
  real-time"); keep ~1–2 s of lead when replaying files. The relay should recover from it.
- SAM reads a digital stopwatch as "multimeter" (0.94): add decoy concepts or rely on
  taught identity for the multimeter.

- `sam3` on Windows needs `triton-windows`, `einops`, `pycocotools`, `psutil` (none are
  declared by the package).
- HF cache warns about symlinks on Windows; harmless (Developer Mode would silence it).
- Paths inside JSON launch configs: use forward slashes (`\f` became a form feed once).
- The SAM 3.1 API bills one image per concept per frame: never point `meta_sam` at the
  continuous camera stream (~$54/h at 12 concepts, one frame every 2 s).
- Contributor mode on Spark trains Meta's models on our bench frames: keep it off.

- Patching Python files from bash heredocs turns `\b` into a backspace character (it broke
  regexes in `agent/system1.py` twice). Patch with the Edit tool or a script file, then check
  for `\x08` in the file.
- System 1 needs `USE_TF=0` (transformers otherwise probes TensorFlow); `.env` sets it.
- Test settings pin `system1=""`; with `FM_SYSTEM1` in `.env` the tests would load Laya.
- Laya's calibrated confidences top out ~0.97-0.98: keep the cut-off fine-grained (0.96).
- `agent-eval` worlds live in `%TEMP%/fm-eval-*` (events `system1`, `system1_check` show
  fast decisions and Spark's background verdicts).
- The voice: `Speaker.ready` only after the warm-up sentence; the phone re-reads
  `providers.tts` on each refresh because the voice loads after the page.

## Key docs

- [`PRD.md`](PRD.md): scope, milestones (§13), acceptance criteria (§14), model split (§7.3)
- [`META_INTEGRATION.md`](META_INTEGRATION.md): per-model setup, prices, status table
