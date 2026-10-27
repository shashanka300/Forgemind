# Demo: script, setup and shot list

Target length: about 3 minutes (check the official submission rules for the limit).
Every scene maps to one acceptance check (PRD §14). Scenes 2 and 6 need a session recorded
the day before the shoot.

## Setup checklist (the day before)

- [ ] **Bench camera** mounted above the bench, looking down; the whole working area in
      frame, your face out of it. White desk light on (the webcam drops to ~17 fps in dim
      light and moving objects blur).
- [ ] `forgemind bench-region "<a short clip from the bench camera>" --at 5 --write`, then
      check the preview (keyboard/laptop masked, nothing that holds tools masked).
- [ ] **Zone tags:** `forgemind tags`, print at 100 %, one on the bench, one on the shelf,
      one in the kitchen (or wherever you'll walk to).
- [ ] **zrok:** `zrok enable <token>` once; `forgemind serve --zrok` prints the phone link.
- [ ] **Demo objects** on the bench: multimeter (with a readable display), soldering iron
      **unplugged** (the alert is about you being away, not heat), a 2.5 mm hex key, a
      breadboard with a voltage divider, one look-alike pair if you can.
- [ ] **Teach** (phone, *Teach object* or say it): "this is my soldering iron", "this is my
      multimeter", "this is my 2.5 mm hex key". Turn each object a little and say it again.
- [ ] **Yesterday's session** (for scenes 2 and 6): "start a session called voltage divider
      test", "the steps are: wire the divider, measure the output, swap R2 for 22k",
      "I wired the divider", "set R1 to 10k", "set R2 to 20k", then "read the multimeter"
      (close-up of a real reading), then "end the session".
- [ ] Check memory on the phone (*Memory* tab): objects, notes. Forget anything wrong.
- [ ] `forgemind spend` to note the budget before the shoot.

## Script

| # | Scene (acceptance check) | You do / say | ForgeMind does | Shot |
|---|---|---|---|---|
| 0 | Intro (10 s) | "This is ForgeMind: a bench camera, my phone, and a memory." | — | Wide shot of the bench + phone in hand |
| 1 | **Resume** (#2) | Walk to the bench, tap *Hands-free*. "Let's continue the voltage divider test from yesterday." | "Resuming 'voltage divider test', last active 1 day ago. Done: wire the divider … Next up: measure the output." | Phone screen + face/voice |
| 2 | **Read the meter** (#3) | Probe the divider. "Read the multimeter." Hold the phone close. | "Hold the phone close to the multimeter display" → "The multimeter display reads 3.31 V. Logged as output voltage, read from the display." Session tab shows the source | Over-the-shoulder on the phone close-up, then the Session tab |
| 3 | **Where is it?** (#1) | Earlier, carry the hex key to the shelf. At the bench: "Where's my 2.5 mm hex key?" | Card with the snapshot: "last seen on the shelf by your phone, … When you get to the shelf, point the phone around and I'll highlight it." Walk to the shelf: green highlight + "There's your 2.5 mm hex key." | Follow-cam walking to the shelf; phone screen |
| 4 | **Stale memory** (#5) | Ask for something not seen for a while: "Where's the multimeter?" (move it away beforehand, off camera) | "…a while ago. Could you point the phone at the bench so I can check it's still there?" → point at the bench → "I don't see the multimeter on the bench. It must have moved." | Phone screen |
| 5 | **Safety** (#4) | Leave the (unplugged) iron on the bench, walk to the kitchen with the phone, wait about a minute | Phone vibrates + speaks: "The soldering iron is on the bench and nobody has been there for 1 min. Your phone is in the kitchen." | Bench camera view + you in the kitchen getting the alert (split screen) |
| 6 | **Report** (#6) | "Write it up." Open the report on the phone. | Report with configuration, method, results, every fact tagged [E#]; the missing input voltage flagged | Scroll the report on the phone |
| 7 | Close (15 s) | Overlay the scorecard numbers | — | `docs/EVAL.md` numbers as captions: 15/15 tools, ~2.2 s answers, ~270 ms per frame, 89 % identity recall at 90 % precision, ~$1 API spend |

## Fallbacks during the shoot

- Model API slow or down → `FM_FALLBACK_PROVIDER=compat` with Glimmer running locally
  (see HANDOFF), or re-take the scene.
- Speech misheard → type the line into the phone's text box (same path).
- The iron isn't recognised → it wasn't taught on this camera angle: teach it again from
  the bench camera's view (hold it under the camera, "this is my soldering iron").
- A recorded bench clip can stand in for the live camera during rehearsals:
  `forgemind static-cam --device clip.mp4 --loop --token <t>`.

## Narration notes

Lead with the idea (the phone is the robot's body; memory follows you around), then show,
don't explain, each check. Mention the Meta models on screen as they act: Muse Spark
(agent, display reading, report), Voice Transcribe (hands-free), SAM 3.1 + Perception
Encoder (the cameras), and that it all runs on one RTX 5090 plus the Meta API for about $1.
