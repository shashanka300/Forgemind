"""Time spoken turns end to end through the hands-free relay (real Voice Transcribe).

    python eval/voice_latency.py "where are my headphones" --server http://127.0.0.1:8770 --token devtoken

Synthesises the question with Windows speech (System.Speech), streams it like the phone
does (16 kHz PCM16 in 20 ms frames, in real time, then silence) and records when each
message comes back, measured from the end of the speech audio:
  - old path: the phone gets "turn" and posts it to /api/chat
  - server turns: the server answers itself and pushes "reply" (?server_turns=1)
Voice Transcribe costs $0.18 per audio hour (~$0.0003 per question).
"""
import argparse
import asyncio
import json
import os
import subprocess
import tempfile
import time
import urllib.request
import wave

FRAME = 320                      # 20 ms of 16 kHz mono


def synthesize(text):
    path = os.path.join(tempfile.gettempdir(), "fm_q_%d.wav" % abs(hash(text)))
    ps = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
          "$f = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, "
          "[System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono); "
          f"$s.SetOutputToWaveFile('{path}', $f); $s.Speak('{text}'); $s.Dispose()")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
    with wave.open(path) as w:
        pcm = w.readframes(w.getnframes())
    # the synthesiser pads the end with silence: time from the last audible sample
    import array
    a = array.array("h", pcm)
    last = max((i for i in range(len(a)) if abs(a[i]) > 500), default=len(a) - 1)
    return pcm[:2 * (last + 1)]


def post_chat(server, token, text):
    req = urllib.request.Request(server.rstrip("/") + "/api/chat", data=json.dumps({"text": text}).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {token}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


async def one(server, token, pcm, server_turns, silence_s=3.0):
    import websockets
    ws_url = server.replace("http", "ws", 1).rstrip("/") + f"/ws/asr?mode=hands_free&t={token}"
    if server_turns:
        ws_url += "&server_turns=1"
    events, speech_end = [], None
    async with websockets.connect(ws_url, max_size=2 ** 22) as ws:
        assert json.loads(await ws.recv())["type"] == "ready"

        async def stream():
            nonlocal speech_end
            t0 = time.perf_counter()
            speech_end = t0 + len(pcm) / 32000          # when the last word has been sent
            audio = pcm + b"\x00\x00" * int(16000 * silence_s)
            for i in range(0, len(audio), FRAME * 2):
                await ws.send(audio[i:i + FRAME * 2])
                await asyncio.sleep(max(0, t0 + (i / 2 + FRAME) / 16000 - time.perf_counter()))

        async def listen():
            while True:
                m = json.loads(await ws.recv())
                events.append((time.perf_counter(), m))
                if m["type"] == "reply" or (m["type"] == "turn" and not server_turns):
                    if m["type"] == "turn":
                        r = await asyncio.to_thread(post_chat, server, token, m["text"])
                        events.append((time.perf_counter(), {"type": "reply", **r}))
                    return
        sender = asyncio.create_task(stream())
        try:
            await asyncio.wait_for(listen(), 20)
        finally:
            sender.cancel()
    rel = lambda t: round((t - speech_end) * 1000)
    out = {"server_turns": server_turns}
    for t, m in events:
        if m["type"] in ("turn", "reply") and m["type"] not in out:
            out[m["type"]] = rel(t)
            if m["type"] == "turn":
                out["early"] = bool(m.get("early"))
            if m["type"] == "reply":
                out["fast"] = bool(m.get("fast"))
                out["answer"] = m.get("reply", "")[:70]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("questions", nargs="+")
    ap.add_argument("--server", default="http://127.0.0.1:8770")
    ap.add_argument("--token", default="devtoken")
    ap.add_argument("--repeat", type=int, default=2)
    a = ap.parse_args()
    for q in a.questions:
        pcm = synthesize(q)
        print(f"\n{q!r}: {len(pcm) / 32000:.2f} s of speech")
        for server_turns in (False, True):
            for _ in range(a.repeat):
                print("  ", asyncio.run(one(a.server, a.token, pcm, server_turns)))


if __name__ == "__main__":
    main()
