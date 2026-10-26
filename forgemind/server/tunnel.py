"""Public HTTPS link via zrok (ported from the gallery app).

Phones only allow the live camera on HTTPS (or localhost), so the phone uses
this link. Works with zrok 1.x (`zrok`) and 2.x (`zrok2`); it must already
be enabled (`zrok enable <account token>`).
"""
import re
import shutil
import subprocess
import threading
from urllib.parse import quote


def start_zrok(port, token, on_url=None):
    exe = shutil.which("zrok2") or shutil.which("zrok")
    if not exe:
        print("zrok not found. Install it from https://zrok.io, run "
              "`zrok enable <your account token>`, then try again.")
        return None
    proc = subprocess.Popen(
        [exe, "share", "public", "--headless", f"http://127.0.0.1:{port}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def read():
        announced = endpoints = False
        for line in proc.stdout:
            endpoints = endpoints or "following endpoints" in line
            rest = line.split("following endpoints", 1)[-1].replace("\\n", " ")
            m = (re.search(r"https?://([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)", rest)
                 or re.search(r"\b([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+){2,})\b", rest))
            if endpoints and m and not announced:
                url = f"https://{m.group(1)}"
                link = f"{url}/?t={quote(token)}" if token else url
                print("\n  phone link (keep it private - it unlocks ForgeMind):")
                print(f"  {link}")
                print(f"  bench phone:  {link}{'&' if token else '?'}role=static\n")
                if on_url:
                    on_url(link)
                announced = True
            elif not announced and line.strip():
                print(f"  zrok: {line.rstrip()}")
        if proc.poll() not in (None, 0):
            print(f"zrok exited with code {proc.returncode}")

    threading.Thread(target=read, daemon=True).start()
    return proc
