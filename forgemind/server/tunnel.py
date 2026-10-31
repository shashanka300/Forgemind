"""Public HTTPS link via zrok (ported from the gallery app).

Phones only allow the live camera on HTTPS (or localhost), so the phone uses
this link. Works with zrok 1.x (`zrok`) and 2.x (`zrok2`); it must already
be enabled (`zrok enable <account token>`).
"""
import re
import shutil
import subprocess
import threading


def _release_name(exe, name):
    """Delete a dead share still holding our reserved name.

    A server that was killed never tells zrok it stopped, and the next share is refused
    ("name ... is in use by share ...").
    """
    out = subprocess.run([exe, "list", "shares"], capture_output=True, encoding="utf-8",
                         errors="replace", timeout=30).stdout or ""
    for line in out.splitlines():
        m = re.search(r"\b([a-z0-9]{12})\b", line)
        if m and f"{name}." in line:
            subprocess.run([exe, "delete", "share", m.group(1)], capture_output=True, timeout=30)


def start_zrok(port, on_url=None, name=None):
    """`name`: a reserved zrok name (FM_ZROK_NAME), so the address survives restarts.

    The link never carries the access token: the page asks for it (app.py /login).
    """
    exe = shutil.which("zrok2") or shutil.which("zrok")
    if not exe:
        print("zrok not found. Install it from https://zrok.io, run "
              "`zrok enable <your account token>`, then try again.")
        return None
    if name:
        _release_name(exe, name)
    proc = subprocess.Popen(
        [exe, "share", "public", "--headless", *(["-n", f"public:{name}"] if name else []),
         f"http://127.0.0.1:{port}"],
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
                link = f"{url}/"
                print("\n  phone link (asks for the access token):")
                print(f"  {link}")
                print(f"  bench phone:  {link}?role=static\n")
                if on_url:
                    on_url(link)
                announced = True
            elif not announced and line.strip():
                print(f"  zrok: {line.rstrip()}")
        if proc.poll() not in (None, 0):
            print(f"zrok exited with code {proc.returncode}")

    threading.Thread(target=read, daemon=True).start()
    return proc
