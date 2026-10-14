"""Plain-English check of a phone-as-camera connection (DroidCam, IP Webcam, Iriun, Camo). Run it with Check_phone_camera.bat.

It looks at, in this order: is the phone app's PC client running, which cameras Windows lists and whether pictures arrive from them,
and (if you type the phone's address) whether the phone can be reached over Wi-Fi, whether it serves a video stream, and whether this
program can read that stream. It only reads: it changes no settings and sends nothing except one request to the address you type.
The result is printed and saved to phone_camera_check.txt in the project folder.
"""
import os
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")          # before OpenCV is loaded: no driver chatter in the report

STREAM_PATHS = ("/video", "/mjpegfeed?1280x720", "/mjpegfeed")       # DroidCam (new, old) and IP Webcam use these
PHONE_CAMERA_WORDS = ("droidcam", "iriun", "camo", "epoccam", "ip webcam")


# ------------------------------------------------------------------------------------------------ probes
def processes_matching(words=("droidcam",)):
    """Names of running programs whose name contains one of the words (Windows `tasklist`)."""
    try:
        out = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True, timeout=15).stdout
    except Exception:  # noqa: BLE001 - no tasklist: nothing known
        return []
    names = {line.split('","')[0].strip('"') for line in out.splitlines() if line.strip()}
    return sorted(n for n in names if any(w in n.lower() for w in words))


def tcp_reachable(host, port, timeout=3.0):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True, ""
    except socket.timeout:
        return False, "no answer within 3 seconds (a different network, a firewall or client isolation)"
    except ConnectionRefusedError:
        return False, "the phone refused the connection (the phone app is not running or is on another port)"
    except OSError as e:
        return False, str(e)


def http_probe(url, timeout=5.0):
    """What a stream address answers: status, content type, and whether the first bytes look like a video stream."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "limbtrack-check"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            ctype = r.headers.get("Content-Type", "")
            head = r.read(2048)
        stream = "multipart" in ctype.lower() or head[:2] == b"\xff\xd8" or b"--" in head[:200] and b"Content-Type: image/jpeg" in head
        return {"status": r.status, "content_type": ctype, "stream": bool(stream), "error": ""}
    except Exception as e:  # noqa: BLE001 - reported as text
        return {"status": None, "content_type": "", "stream": False, "error": f"{type(e).__name__}: {e}"}


def read_frames(source, seconds=3.0):
    """Open a camera index or stream address with the program's own camera class and count pictures."""
    from .camera import CameraOpenTimeout, CameraSource, call_with_timeout
    try:
        cam = call_with_timeout(lambda: CameraSource(source), 45.0, "The camera")      # (the camera class has its own, shorter, wait)
    except CameraOpenTimeout as e:
        return {"frames": 0, "fps": 0.0, "size": None, "backend": None, "error": str(e)}
    except Exception as e:  # noqa: BLE001
        return {"frames": 0, "fps": 0.0, "size": None, "backend": None, "error": str(e)}
    n, t0, size = 0, time.perf_counter(), None
    try:
        while time.perf_counter() - t0 < seconds:
            f, _ = cam.read()
            if f is not None:
                n += 1
                size = (f.shape[1], f.shape[0])
        return {"frames": n, "fps": n / seconds, "size": size, "backend": getattr(cam, "backend", None), "error": ""}
    finally:
        cam.close()


# ------------------------------------------------------------------------------------------------ the verdict (plain language)
def verdict(f):
    """Plain-English lines from the findings `f` (a dict, see main()). Pure: easy to test."""
    out = []
    phone_cams = f.get("phone_cameras", [])        # [(index, name, read_result)]
    if f.get("client_running") is False and not phone_cams:
        out.append("The phone's PC program (for example DroidCam Client) is not running and no phone camera is listed. Install/start it on "
                   "this PC first, then connect it to the phone inside that program.")
    for index, name, r in phone_cams:
        if r["frames"] > 0:
            out.append(f"OK: '{name}' (camera {index}) delivers pictures: {r['size'][0]}x{r['size'][1]}, {r['fps']:.0f} fps"
                       + (f" via {r['backend']}" if r.get("backend") else "") + ". In the program choose it in the camera list and press Start camera.")
        else:
            out.append(f"'{name}' (camera {index}) is listed but no pictures arrive ({r['error'] or 'nothing was delivered'}). The usual cause: the "
                       "PC program is open but NOT connected to the phone. In DroidCam Client add your phone (new client: Add > DroidCam, then "
                       "Refresh Device List; older client: the Wi-Fi or USB tab, then Start) and wait until the preview in that window moves. "
                       "Only then can this program see the picture. Also close any other program that uses the camera (browser tab, "
                       "Teams, OBS). If the preview IS moving and this still fails, close this program completely and run the check again.")
    addr = f.get("address")
    if addr:
        reach = f.get("reach")
        if reach is not None and not reach[0]:
            out.append(f"The phone at {addr['host']}:{addr['port']} cannot be reached: {reach[1]}. Check: the phone app is open and in the "
                       "foreground with the screen on; the phone and this PC are on the SAME Wi-Fi (not 'guest' or a different band with "
                       "client isolation; a VPN on the PC can also block it); the IP address on the phone screen has not changed (it often "
                       "does after a restart: use the one shown now); Windows Firewall is not blocking the DroidCam program.")
        elif reach is not None:
            good = [p for p in f.get("paths", []) if p["probe"]["stream"]]
            if not good:
                bad = f.get("paths", [])
                out.append(f"The phone answers on port {addr['port']} but none of the usual video addresses delivered a stream "
                           f"({', '.join(p['path'] for p in bad)}). Open http://{addr['host']}:{addr['port']} in a browser to see what the "
                           "phone app offers and use the video address it shows. Some phone apps allow only ONE viewer: close the DroidCam "
                           "Client or any browser tab that is already watching.")
            else:
                best = good[0]
                if best["read"]["frames"] > 0:
                    out.append(f"OK: {best['url']} works: {best['read']['size'][0]}x{best['read']['size'][1]}, {best['read']['fps']:.0f} fps. "
                               f"Type exactly this address in the stream box and press Connect to stream.")
                else:
                    out.append(f"The phone serves a video stream at {best['url']} but this program could not read it "
                               f"({best['read']['error'] or 'no pictures'}). Close other viewers of the stream (the phone may allow only one), "
                               "or use the DroidCam Client route (camera list) instead.")
    if not out:
        out.append("Nothing to report: no phone camera was found and no address was tried.")
    return out


# ------------------------------------------------------------------------------------------------ the program
def main(argv=None):
    from .camera import camera_names
    lines = []

    def say(text=""):
        print(text, flush=True)
        lines.append(text)

    say("Phone camera check (reads only; nothing is changed)\n")
    findings = {}
    procs = processes_matching(("droidcam", "iriun", "camo", "epoccam"))
    findings["client_running"] = bool(procs)
    say("Phone camera programs running: " + (", ".join(procs) if procs else "none found"))
    names = camera_names()
    say("Cameras Windows lists: " + (", ".join(f"{i}: {n}" for i, n in enumerate(names)) if names else "none"))
    findings["phone_cameras"] = []
    for i, n in enumerate(names):
        if any(w in n.lower() for w in PHONE_CAMERA_WORDS):
            say(f"Trying camera {i} '{n}' (waits up to 20 s if the phone is not connected) ...")
            r = read_frames(i, 3.0)
            say(f"  pictures: {r['frames']}" + (f", {r['size'][0]}x{r['size'][1]}, {r['fps']:.1f} fps" if r["frames"] else f"  ({r['error'] or 'none'})"))
            findings["phone_cameras"].append((i, n, r))
    typed = (argv[0] if argv else input("\nPhone address for Wi-Fi, as shown on the phone (for example 192.168.0.100:4747). "
                                        "Press Enter to skip: ")).strip()
    if typed:
        url = typed if "://" in typed else "http://" + typed
        parts = urlsplit(url)
        host, port = parts.hostname, parts.port or 4747
        findings["address"] = {"host": host, "port": port}
        say(f"\nReaching {host}:{port} ...")
        findings["reach"] = tcp_reachable(host, port)
        say("  " + ("reachable" if findings["reach"][0] else "NOT reachable: " + findings["reach"][1]))
        findings["paths"] = []
        if findings["reach"][0]:
            for path in ([parts.path + (("?" + parts.query) if parts.query else "")] if parts.path not in ("", "/") else STREAM_PATHS):
                u = f"http://{host}:{port}{path}"
                say(f"Asking {u} ...")
                probe = http_probe(u)
                say(f"  {'video stream' if probe['stream'] else 'not a video stream'} "
                    f"({probe['status'] or probe['error']}{', ' + probe['content_type'] if probe['content_type'] else ''})")
                read = {"frames": 0, "fps": 0.0, "size": None, "error": ""}
                if probe["stream"]:
                    say("  trying to read it with this program for 3 seconds ...")
                    read = read_frames(u, 3.0)
                    say(f"  pictures: {read['frames']}" + (f", {read['size'][0]}x{read['size'][1]}, {read['fps']:.1f} fps" if read["frames"] else f"  ({read['error'] or 'none'})"))
                findings["paths"].append({"path": path, "url": u, "probe": probe, "read": read})
                if read["frames"]:
                    break
    say("\nWhat this means:")
    for v in verdict(findings):
        say("- " + v)
    try:
        Path("phone_camera_check.txt").write_text("\n".join(lines), encoding="utf-8")
        print("\n(Saved to phone_camera_check.txt)")
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    code = main(sys.argv[1:])
    sys.stdout.flush()
    os._exit(code)                      # a driver call that never returned must not keep the window open
