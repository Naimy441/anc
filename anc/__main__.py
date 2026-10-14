"""Entry point: ``python -m anc`` starts the engine + UI on http://127.0.0.1:8765."""
from __future__ import annotations

import argparse
import multiprocessing as mp
import threading
import webbrowser


def main() -> None:
    ap = argparse.ArgumentParser(prog="anc", description="Real-time noise cancellation")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="don't open the UI automatically")
    ap.add_argument("--simulate", action="store_true",
                    help="use a synthetic noisy-voice source instead of audio hardware")
    ap.add_argument("--simulate-feedback", action="store_true",
                    help="with --simulate: add a speaker-to-mic loop that howls at 4.8 kHz")
    ap.add_argument("--list-devices", action="store_true", help="print audio devices and exit")
    args = ap.parse_args()

    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return

    import os

    import uvicorn

    if args.simulate_feedback:
        os.environ["ANC_SIM_FEEDBACK"] = "1"

    from .server import create_app

    app = create_app(simulate=args.simulate)
    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host}:{args.port}"
    print(f"\n  ANC is running →  {url}\n  Press Ctrl+C to quit.\n", flush=True)
    if not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", ws_ping_interval=20)


if __name__ == "__main__":
    mp.freeze_support()
    main()
