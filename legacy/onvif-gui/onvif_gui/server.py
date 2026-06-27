import http.server
import socketserver
import os
import signal
import sys
from pathlib import Path

PORT = 8800
if sys.platform == "win32":
    DIRECTORY = f"{os.environ['HOMEPATH']}/.cache/onvif-gui/proxy"
else:
    DIRECTORY = f"{os.environ['HOME']}/.cache/onvif-gui/proxy"

if sys.platform == "win32":
    DIRECTORY = f"{os.environ['HOMEPATH']}/.cache/onvif-gui/proxy"
elif sys.platform == "darwin":
    DIRECTORY = Path(os.path.dirname(__file__)).parent.absolute() / "cache" / "proxy"
elif sys.platform.startswith("linux"):
    DIRECTORY = f"{os.environ['HOME']}/.cache/onvif-gui/proxy"
else:
    print(f"Unknown platform: {sys.platform}")

def handle_sigterm(signum, frame):
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_sigterm)

class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    disable_nagle_algorithm = True

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def do_POST(self):
        if self.path == "/shutdown":
            self.send_response(200)
            self.end_headers()
            self.server.shutdown()
        else:
            super().do_POST()

if __name__ == "__main__":
    try:
        with Server(("", PORT), Handler) as httpd:
            httpd.serve_forever()
    except Exception as ex:
        print(f"HTTP SERVER ERROR: {ex}")

