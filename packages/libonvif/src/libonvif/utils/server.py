from libonvif.utils.xml import text, NS
from libonvif.datastructures.event import parse_notify
import http.server
import socketserver
import signal
import sys
from functools import partial
from threading import Thread
from typing import Callable, Any

def handle_sigterm(signum, frame):
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_sigterm)

class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    disable_nagle_algorithm = True

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, my_arg=None, **kwargs):
        self.my_arg = my_arg
        super().__init__(*args, **kwargs)

    def do_POST(self):
        if self.path == "/onvif/events":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            xml = body.decode("utf-8")
            alarms = parse_notify(self.client_address[0], xml)
            self.my_arg(alarms) if self.my_arg else ...
            self.send_response(200)
            self.end_headers()

    def log_message(self, format, *args):
        pass

class EventServer:
    def __init__(self, ip_address: str, port: int, callback: Callable[[str, list[dict[str, Any]]], None]):
        self.ip_address = ip_address
        self.port = port
        self.callback = callback
        self.httpd = None
        self.thread = None

    def start(self):
        if self.httpd:
            return

        self.thread = Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        if self.httpd:
            self.httpd.shutdown()

    def _run(self):
        handler = partial(Handler, my_arg=self.callback)

        with Server((self.ip_address, self.port), handler) as httpd:
            self.httpd = httpd
            httpd.serve_forever()

# used for development debugging vvv
PORT = 8856

def my_func(arg: list[dict[str, Any]]):
    for item in arg:
        print(item)

if __name__ == "__main__":
    try:
        handler = partial(Handler, my_arg=my_func)
        with Server(("", PORT), handler) as httpd:
            httpd.serve_forever()
    except Exception as ex:
        print(f"HTTP SERVER ERROR: {ex}")

# used for development debugging ^^^
