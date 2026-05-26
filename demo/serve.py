import http.server
import mimetypes
import webbrowser

mimetypes.add_type('application/octet-stream', '.onnx')
mimetypes.add_type('application/wasm', '.wasm')

class Handler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"  {self.path}  →  {args[1]}")

print("Serving at http://localhost:8080")
url = 'http://localhost:8080/runway_seg_demo.html'
print(f"Serving at {url}")
server = http.server.HTTPServer(('', 8080), Handler)
try:
    webbrowser.open(url)
except Exception:
    pass
server.serve_forever()