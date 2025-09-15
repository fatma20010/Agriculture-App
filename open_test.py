import os
import webbrowser
import http.server
import socketserver
import threading

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the test HTML file
test_html_path = os.path.join(current_dir, 'test_cors.html')

# Check if the file exists
if not os.path.exists(test_html_path):
    print(f"Error: {test_html_path} not found")
    exit(1)

# Start a simple HTTP server to serve the HTML file
PORT = 8000
Handler = http.server.SimpleHTTPRequestHandler

def start_server():
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Serving at http://localhost:{PORT}")
        httpd.serve_forever()

# Start the server in a separate thread
server_thread = threading.Thread(target=start_server, daemon=True)
server_thread.start()

# Open the browser
url = f"http://localhost:{PORT}/test_cors.html"
print(f"Opening {url} in your default browser...")
webbrowser.open(url)

# Keep the server running until the user presses Ctrl+C
try:
    while True:
        input("Press Ctrl+C to stop the server...")
except KeyboardInterrupt:
    print("\nServer stopped.") 