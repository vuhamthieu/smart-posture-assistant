#!/usr/bin/env python3
"""
Lightweight OTA API for Smart Posture Assistant
Pure Python - no external dependencies
"""

from http.server import HTTPServer, BaseHTTPRequestHandler
import subprocess
import json
import os
from pathlib import Path

# Configuration
PORT = 8080
REPO_DIR = Path("/home/theo/smart-posture-assistant")
UPDATE_SCRIPT = REPO_DIR / "update.sh"
SECRET = os.getenv("UPDATE_SECRET", "posture_secret_2024")

class OTAHandler(BaseHTTPRequestHandler):
    
    def log_message(self, format, *args):
        """Silent logging to avoid console spam"""
        pass
    
    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self._send_json({}, 204)
    
    def do_GET(self):
        """Health check + version info"""
        if self.path == '/health':
            try:
                # Get current git commit
                result = subprocess.run(
                    ['git', 'rev-parse', '--short', 'HEAD'],
                    cwd=str(REPO_DIR),
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                version = result.stdout.strip()
                
                # Check if bot is running
                pid_file = REPO_DIR / "bot.pid"
                bot_running = False
                if pid_file.exists():
                    try:
                        pid = int(pid_file.read_text().strip())
                        # Check if process exists
                        subprocess.run(['ps', '-p', str(pid)], capture_output=True, check=True)
                        bot_running = True
                    except:
                        pass
                
                self._send_json({
                    "status": "online",
                    "version": version,
                    "device_id": "pi-posture-001",
                    "bot_running": bot_running,
                    "repo": "smart-posture-assistant"
                })
            except Exception as e:
                self._send_json({
                    "status": "error",
                    "error": str(e)
                }, 500)
        else:
            self._send_json({"error": "Not found"}, 404)
    
    def do_POST(self):
        """Trigger OTA update"""
        if self.path == '/update':
            # Check authorization
            auth_header = self.headers.get('Authorization', '')
            expected_auth = f"Bearer {SECRET}"
            
            if auth_header != expected_auth:
                self._send_json({
                    "error": "Unauthorized",
                    "hint": "Check UPDATE_SECRET environment variable"
                }, 401)
                return
            
            # Trigger update
            try:
                result = subprocess.run(
                    [str(UPDATE_SCRIPT)],
                    cwd=str(REPO_DIR),
                    capture_output=True,
                    text=True,
                    timeout=90
                )
                
                output = result.stdout.strip()
                
                # Parse output
                if "NO_UPDATE" in output:
                    self._send_json({
                        "success": True,
                        "updated": False,
                        "message": "Already up to date"
                    })
                elif "SUCCESS" in output:
                    # Extract new version from output
                    new_version = "unknown"
                    if "SUCCESS:" in output:
                        new_version = output.split("SUCCESS:")[1].strip()
                    
                    self._send_json({
                        "success": True,
                        "updated": True,
                        "version": new_version,
                        "message": "Update successful, bot restarted"
                    })
                else:
                    self._send_json({
                        "success": False,
                        "message": "Update script failed",
                        "output": output[-300:]  # Last 300 chars
                    }, 500)
                    
            except subprocess.TimeoutExpired:
                self._send_json({
                    "error": "Update timeout (>90s)",
                    "hint": "Check update.log on Pi"
                }, 500)
            except FileNotFoundError:
                self._send_json({
                    "error": "Update script not found",
                    "path": str(UPDATE_SCRIPT)
                }, 500)
            except Exception as e:
                self._send_json({
                    "error": str(e)
                }, 500)
        else:
            self._send_json({"error": "Not found"}, 404)

def run(port=PORT):
    server = HTTPServer(('0.0.0.0', port), OTAHandler)
    print("=" * 60)
    print("🚀 OTA Update API Started")
    print("=" * 60)
    print(f"   Port: {port}")
    print(f"   Health: http://0.0.0.0:{port}/health")
    print(f"   Update: POST http://0.0.0.0:{port}/update")
    print(f"   Secret: {SECRET}")
    print("=" * 60)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped!")
        server.shutdown()

if __name__ == '__main__':
    run()
