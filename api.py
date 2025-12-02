#!/usr/bin/env python3
"""
OTA API for Smart Posture Assistant
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
                result = subprocess.run(
                    ['git', 'rev-parse', '--short', 'HEAD'],
                    cwd=str(REPO_DIR),
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                version = result.stdout.strip()
                
                status_result = subprocess.run(
                    ['systemctl', 'is-active', 'posture-bot'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                bot_running = status_result.stdout.strip() == 'active'
                
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
        """Handle POST requests"""
        
        if self.path == '/control':
            auth_header = self.headers.get('Authorization', '')
            if auth_header != f"Bearer {SECRET}":
                self._send_json({"error": "Unauthorized"}, 401)
                return
            
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length).decode() if content_length > 0 else "{}"
                data = json.loads(body)
                action = data.get('action', '')
                
                if action not in ['start', 'stop', 'restart', 'status']:
                    self._send_json({
                        "error": "Invalid action. Use: start/stop/restart/status"
                    }, 400)
                    return
                
                result = subprocess.run(
                    ['sudo', 'systemctl', action, 'posture-bot'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                status_result = subprocess.run(
                    ['systemctl', 'is-active', 'posture-bot'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                bot_running = status_result.stdout.strip() == 'active'
                
                self._send_json({
                    "success": result.returncode == 0,
                    "action": action,
                    "bot_running": bot_running,
                    "message": f"Bot {action} {'successful' if result.returncode == 0 else 'failed'}"
                })
                
            except subprocess.TimeoutExpired:
                self._send_json({"error": "Command timeout"}, 500)
            except Exception as e:
                self._send_json({"error": str(e)}, 500)
            
            return
        
        elif self.path == '/update':
            auth_header = self.headers.get('Authorization', '')
            if auth_header != f"Bearer {SECRET}":
                self._send_json({
                    "error": "Unauthorized",
                    "hint": "Check UPDATE_SECRET environment variable"
                }, 401)
                return
            
            try:
                result = subprocess.run(
                    [str(UPDATE_SCRIPT)],
                    cwd=str(REPO_DIR),
                    capture_output=True,
                    text=True,
                    timeout=90
                )
                
                output = result.stdout.strip()
                
                if "NO_UPDATE" in output:
                    self._send_json({
                        "success": True,
                        "updated": False,
                        "message": "Already up to date"
                    })
                elif "SUCCESS" in output:
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
                        "output": output[-300:]
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
    print("OTA Update API Started")
    print("=" * 60)
    print(f"   Port: {port}")
    print(f"   Health: http://0.0.0.0:{port}/health")
    print(f"   Update: POST http://0.0.0.0:{port}/update")
    print(f"   Control: POST http://0.0.0.0:{port}/control")
    print(f"   Secret: {SECRET}")
    print("=" * 60)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
        server.shutdown()

if __name__ == '__main__':
    run()

