import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import socket
import uvicorn
from app.main import app

if __name__ == "__main__":
    print("🚀 Starting Deepfake Detection Web Interface...")
    
    # Check if port 8000 is already in use
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = 8000
    
    # Try to bind to port 8000, if it fails try other ports up to 8010
    for test_port in range(8000, 8011):
        try:
            sock.bind(('0.0.0.0', test_port))
            port = test_port
            break
        except socket.error:
            print(f"⚠️  Port {test_port} is already in use, trying next port...")
            continue
    
    # Close the socket
    sock.close()
    
    print(f"🌐 Access the application at http://localhost:{port}")
    
    try:
        uvicorn.run(
            "app.main:app", 
            host="0.0.0.0", 
            port=port,
            timeout_keep_alive=300,
            limit_concurrency=10,
            timeout_graceful_shutdown=60,
        )
    except Exception as e:
        print(f"❌ Error starting server: {e}")