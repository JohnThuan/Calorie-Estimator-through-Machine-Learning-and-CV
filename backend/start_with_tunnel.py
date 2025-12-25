"""
Start the FastAPI backend with ngrok tunnel
This allows your phone to access the backend from anywhere
"""
from pyngrok import ngrok
import uvicorn
import threading

# Start ngrok tunnel
public_url = ngrok.connect(8000)
print("\n" + "="*60)
print("Backend Server with Tunnel")
print("="*60)
print(f"\nYour Backend Tunnel URL: {public_url}")
print(f"\nUpdate mobile/src/services/api.ts:")
print(f"const API_BASE_URL = '{public_url}';")
print("\n" + "="*60 + "\n")

# Start FastAPI server
uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)


