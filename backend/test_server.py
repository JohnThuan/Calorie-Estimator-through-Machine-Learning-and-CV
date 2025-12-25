"""
Quick test to check if server is accessible
"""
import requests

try:
    response = requests.get('http://localhost:8000/')
    print("SUCCESS: Server accessible!")
    print(f"Response: {response.json()}")
except Exception as e:
    print(f"ERROR: Server not accessible: {e}")

