import requests

url = "http://127.0.0.1:5001/analyze"
text = "The army helped rescue injured civilians and maintained law and order."

response = requests.post(url, json={"text": text})

if response.status_code == 200:
    print("✅ Backend is working! Response:")
    print(response.json())
else:
    print("❌ Backend error:", response.status_code)
    print(response.text)
