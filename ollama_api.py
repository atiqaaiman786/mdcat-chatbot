# ollama_api.py

import requests

def query_ollama(prompt, model="gemma:2b"):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False}
        )
        if response.status_code == 200:
            return response.json()['response']
        else:
            return f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"Exception: {str(e)}"
