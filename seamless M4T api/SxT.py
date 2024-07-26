import requests

def speech_to_text(audio_file_path, tgt_lang, api_token):
    api_url = "https://api-inference.huggingface.co/models/facebook/seamless-m4t-v2-large"
    headers = {"Authorization": f"Bearer {api_token}"}
    with open(audio_file_path, "rb") as f:
        audio_data = f.read()
    payload = {
        "inputs": audio_data,
        "parameters": {"tgt_lang": tgt_lang, "generate_speech": False}
    }
    response = requests.post(api_url, headers=headers, data=payload)
    return response.json()
