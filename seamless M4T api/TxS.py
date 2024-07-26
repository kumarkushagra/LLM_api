import requests

def text_to_speech(text, src_lang, tgt_lang, api_token):
    api_url = "https://api-inference.huggingface.co/models/facebook/seamless-m4t-v2-large"
    headers = {"Authorization": f"Bearer {api_token}"}
    payload = {
        "inputs": text,
        "parameters": {"src_lang": src_lang, "tgt_lang": tgt_lang, "generate_speech": True}
    }
    response = requests.post(api_url, headers=headers, json=payload)
    return response.content  # This will be the audio array
