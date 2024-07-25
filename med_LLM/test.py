 
import requests
 
def test_generate_response(prompt):
    url = "http://localhost:8000/generate"
    response = requests.post(url, json={"prompt": prompt})
   
    if response.status_code == 200:
        json_response = response.json()
        if "response" in json_response:
            print("Generated response:", json_response["response"])
        else:
            print("Error: 'response' key not found in the response JSON")
    else:
        print("Error:", response.status_code, response.text)
 
if __name__ == "__main__":
    prompt = input("Enter your prompt: ")
    test_generate_response(prompt)
 
 