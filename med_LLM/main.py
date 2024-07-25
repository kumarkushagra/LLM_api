# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from model import generate_response
 
app = FastAPI()
 
class PromptRequest(BaseModel):
    prompt: str
 
@app.post("/generate")
async def generate(prompt_request: PromptRequest):
    try:
        prompt = prompt_request.prompt
        response = generate_response(prompt)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
 
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
 