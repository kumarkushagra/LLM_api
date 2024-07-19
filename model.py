from huggingface_hub import hf_hub_download
from llama_cpp import Llama
 
def generate_response(prompt):
    model_name = "mradermacher/Llama3-Med42-8B-GGUF"
    model_file = "Llama3-Med42-8B.Q8_0.gguf"
    model_path = hf_hub_download(repo_id=model_name, filename=model_file)
 
    llm = Llama(
        model_path=model_path,
        n_ctx=1000,
        n_threads=32,
        n_gpu_layers=0
    )
 
    generation_kwargs = {
        "max_tokens": 20000,
        "stop": ["</s>"],
        "echo": False,
        "top_k": 1
    }
 
    res = llm(prompt, **generation_kwargs)
    generated_text = res["choices"][0]["text"].strip()
 
    return generated_text
 
if __name__ == "__main__":
    # Example prompt for testing
    test_prompt = """
        [INST]
        <<SYS>>
        Read the radiologist report given below and return a one-word prompt indicating whether it is Normal or an abnormality in the following format:
        Normal
        or
        <abnormality>_<type of abnormality>
        Your response MUST be in one word.
        Choose the abnormality from the following only (if not found, return Bleed_Others):
        - Bleed_Epidural
        - Bleed_Subdural
        - Bleed_Subarachnoid
        - Bleed_Contusion
        - Bleed_Intraventricular
        - Bleed_Hematoma
        - Bleed_Hemorrhage
        - Bleed_Others
        - Fracture (do not specify the type of fracture)
        - Midline_shift
        - Cervical
 
        Your response MUST be one of these words.
        Do NOT provide explanations.
        Do NOT add any other text.
        Do NOT give explanations.
        Do NOT add any context.
        ONLY return the one-word prompt from the list above.
 
        Example:
        Input: "Fracture left zygoma, left orbital floor, left lateral orbital wall, anterior and lateral wall of maxillary sinus, and bilateral nasal bone."
        Output: Fracture
        <</SYS>>
 
        Radiologist Report: {
        Fracture left zygoma, left orbital floor, left lateral orbital wall, anteior and lateral wall of maxillary sinus, and bilateral nasal bone.
 
 
        }
        SUMMARIZE IN ONLY ONE WORD
        Output:
        [/INST]
        """
 
    # Call the function and print the generated response
    generated_response = generate_response(test_prompt)
    print("Generated Response:", generated_response)
 