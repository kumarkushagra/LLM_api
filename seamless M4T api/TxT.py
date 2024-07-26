import torch
from transformers import *

from code_id import text_decoder_lang_to_code_id  # Import the dictionary

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = r"F:\facebook\hf-seamless-m4t-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    return model, tokenizer, device


# Function for text to text translation
def text_to_text(input_text, input_lang, output_lang):
    model, tokenizer, device = main()
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    inputs["forced_bos_token_id"] = text_decoder_lang_to_code_id[output_lang]
    outputs = model.generate(**inputs)
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return translated_text