# TxT.py
from main import load_model

def text_to_text(input_text, input_lang, output_lang):
    processor, model = load_model("TxT")
    text_inputs = processor(text=input_text, src_lang=input_lang, return_tensors="pt")
    output_tokens = model.generate(**text_inputs, tgt_lang=output_lang)
    translated_text = processor.decode(output_tokens[0], skip_special_tokens=True)
    return translated_text

if __name__ == "__main__":
    input_text = "Hello, how are you?"
    input_lang = "eng"
    output_lang = "fra"
    print(text_to_text(input_text, input_lang, output_lang))
