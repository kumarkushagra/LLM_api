# TxS.py
import torchaudio
from main import load_model

def text_to_speech(input_text, input_lang, output_lang):
    processor, model = load_model("TxS")
    text_inputs = processor(text=input_text, src_lang=input_lang, return_tensors="pt")
    audio_array = model.generate(**text_inputs, tgt_lang=output_lang)[0].cpu().squeeze()
    audio_path = "output.wav"
    torchaudio.save(audio_path, audio_array, sample_rate=model.config.sampling_rate)
    return audio_path

if __name__ == "__main__":
    input_text = "Hello, how are you?"
    input_lang = "eng"
    output_lang = "fra"
    print(text_to_speech(input_text, input_lang, output_lang))
