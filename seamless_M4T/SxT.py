# SxT.py
import torchaudio
from main import load_model

def speech_to_text(audio_path, input_lang, output_lang):
    processor, model = load_model("SxT")
    audio, orig_freq = torchaudio.load(audio_path)
    audio = torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=16000)
    audio_inputs = processor(audios=audio, return_tensors="pt")
    output_tokens = model.generate(**audio_inputs, tgt_lang=output_lang)
    translated_text = processor.decode(output_tokens[0], skip_special_tokens=True)
    return translated_text

if __name__ == "__main__":
    audio_path = "path_to_audio.wav"
    input_lang = "eng"
    output_lang = "fra"
    print(speech_to_text(audio_path, input_lang, output_lang))
