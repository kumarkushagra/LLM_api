# SxS.py
import torchaudio
from main import load_model

def speech_to_speech(audio_path, input_lang, output_lang):
    processor, model = load_model("SxS")
    audio, orig_freq = torchaudio.load(audio_path)
    audio = torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=16000)
    audio_inputs = processor(audios=audio, return_tensors="pt")
    audio_array = model.generate(**audio_inputs, tgt_lang=output_lang)[0].cpu().squeeze()
    audio_path_output = "output.wav"
    torchaudio.save(audio_path_output, audio_array, sample_rate=model.config.sampling_rate)
    return audio_path_output

if __name__ == "__main__":
    audio_path = "path_to_audio.wav"
    input_lang = "eng"
    output_lang = "fra"
    print(speech_to_speech(audio_path, input_lang, output_lang))
