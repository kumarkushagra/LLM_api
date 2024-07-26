import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForSeq2SeqLM
import scipy.io.wavfile as wav
import numpy as np
from code_id import text_decoder_lang_to_code_id  # Import the dictionary

# Function to load the model on the device
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = r"F:\facebook\hf-seamless-m4t-large"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    return processor, model, device

# Function for text to speech translation
def text_to_speech(input_text, input_lang, output_lang, output_path):
    model_name = r"F:\facebook\hf-seamless-m4t-large"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    text_inputs = processor(text=input_text, src_lang=input_lang, return_tensors="pt")
    # Generate speech output
    audio_array_from_text = model.generate(**text_inputs, tgt_lang=output_lang)[0].cpu().numpy().squeeze()
    # Save the audio to a file
    sample_rate = model.config.sampling_rate
    wav.write(output_path, rate=sample_rate, data=audio_array_from_text.astype(np.float32))
    return output_path

# Function for speech to text translation
def speech_to_text(audio_path, input_lang, output_lang):
    model_name = r"F:\facebook\hf-seamless-m4t-large"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    audio, sample_rate = sf.read(audio_path)
    audio_inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
    # Generate text output
    generated_ids = model.generate(**audio_inputs)
    transcription = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return transcription

# Function for speech to speech translation
def speech_to_speech(audio_path, input_lang, output_lang, output_path):
    model_name = r"F:\facebook\hf-seamless-m4t-large"
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    audio, sample_rate = sf.read(audio_path)
    audio_inputs = processor(audio, sampling_rate=sample_rate, return_tensors="pt")
    audio_inputs["forced_bos_token_id"] = text_decoder_lang_to_code_id[output_lang]
    # Generate translated speech
    generated_speech = model.generate(**audio_inputs)
    audio_array_from_speech = processor.batch_decode(generated_speech, skip_special_tokens=True)[0]
    # Save the audio to a file
    wav.write(output_path, rate=sample_rate, data=np.array(audio_array_from_speech).astype(np.float32))
    return output_path

if __name__ == "__main__":
    # Example usage
    output_speech_path = "output.wav"
    text_to_speech("Hello How are you . SeamlessM4TModel is transformers top level model to generate speech and text, but you can also use dedicated models that perform the task without additional components, thus reducing the memory footprint. For example, you can replace the audio-to-audio generation snippet with the model dedicated to the S2ST task, the rest is exactly the same code", "eng", "fra", output_speech_path)
    print("Text to Speech output saved at:", output_speech_path)
    
    # transcription = speech_to_text("input.wav", "eng", "fra")
    # print("Speech to Text Transcription:", transcription)
    
    # speech_output_path = "speech_output.wav"
    # speech_to_speech("input.wav", "eng", "fra", speech_output_path)
    # print("Speech to Speech output saved at:", speech_output_path)
