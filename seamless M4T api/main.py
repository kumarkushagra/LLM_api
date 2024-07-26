from transformers import AutoProcessor, SeamlessM4TModel
from IPython.display import Audio

from test import text_to_speech

processor = AutoProcessor.from_pretrained(r"F:\facebook\hf-seamless-m4t-large")
model = SeamlessM4TModel.from_pretrained(r"F:\facebook\hf-seamless-m4t-large")

if __name__ == "__main__":
    # Example usage
    input_text = "Hello How are you . SeamlessM4TModel is transformers top level model to generate speech and text, but you can also use dedicated models that perform the task without additional components, thus reducing the memory footprint. For example, you can replace the audio-to-audio generation snippet with the model dedicated to the S2ST task, the rest is exactly the same code"
    output_speech_path = "output.mp3"
    text_to_speech(processor,model, input_text, "eng", "fra", output_speech_path)
    print("Text to Speech output saved at:", output_speech_path)
    
    # transcription = speech_to_text("input.wav", "eng", "fra")
    # print("Speech to Text Transcription:", transcription)
    
    # speech_output_path = "speech_output.wav"
    # speech_to_speech("input.wav", "eng", "fra", speech_output_path)
    # print("Speech to Speech output saved at:", speech_output_path)
