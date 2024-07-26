import os
import scipy.io.wavfile
from transformers import *
from IPython.display import Audio


tokenizer = AutoTokenizer.from_pretrained(r"F:\facebook\hf-seamless-m4t-large")
model = AutoModelForSeq2SeqLM.from_pretrained(r"F:\facebook\hf-seamless-m4t-large")
processor = AutoProcessor.from_pretrained(r"F:\facebook\hf-seamless-m4t-large")
text = "Hello How are you . SeamlessM4TModel is transformers top level model to generate speech and text, but you can also use dedicated models that perform the task without additional components, thus reducing the memory footprint. For example, you can replace the audio-to-audio generation snippet with the model dedicated to the S2ST task, the rest is exactly the same code:"
# Process text input
text_inputs = processor(text=text, src_lang="eng", return_tensors="pt")
 
# Generate speech output
audio_array_from_text = model.generate(**text_inputs, tgt_lang="eng")[0].cpu().numpy().squeeze()
 
# Save the audio to a file
import scipy.io.wavfile
sample_rate = model.config.sampling_rate
scipy.io.wavfile.write("output1.wav", rate=sample_rate, data=audio_array_from_text)
 
 
Audio(audio_array_from_text, rate=sample_rate)