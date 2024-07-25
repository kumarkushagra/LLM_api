# # import the model here and import diffeerent functions to perform (Text X Speach) tasks

# # Use a pipeline as a high-level helper
# from transformers import pipeline

# pipe = pipeline("automatic-speech-recognition", model="facebook/seamless-m4t-v2-large")


# main.py
from transformers import AutoProcessor, SeamlessM4TForTextToText, SeamlessM4TForTextToSpeech, SeamlessM4TForSpeechToText, SeamlessM4TForSpeechToSpeech

def load_model(task):
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    if task == "TxT":
        model = SeamlessM4TForTextToText.from_pretrained("facebook/seamless-m4t-v2-large")
    elif task == "TxS":
        model = SeamlessM4TForTextToSpeech.from_pretrained("facebook/seamless-m4t-v2-large")
    elif task == "SxT":
        model = SeamlessM4TForSpeechToText.from_pretrained("facebook/seamless-m4t-v2-large")
    elif task == "SxS":
        model = SeamlessM4TForSpeechToSpeech.from_pretrained("facebook/seamless-m4t-v2-large")
    else:
        raise ValueError("Invalid task")
    return processor, model
