from pydub import AudioSegment
import os
import subprocess

def convert_wav_to_mp3(file_path):
    if not file_path.lower().endswith('.wav'):
        raise ValueError("The file is not a .wav file")
    
    mp3_file_path = file_path[:-4] + '.mp3'
    
    try:
        audio = AudioSegment.from_wav(file_path)
        audio.export(mp3_file_path, format='mp3')
    except Exception as e:
        print(f"pydub encountered an issue: {e}")
        print("Falling back to using FFmpeg directly.")
        command = f"ffmpeg -i \"{file_path}\" \"{mp3_file_path}\""
        subprocess.run(command, shell=True)
    
    os.remove(file_path)
    print(f"Converted {file_path} to {mp3_file_path}")

# Example usage
convert_wav_to_mp3(r'F:\LLM_api\seamless M4T api\output1.wav')
