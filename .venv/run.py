'''
import subprocess
import sys
import os
import cv2 

def run_inference():  
    venv_python = os.path.abspath(os.path.join(os.getcwd(), "Scripts", "python.exe")) 
    inference_path = os.path.abspath("../wav2lip/inference.py") 

    img_path = os.path.abspath("../wav2lip/examples/girl.png") 

    audio_path = os.path.abspath("../wav2lip/examples/audio.wav") 

    command = [
        venv_python,
        inference_path,
        #"--checkpoint_path", os.path.abspath("../wav2lip/checkpoints/wav2lip.pth"),
        "--checkpoint_path", os.path.abspath("../wav2lip/checkpoints/wav2lip_gan.pth"),
        "--face", img_path,
        "--audio", audio_path, 
        "--pads", "0", "20", "0", "0", # controls how much extra space is added to the image.
        "--resize_factor", "1", # 1 means no resizing, 2 means double the size, etc.
        #top bottom left right. 
    ]

    print("Running command:", ' '.join(command))
    subprocess.run(command, cwd=os.path.abspath("../wav2lip"))

    # Path to output file from Wav2Lip (adjust if needed)
    raw_video_path = os.path.abspath("../wav2lip/temp/result.avi")  # or temp/result.avi
    output_path = os.path.abspath("../wav2lip/results/result_voice.mp4")

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", raw_video_path,
        "-i", audio_path,
        "-qscale:v", "2",  # High quality
        "-c:a", "aac",
        "-strict", "experimental",
        output_path
    ]

    print("Running FFmpeg command:", ' '.join(ffmpeg_cmd))
    subprocess.run(ffmpeg_cmd)

if __name__ == "__main__":
    run_inference()

    
# run.py
import os
from inference_wrapper import generate_video

def run_inference(model):
    face_path = os.path.abspath("../wav2lip/examples/avatar.jpg")
    audio_path = os.path.abspath("../wav2lip/examples/audio.wav")
    output_path = os.path.abspath("../wav2lip/results/result_voice.mp4")
    checkpoint_path = os.path.abspath("../wav2lip/checkpoints/wav2lip_gan.pth")

    generate_video(face_path, audio_path, output_path, checkpoint_path, model)
''' 

# .venv/run.py
import os
from inference_wrapper import generate_video, load_wav2lip_model

def run_inference(model):
    input_video_path = os.path.abspath("../wav2lip/examples/input_video.mp4")
    audio_path = os.path.abspath("../wav2lip/examples/audio.wav")
    output_path = os.path.abspath("../wav2lip/results/result_voice.mp4")
    checkpoint_path = os.path.abspath("../wav2lip/checkpoints/wav2lip_gan.pth")

    #model = load_wav2lip_model(checkpoint_path)
    generate_video(input_video_path, audio_path, output_path, checkpoint_path, model)

if __name__ == "__main__":
    run_inference()




