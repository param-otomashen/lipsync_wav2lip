
# .venv/inference_wrapper.py

import sys
import os
import torch
import numpy as np
import cv2
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'wav2lip')) 
sys.path.append(path) 
print(path)
import audio as audio
from inference import load_model, datagen, mel_step_size

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'wav2lip')))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL = None

def load_wav2lip_model(checkpoint_path):
    global MODEL
    if MODEL is None:
        MODEL = load_model(checkpoint_path)
    return MODEL

def generate_video(input_video_path, audio_path, output_path, checkpoint_path, model):
    import hashlib

    def quick_video_hash(path): 
        st = os.stat(path)
        key = f"{os.path.abspath(path)}|{st.st_size}|{int(st.st_mtime)}"
        return hashlib.sha256(key.encode()).hexdigest()


    print(f"input_video_path is : {input_video_path}")
    
    video_hash = quick_video_hash(input_video_path)
    print(f"hash path for generate_video function is; {video_hash}")
    frames_cache_dir = r"C:\Users\Jasraj\Downloads\wav2lip\wav2lip\wav2lip\cache"
    #os.makedirs(frames_cache_dir, exist_ok=True)
    frames_cache_path = os.path.join(frames_cache_dir, f"{video_hash}_frames.npy") 

    if os.path.exists(frames_cache_path):
        print("✅ Loading cached video frames...")
        frames = np.load(frames_cache_path, allow_pickle=True) #taking lot of time
        frames = frames.tolist()      
        #frames = [f.astype(np.uint8) for f in frames]  
        
    else: 
        print("📥 Reading frames from video...")
        video_stream = cv2.VideoCapture(input_video_path)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        if not video_stream.isOpened():
            raise IOError(f"❌ Failed to open video file: {input_video_path}")
        frames = []
        while True:
            success, frame = video_stream.read()
            if not success:
                break
            if frame is None or len(frame.shape) != 3 or frame.shape[2] != 3:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        video_stream.release()
        frames_uint8 = [f.astype(np.uint8) for f in frames]

        np.save(frames_cache_path, np.array(frames_uint8, dtype=object), allow_pickle=True)  # Save as object array for ragged data

    print(f"Number of frames is: {len(frames)}")
    fps = fps if 'fps' in locals() else 25  # fallback if cache was used

    print("loading audio.load_wav function.. ")
    wav = audio.load_wav(audio_path, 16000)
    print("done with load_wav..")
    mel = audio.melspectrogram(wav)
    print("done with audio.melspectrogram..")
    if np.isnan(mel.reshape(-1)).sum() > 0:
        raise ValueError('Mel contains nan!')

    mel_chunks = []
    mel_idx_multiplier = 80. / fps
    i = 0
    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
        i += 1

    frames = frames[:len(mel_chunks)]
    gen = datagen(frames.copy(), mel_chunks)

    frame_h, frame_w = frames[0].shape[:-1] 
    os.makedirs('temp', exist_ok=True)
    out = cv2.VideoWriter('temp/result.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_w, frame_h))

    for img_batch, mel_batch, frame_batch, coords_batch in gen:
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

        for p, f, c in zip(pred, frame_batch, coords_batch):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            f[y1:y2, x1:x2] = p 
            f_bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            out.write(f_bgr)

    out.release()
    ffmpeg_cmd = f'ffmpeg -y -i temp/result.avi -i "{audio_path}" -c:v libx264 -preset slow -crf 18 -c:a aac -b:a 192k -shortest "{output_path}"'
    os.system(ffmpeg_cmd)
    print(f"✅ Video saved to {output_path}")
