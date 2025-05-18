from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse, audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, face_detection
from models import Wav2Lip
import platform
import hashlib 
import pickle 

# 🔵 Setup argparse but don't parse immediately
parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--checkpoint_path', type=str, help='Name of saved checkpoint to load weights from', required=True)
parser.add_argument('--face', type=str, help='Filepath of video/image that contains faces to use', required=True)
parser.add_argument('--audio', type=str, help='Filepath of video/audio file to use as raw audio source', required=True)
parser.add_argument('--outfile', type=str, help='Video path to save result.', default='results/result_voice.mp4')
parser.add_argument('--static', type=bool, help='If True, use only first video frame', default=False)
parser.add_argument('--fps', type=float, default=25., required=False)
parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], help='Padding (top, bottom, left, right)')
parser.add_argument('--face_det_batch_size', type=int, default=16, help='Face detection batch size')
parser.add_argument('--wav2lip_batch_size', type=int, default=128, help='Wav2Lip batch size')
parser.add_argument('--resize_factor', default=1, type=int, help='Reduce resolution by this factor')
parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], help='Crop (top, bottom, left, right)')
parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], help='Fixed bounding box (optional)')
parser.add_argument('--rotate', default=False, action='store_true', help='Rotate video 90 degrees right')
parser.add_argument('--nosmooth', default=False, action='store_true', help='Prevent smoothing face detections')

args = None  # 🛑 Important: DON'T parse at import

# 🔵 Manual constants
mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

# 🔵 All functions below

face_alignment_detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, flip_input=False, device=device)

def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes

'''
def _hash_images(images):
    """Create a hash based on the content of the list of images."""
    try:
        hash_md5 = hashlib.sha256()
        for img in images:
            hash_md5.update(img.tobytes())
        return hash_md5.hexdigest()
    except Exception as e:
        print("⚠️ Failed to generate hash from images, skipping cache.")
        return None
'''
def _hash_video_path(video_path):
    key = os.path.abspath(video_path).encode()
    return hashlib.sha256(key).hexdigest()

def face_detect(images, pads=[0, 10, 0, 0], nosmooth=False, cache_dir=r"C:\Users\Jasraj\Downloads\wav2lip\wav2lip\wav2lip\cache"):
    print("🔎 Starting face detection...")

    video_hash = _hash_video_path(r"C:\Users\Jasraj\Downloads\wav2lip\wav2lip\wav2lip\examples\input_video.mp4")  
    print(f"hash path for face_detect function is: {video_hash}")
    if video_hash is None: #in case the images themselves could not be hashed due to error
        print("⚠️ Could not hash input images. Skipping face detection cache.") 
    else:
        cache_path = os.path.join(cache_dir, video_hash, 'faces.npz') #images themselves have been hashed, cache_path formed 
        #to save the result inside 
        if os.path.exists(cache_path):  #in case cache_path exists, cached data is loaded from it
            print(f"📦 Loading cached face detection from: {cache_path}")
            data = np.load(cache_path, allow_pickle=True)
            face_imgs = data['face_imgs']
            coords = data['coords']
            return list(zip(face_imgs, coords))

    try:
        detector = face_alignment_detector
        print("detector loaded...")
        batch_size = 1

        while True:
            predictions = []
            try:
                print(f"📸 Number of input frames: {len(images)}")
                for i in range(0, len(images), batch_size):
                    #arr = np.array(images[i:i + batch_size])
                    arr = np.array(images[i:i + batch_size], dtype=np.uint8)
                    predictions.extend(detector.get_detections_for_batch(arr))
            except RuntimeError:
                if batch_size == 1:
                    raise RuntimeError('OOM error in face detection. Try resizing input.')
                batch_size //= 2
                print('Recovering from OOM; New batch size: {}'.format(batch_size))
                continue
            break

        results = []
        pady1, pady2, padx1, padx2 = pads
        for rect, image in zip(predictions, images):
            if rect is None:
                cv2.imwrite('temp/faulty_frame.jpg', image)
                raise ValueError('Face not detected!')

            y1 = max(0, rect[1] - pady1)
            y2 = min(image.shape[0], rect[3] + pady2)
            x1 = max(0, rect[0] - padx1)
            x2 = min(image.shape[1], rect[2] + padx2)

            results.append([x1, y1, x2, y2])

        boxes = np.array(results)
        if not nosmooth:
            boxes = get_smoothened_boxes(boxes, T=5)

        face_imgs = []
        coords = []
        for image, (x1, y1, x2, y2) in zip(images, boxes):
            face_imgs.append(image[y1:y2, x1:x2])
            coords.append((y1, y2, x1, x2))

        # Save to cache
        if video_hash is not None:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            np.savez_compressed(
            cache_path,
            face_imgs=np.array(face_imgs, dtype=object),   # <- add dtype=object
            coords   =np.array(coords,    dtype=object)    # <- add dtype=object
            )
        print(f"💾 Saved face detection cache to: {cache_path}")

        return list(zip(face_imgs, coords))

    except Exception as e:
        print(f"💥 Error during face alignment: {e}")
        raise

def datagen(frames, mels): 
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
    print(f"🔍 Running face detection on {len(frames)} frames...")
    try:
        face_det_results = face_detect(frames)
        print("face detection succeeded...")
    except Exception as e:
        print("❌ Face detection failed:", e)
        raise

    for i, m in enumerate(mels):
        idx = 0 if len(frames) == 1 else i % len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx]
        face = face.copy()
        coords = coords.copy()
        face = cv2.resize(face, (96, 96))
            
        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= 128:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, img_masked.shape[1]//2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, img_masked.shape[1]//2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch

def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint

def load_model(path):
    model = Wav2Lip()
    print("Loading Wav2Lip checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        new_s[k.replace('module.', '')] = v
    model.load_state_dict(new_s)

    model = model.to(device)
    return model.eval()

# 🔵 Only parse args if running manually
if __name__ == "__main__":
    args = parser.parse_args()
    args.img_size = 96
    print("✅ Args parsed successfully")
