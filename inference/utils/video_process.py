import os
import cv2  
import time
import base64
import random
import numpy as np
# import google.generativeai as genai
from decord import VideoReader, cpu
import io
import av
import hashlib
import requests
import numpy.typing as npt
import json
from PIL import Image


def prepare_base64frames(model_name, video_path, total_frames, video_tmp_dir,  video_read_type):
    if video_read_type == "decord":
        frames = _read_with_decord(
            video_path,
            num_frames=total_frames,
        )
    else:
        frames = _read_with_av(
            video_path,
            num_frames=total_frames,
        )

    base64frames = []
    for i, frame in enumerate(frames):
        frame = cv2.resize(frame, (224, 224))
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        base64frames.append(frame_base64)
        
    return base64frames

def prepare_base64_video(video_path):
    video_base = base64.b64encode(open(video_path, "rb").read()).decode('utf-8')

    return video_base


def read_video(video_path: str, total_frames: int):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise ValueError("Could not open video file.")
    try:
        base64_frames = []
        while True:
            success, frame = video.read()
            if not success:
                break 
            _, buffer = cv2.imencode('.jpg', frame)
            
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            base64_frames.append(frame_base64)

        random.seed(42)
        if total_frames == 1:
            selected_indices = [np.random.choice(range(total_frames))]
        else:
            selected_indices = np.linspace(0, len(base64_frames) - 1, total_frames, dtype=int)

        selected_base64_frames = [base64_frames[index] for index in selected_indices]

        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)
        duration = frame_count / fps if fps else 0

        return selected_base64_frames, duration
    finally:
        video.release()

def get_duration(video_path: str):
    video = cv2.VideoCapture(video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    duration = int(frame_count / fps)
    return duration

def sample_frames_from_video(frames: npt.NDArray,
                             num_frames: int) -> npt.NDArray:
    total_frames = frames.shape[0]
    num_frames = min(num_frames, total_frames)

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    sampled_frames = frames[frame_indices, ...]
    return sampled_frames


def video_to_ndarrays(
    path: str, 
    num_frames: int = -1, 
    video_read_type: str = "decord"
) -> npt.NDArray:
    assert video_read_type in ["decord", "av"], "video_read_type should be 'decord' or 'av'"
    
    try:
        if video_read_type == "decord":
            return _read_with_decord(path, num_frames)
        else:
            return _read_with_av(path, num_frames)
    except Exception as e:
        raise ValueError(f"Failed to read video {path} with {video_read_type}: {str(e)}")


def _read_with_decord(path: str, num_frames: int) -> np.ndarray:
    video_reader = VideoReader(path, ctx=cpu(0), num_threads=1)
    vlen = len(video_reader)
    if vlen == 0:
        raise ValueError(f"Empty video: {path}")

    if num_frames == -1:
        fps = video_reader.get_avg_fps()
        interval = max(1, int(fps))
        frame_indices = list(range(0, vlen, interval))
    else:
        frame_indices = get_middle_frame_indices(vlen, num_frames)

    frames = []
    
    for idx in frame_indices:
        try:
            frame = video_reader[idx].asnumpy()
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
        except Exception as e:
            print(f"Warning: Failed to read frame {idx}: {str(e)}")
    
    if not frames:
        raise ValueError(f"No valid frames read from {path}")
    
    frames = np.stack(frames)
    
    if num_frames != -1 and len(frames) < num_frames:
        raise ValueError(
            f"Insufficient frames after sampling: expected {num_frames}, "
            f"got {len(frames)} from {path}"
        )
    
    return frames


def _read_with_av(path: str, num_frames: int) -> np.ndarray:
    container = av.open(path)

    stream = container.streams.video[0]
    vlen = stream.frames
    
    if num_frames == -1:
        fps = float(stream.average_rate)
        interval = max(1, int(fps))
        frame_indices = list(range(0, vlen, interval))
    else:
        frame_indices = get_middle_frame_indices(vlen, num_frames)
    frames = []
    
    for idx, frame in enumerate(container.decode(stream)):
        if idx in frame_indices:
            frame = frame.reformat(format='rgb24').to_ndarray()
            frame = cv2.resize(frame, (224, 224))
            frames.append(frame)
    
    return np.stack(frames)

def get_middle_frame_indices(total_frames: int, num_samples: int) -> list:
    if num_samples >= total_frames:
        return list(range(total_frames))
    
    intervals = np.linspace(0, total_frames, num=num_samples + 1, dtype=int)
    indices = []
    for i in range(num_samples):
        start = intervals[i]
        end = intervals[i+1]
        indices.append((start + end) // 2)
    
    return indices

def video_to_ndarrays_fps(path: str, fps = 1, max_frames = 64) -> npt.NDArray:
    video_name = os.path.basename(path).split('.')[0]
    save_dir = os.path.join(os.path.dirname(path), f"{fps}fps_{max_frames}frames")
    npy_path = os.path.join(save_dir, "frames.npy")
    
    if os.path.exists(npy_path):
        return np.load(npy_path)
    else:
        num_frames = fps * get_duration(path)
        num_frames = int(min(num_frames, max_frames))

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file {path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        for i in range(total_frames):
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        cap.release()

        frames = np.stack(frames)
        frames = sample_frames_from_video(frames, num_frames)
        os.makedirs(save_dir, exist_ok=True)
        for i, frame in enumerate(frames):
            frame_path = os.path.join(save_dir, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)

        np.save(npy_path, frames)
        return frames

def load_video_frames(video_path, video_read_type='decord', total_frames=128):
    """Load video frames and convert to PIL Images"""
    frames = []
    
    # Try to read video using decord first, fallback to av if it fails
    try:
        if video_read_type == 'decord':
            print(f"Attempting to read video using decord: {video_path}")
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            max_frame = len(vr) - 1
            fps = float(vr.get_avg_fps())
            
            # Sample frames uniformly
            frame_indices = get_index(None, fps, max_frame, first_idx=0, num_segments=total_frames)
            
            for frame_index in frame_indices:
                if frame_index < len(vr):
                    img = Image.fromarray(vr[frame_index].asnumpy()).convert("RGB")
                    # Resize to 224x224
                    img = img.resize((224, 224), Image.BILINEAR)
                    frames.append(img)
            
            print(f"Decord reading successful, obtained {len(frames)} frames")
            
        elif video_read_type == 'av':
            print(f"Reading video using av: {video_path}")
            container = av.open(video_path)
            stream = container.streams.video[0]
            fps = float(stream.average_rate)
            max_frame = int(stream.duration * stream.time_base * fps)
            
            frame_indices = set(get_index(None, fps, max_frame, first_idx=0, num_segments=total_frames))
            
            for i, frame in enumerate(container.decode(stream)):
                if i > max(frame_indices):
                    break
                if i in frame_indices:
                    img = frame.to_image().convert("RGB")
                    # Resize to 224x224
                    img = img.resize((224, 224), Image.BILINEAR)
                    frames.append(img)
            
            print(f"AV reading successful, obtained {len(frames)} frames")
            
        else:
            raise ValueError(f"Unsupported video_read_type: {video_read_type}")
            
    except Exception as e:
        print(f"Failed to read video using {video_read_type}: {e}")
        
        # If decord fails, try using av as backup
        if video_read_type == 'decord':
            print("Trying av as backup method...")
            try:
                container = av.open(video_path)
                stream = container.streams.video[0]
                fps = float(stream.average_rate)
                max_frame = int(stream.duration * stream.time_base * fps)
                
                frame_indices = set(get_index(None, fps, max_frame, first_idx=0, num_segments=total_frames))
                
                for i, frame in enumerate(container.decode(stream)):
                    if i > max(frame_indices):
                        break
                    if i in frame_indices:
                        img = frame.to_image().convert("RGB")
                        # Resize to 224x224
                        img = img.resize((224, 224), Image.BILINEAR)
                        frames.append(img)
                
                print(f"AV backup method successful, obtained {len(frames)} frames")
                
            except Exception as backup_e:
                print(f"AV backup method also failed: {backup_e}")
                raise backup_e
        else:
            raise e
    
    return frames

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([int(start_idx + (seg_size / 2) + np.round(seg_size * idx)) for idx in range(num_segments)])
    return frame_indices