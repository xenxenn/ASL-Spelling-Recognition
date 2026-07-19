import json
import os
import cv2
import mediapipe as mp
import numpy as np
import pickle
from collections import defaultdict
from tqdm import tqdm

print("=" * 60)
print("WLASL DATASET PROCESSOR")
print("=" * 60)

# Path files
json_file = 'WLASL_v0.3.json'
videos_dir = 'videos'
output_dir = './dynamic_data'  # Langsung ke dynamic_data

if not os.path.exists(json_file):
    print(f"❌ File {json_file} tidak ditemukan!")
    exit()

if not os.path.exists(videos_dir):
    print(f"❌ Folder {videos_dir} tidak ditemukan!")
    exit()

os.makedirs(output_dir, exist_ok=True)

# Load JSON
print(f"\nLoading {json_file}...")
with open(json_file, 'r') as f:
    wlasl_data = json.load(f)

print(f"✓ Loaded {len(wlasl_data)} words")

# Cek video yang tersedia
available_videos = set(os.listdir(videos_dir))
print(f"✓ Found {len(available_videos)} video files")

# Mapping gloss ke videos
gloss_to_videos = defaultdict(list)

print("\nMapping videos to words...")
for entry in wlasl_data:
    gloss = entry.get('gloss', '').lower()
    instances = entry.get('instances', [])
    
    for inst in instances:
        video_id = inst.get('video_id', '')
        # Cek berbagai format nama file
        possible_names = [
            f"{video_id}.mp4",
            f"{video_id}.avi",
            f"{video_id}.mov",
            f"{video_id}.mkv"
        ]
        
        for vid_name in possible_names:
            if vid_name in available_videos:
                gloss_to_videos[gloss].append({
                    'filename': vid_name,
                    'video_id': video_id,
                    'bbox': inst.get('bbox', None),
                    'frame_start': inst.get('frame_start', 0),
                    'frame_end': inst.get('frame_end', -1)
                })
                break

# Statistik
print("\n" + "=" * 60)
print("AVAILABLE WORDS WITH VIDEOS")
print("=" * 60)

sorted_glosses = sorted(gloss_to_videos.items(), key=lambda x: len(x[1]), reverse=True)
print(f"\nTotal words dengan video: {len(sorted_glosses)}")
print("\nTop 30 words:")
for i, (gloss, videos) in enumerate(sorted_glosses[:30], 1):
    print(f"{i:2d}. {gloss:20s}: {len(videos):3d} videos")

# Pilih kata
print("\n" + "=" * 60)
print("PILIH KATA UNTUK DIPROSES")
print("=" * 60)
print("Contoh: book,drink,computer,house")
print("Atau: TOP20 untuk 20 kata teratas")

selected_input = input("\nMasukkan pilihan: ").strip().lower()

if selected_input.startswith("top"):
    try:
        n = int(selected_input[3:])
        selected_words = [g for g, _ in sorted_glosses[:n]]
    except:
        selected_words = [g for g, _ in sorted_glosses[:20]]
else:
    selected_words = [w.strip() for w in selected_input.split(',')]

# Filter valid
valid_words = [w for w in selected_words if w in gloss_to_videos]

if not valid_words:
    print("❌ Tidak ada kata valid!")
    exit()

print(f"\n✓ Akan memproses {len(valid_words)} kata")

# Limit videos per word
max_videos_per_word = int(input("Max videos per kata (contoh: 30): ").strip() or "30")

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)

SEQUENCE_LENGTH = 30  # 30 frame per gesture

# Process videos
print("\n" + "=" * 60)
print("PROCESSING VIDEOS")
print("=" * 60)

total_processed = 0
total_failed = 0

for word_idx, gloss in enumerate(valid_words, 1):
    videos = gloss_to_videos[gloss][:max_videos_per_word]
    
    print(f"\n[{word_idx}/{len(valid_words)}] Processing: {gloss} ({len(videos)} videos)")
    
    word_dir = os.path.join(output_dir, gloss)
    os.makedirs(word_dir, exist_ok=True)
    
    # Cek existing files
    existing_files = [f for f in os.listdir(word_dir) if f.endswith('.pkl')]
    success_count = len(existing_files)
    
    for vid_idx, video_info in enumerate(tqdm(videos, desc=f"  {gloss}")):
        video_path = os.path.join(videos_dir, video_info['filename'])
        
        if not os.path.exists(video_path):
            total_failed += 1
            continue
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            frame_start = video_info.get('frame_start', 0)
            frame_end = video_info.get('frame_end', -1)
            
            # Skip ke frame_start
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
            
            sequence_data = []
            frame_count = 0
            max_frames = SEQUENCE_LENGTH * 2  # Ambil max 60 frame
            
            while cap.isOpened() and frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Check frame_end
                current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if frame_end > 0 and current_frame > frame_end:
                    break
                
                # Process frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)
                
                if results.multi_hand_landmarks:
                    frame_landmarks = []
                    
                    for hand_landmarks in results.multi_hand_landmarks:
                        hand_data = []
                        for lm in hand_landmarks.landmark:
                            hand_data.extend([lm.x, lm.y, lm.z])
                        frame_landmarks.append(hand_data)
                    
                    # Padding untuk 2 tangan
                    while len(frame_landmarks) < 2:
                        frame_landmarks.append([0] * 63)
                    
                    sequence_data.append(frame_landmarks)
                    frame_count += 1
            
            cap.release()
            
            # Simpan jika ada data cukup
            if len(sequence_data) >= SEQUENCE_LENGTH:
                # Ambil 30 frame (resample)
                indices = np.linspace(0, len(sequence_data) - 1, SEQUENCE_LENGTH, dtype=int)
                resampled = [sequence_data[i] for i in indices]
                
                output_file = os.path.join(word_dir, f"seq_{success_count}.pkl")
                with open(output_file, 'wb') as f:
                    pickle.dump(np.array(resampled), f)
                
                success_count += 1
                total_processed += 1
            else:
                total_failed += 1
        
        except Exception as e:
            total_failed += 1
            continue
    
    print(f"  ✓ {gloss}: {success_count} sequences saved")

print("\n" + "=" * 60)
print("PROCESSING COMPLETED")
print("=" * 60)
print(f"Total berhasil: {total_processed} sequences")
print(f"Total gagal: {total_failed} videos")
print(f"\nData tersimpan di: {output_dir}/")

print("\n" + "=" * 60)
print("NEXT STEPS:")
print("=" * 60)
print("1. python train_dynamic.py")
print("2. python unified_asl_system.py")
print("=" * 60)