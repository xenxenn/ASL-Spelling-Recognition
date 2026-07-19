import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split

DATA_DIR = './dynamic_data'

print("=" * 60)
print("CREATE DYNAMIC GESTURE DATASET")
print("=" * 60)

sequences = []
labels = []
label_names = []

# Load semua sequence data
for label_dir in os.listdir(DATA_DIR):
    label_path = os.path.join(DATA_DIR, label_dir)
    if not os.path.isdir(label_path):
        continue
    
    print(f"\nProcessing: {label_dir}")
    label_names.append(label_dir)
    
    sequence_files = [f for f in os.listdir(label_path) if f.endswith('.pkl')]
    print(f"  Found {len(sequence_files)} sequences")
    
    for seq_file in sequence_files:
        seq_path = os.path.join(label_path, seq_file)
        try:
            with open(seq_path, 'rb') as f:
                sequence_data = pickle.load(f)
                
                # Flatten: (30 frames, 2 hands, 63 coords) -> (30, 126)
                flattened = []
                for frame in sequence_data:
                    frame_flat = []
                    for hand in frame:
                        frame_flat.extend(hand)
                    flattened.append(frame_flat)
                
                sequences.append(flattened)
                labels.append(label_dir)
        except Exception as e:
            print(f"  ⚠ Error loading {seq_file}: {e}")
            continue
    
    print(f"  ✓ Loaded {len([l for l in labels if l == label_dir])} sequences")

# Convert to numpy arrays
X = np.array(sequences)  # Shape: (num_samples, 30, 126)
y = np.array(labels)

print("\n" + "=" * 60)
print("DATASET SUMMARY")
print("=" * 60)
print(f"Total sequences: {len(X)}")
print(f"Sequence shape: {X.shape}")
print(f"Labels: {len(label_names)}")
print(f"Label distribution:")
for label in label_names:
    count = np.sum(y == label)
    print(f"  {label}: {count} sequences")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\n" + "=" * 60)
print("SPLIT DATA")
print("=" * 60)
print(f"Training: {len(X_train)} sequences")
print(f"Testing: {len(X_test)} sequences")

# Save dataset
dataset = {
    'X_train': X_train,
    'X_test': X_test,
    'y_train': y_train,
    'y_test': y_test,
    'label_names': label_names
}

with open('dynamic_dataset.pkl', 'wb') as f:
    pickle.dump(dataset, f)

print("\n✓ Dataset saved to: dynamic_dataset.pkl")
print("\n" + "=" * 60)
print("NEXT STEP:")
print("python train_dynamic_model.py")
print("=" * 60)