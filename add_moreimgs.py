import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Semua label yang tersedia
all_labels = [
    "A","B","C","D","E","F","G","H","I","J","K","L","M",
    "N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
    "1","2","3","4","5","6","7","8","9",
    "SPACE","DELETE"
]

print("=" * 60)
print("TAMBAH SAMPLE DATA TRAINING")
print("=" * 60)
print("\nLabel yang tersedia:")
for i, label in enumerate(all_labels, 1):
    existing_count = 0
    class_dir = os.path.join(DATA_DIR, label)
    if os.path.exists(class_dir):
        existing_count = len([f for f in os.listdir(class_dir) if f.endswith('.jpg')])
    print(f"{i:2d}. {label:6s} - {existing_count} gambar existing")

print("\n" + "=" * 60)
print("Pilih label yang ingin ditambah samplenya")
print("Contoh: A,N,DELETE  atau  1,2,3  atau  ALL")
print("=" * 60)

selected_input = input("\nMasukkan label (pisahkan dengan koma): ").strip().upper()

# Parse input
if selected_input == "ALL":
    labels_to_collect = all_labels
else:
    labels_to_collect = [l.strip() for l in selected_input.split(',') if l.strip() in all_labels]

if not labels_to_collect:
    print("Tidak ada label valid! Program berhenti.")
    exit()

# Jumlah sample tambahan
print(f"\nLabel yang akan ditambah: {', '.join(labels_to_collect)}")
additional_samples = int(input("Berapa sample tambahan per label? (contoh: 100): "))

cap = cv2.VideoCapture(0)

for label in labels_to_collect:
    class_dir = os.path.join(DATA_DIR, label)
    os.makedirs(class_dir, exist_ok=True)
    
    # Cek jumlah existing images
    existing_files = [f for f in os.listdir(class_dir) if f.endswith('.jpg')]
    if existing_files:
        # Ambil nomor terakhir
        existing_numbers = [int(f.split('.')[0]) for f in existing_files if f.split('.')[0].isdigit()]
        start_counter = max(existing_numbers) + 1 if existing_numbers else 0
    else:
        start_counter = 0
    
    print("\n" + "=" * 60)
    print(f"Label: {label}")
    print(f"Existing samples: {len(existing_files)}")
    print(f"Akan menambah: {additional_samples} sample")
    print(f"Mulai dari nomor: {start_counter}")
    print("=" * 60)
    print("Press 'Q' to start collecting...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Flip agar tidak kebalik
        frame = cv2.flip(frame, 1)
        
        # Tampilan info
        cv2.putText(frame, f'Ready for {label}?', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(frame, f'Existing: {len(existing_files)} | Adding: {additional_samples}', (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, 'Press Q to start', (30, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.imshow("Add More Samples", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    counter = start_counter
    end_counter = start_counter + additional_samples
    
    print(f"Collecting {label}... ({counter} to {end_counter-1})")
    
    while counter < end_counter:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Flip agar tidak kebalik
        frame = cv2.flip(frame, 1)
        
        # Progress indicator
        progress = counter - start_counter + 1
        cv2.putText(frame, f'Collecting: {label}', (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.putText(frame, f'Progress: {progress}/{additional_samples}', (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        
        # Progress bar
        bar_width = 500
        filled = int((progress / additional_samples) * bar_width)
        cv2.rectangle(frame, (30, 120), (30 + bar_width, 150), (100, 100, 100), 2)
        cv2.rectangle(frame, (30, 120), (30 + filled, 150), (0, 255, 0), -1)
        
        cv2.imshow("Add More Samples", frame)
        cv2.waitKey(1)
        
        # Save image
        cv2.imwrite(os.path.join(class_dir, f"{counter}.jpg"), frame)
        counter += 1
    
    print(f"✓ Selesai! Total {label}: {counter} gambar")

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("SELESAI!")
print("=" * 60)
print("\nLangkah selanjutnya:")
print("1. python create_dataset.py  (re-create dataset)")
print("2. python train_classifier.py  (train ulang model)")
print("3. python inference_classifier.py  (test model baru)")
print("=" * 60)