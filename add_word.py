import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Semua label yang ada saat ini
existing_labels = [
    "A","B","C","D","E","F","G","H","I","J","K","L","M",
    "N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
    "1","2","3","4","5","6","7","8","9",
    "SPACE","DELETE","saranghae","ily"
]

print("=" * 60)
print("TAMBAH GESTURE/KALIMAT BARU")
print("=" * 60)
print("\n🔹 Mode 1: Tambah sample untuk gesture yang sudah ada")
print("🔹 Mode 2: Buat gesture/kalimat baru (saranghae, iloveu, dll)")
print("=" * 60)

mode = input("\nPilih mode (1 atau 2): ").strip()

if mode == "1":
    # MODE 1: Tambah sample untuk gesture existing
    print("\n" + "=" * 60)
    print("GESTURE YANG SUDAH ADA:")
    print("=" * 60)
    for i, label in enumerate(existing_labels, 1):
        existing_count = 0
        class_dir = os.path.join(DATA_DIR, label)
        if os.path.exists(class_dir):
            existing_count = len([f for f in os.listdir(class_dir) if f.endswith('.jpg')])
        print(f"{i:2d}. {label:6s} - {existing_count} gambar")
    
    print("\n" + "=" * 60)
    print("Pilih label yang ingin ditambah samplenya")
    print("Contoh: A,N,DELETE  atau  1,2,3  atau  ALL")
    print("=" * 60)
    
    selected_input = input("\nMasukkan label (pisahkan dengan koma): ").strip().upper()
    
    # Parse input
    if selected_input == "ALL":
        labels_to_collect = existing_labels
    else:
        labels_to_collect = [l.strip() for l in selected_input.split(',') if l.strip() in existing_labels]
    
    if not labels_to_collect:
        print("Tidak ada label valid! Program berhenti.")
        exit()

elif mode == "2":
    # MODE 2: Buat gesture/kalimat baru
    print("\n" + "=" * 60)
    print("BUAT GESTURE/KALIMAT BARU")
    print("=" * 60)
    print("Contoh: saranghae, iloveu, hello, thankyou, dll")
    print("(Gunakan huruf kecil/besar, tanpa spasi)")
    print("=" * 60)
    
    new_gestures_input = input("\nMasukkan gesture baru (pisahkan dengan koma): ").strip()
    labels_to_collect = [g.strip() for g in new_gestures_input.split(',') if g.strip()]
    
    if not labels_to_collect:
        print("Tidak ada gesture valid! Program berhenti.")
        exit()
    
    print(f"\n✓ Gesture baru yang akan dibuat: {', '.join(labels_to_collect)}")

else:
    print("Mode tidak valid! Program berhenti.")
    exit()

# Jumlah sample
print(f"\nGesture yang akan dikumpulkan: {', '.join(labels_to_collect)}")
samples_count = int(input("Berapa sample per gesture? (contoh: 200): "))

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
    
    is_new = len(existing_files) == 0
    status = "GESTURE BARU" if is_new else "TAMBAH SAMPLE"
    
    print("\n" + "=" * 60)
    print(f"📌 {status}: {label}")
    print(f"Existing samples: {len(existing_files)}")
    print(f"Akan menambah: {samples_count} sample")
    print(f"Mulai dari nomor: {start_counter}")
    print("=" * 60)
    print("Press 'Q' to start collecting...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Flip agar tidak kebalik
        frame = cv2.flip(frame, 1)
        
        # Background panel
        cv2.rectangle(frame, (0, 0), (640, 200), (40, 40, 40), -1)
        
        # Tampilan info
        status_color = (0, 255, 255) if is_new else (0, 255, 0)
        cv2.putText(frame, f'{status}: {label}', (30, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, status_color, 2)
        cv2.putText(frame, f'Existing: {len(existing_files)} | Will add: {samples_count}', (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, 'Press Q to start collecting', (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow("Collect Gestures", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    counter = start_counter
    end_counter = start_counter + samples_count
    
    print(f"🎥 Collecting {label}... ({counter} to {end_counter-1})")
    
    while counter < end_counter:
        ret, frame = cap.read()
        if not ret:
            continue
        
        # Flip agar tidak kebalik
        frame = cv2.flip(frame, 1)
        
        # Progress indicator
        progress = counter - start_counter + 1
        progress_pct = int((progress / samples_count) * 100)
        
        # Background panel
        cv2.rectangle(frame, (0, 0), (640, 180), (40, 40, 40), -1)
        
        cv2.putText(frame, f'Collecting: {label}', (30, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 255, 0), 2)
        cv2.putText(frame, f'{progress}/{samples_count} ({progress_pct}%)', (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        
        # Progress bar
        bar_width = 500
        filled = int((progress / samples_count) * bar_width)
        cv2.rectangle(frame, (30, 110), (30 + bar_width, 140), (100, 100, 100), 2)
        cv2.rectangle(frame, (30, 110), (30 + filled, 140), (0, 255, 0), -1)
        
        cv2.imshow("Collect Gestures", frame)
        cv2.waitKey(1)
        
        # Save image
        cv2.imwrite(os.path.join(class_dir, f"{counter}.jpg"), frame)
        counter += 1
    
    print(f"✓ Selesai! Total {label}: {counter} gambar")

cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 60)
print("🎉 SELESAI!")
print("=" * 60)
print("\nFolder data yang telah dibuat/diupdate:")
for label in labels_to_collect:
    class_dir = os.path.join(DATA_DIR, label)
    count = len([f for f in os.listdir(class_dir) if f.endswith('.jpg')])
    print(f"  ✓ {label}: {count} gambar")

print("\n" + "=" * 60)
print("LANGKAH SELANJUTNYA:")
print("=" * 60)
print("1. python create_dataset.py  (proses semua data)")
print("2. python train_classifier.py  (train model)")
print("3. python inference_classifier.py  (test hasil)")
print("=" * 60)