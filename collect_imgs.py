import os
import cv2

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

labels = [
    "A","B","C","D","E","F","G","H","I","J","K","L","M",
    "N","O","P","Q","R","S","T","U","V","W","X","Y","Z",
    "1","2","3","4","5","6","7","8","9",
    "SPACE","DELETE"
]
dataset_size = 100  # jumlah gambar per label

cap = cv2.VideoCapture(0)

for label in labels:
    class_dir = os.path.join(DATA_DIR, label)
    os.makedirs(class_dir, exist_ok=True)

    print(f"Collecting for label: {label}")
    print("Press 'Q' to start...")
    
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, f'Ready for {label}? Press Q', (30,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow("frame", frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow("frame", frame)
        cv2.waitKey(1)
        cv2.imwrite(os.path.join(class_dir, f"{counter}.jpg"), frame)
        counter += 1

cap.release()
cv2.destroyAllWindows()