import pickle
import cv2
import mediapipe as mp
import numpy as np

model = pickle.load(open('./model.p', 'rb'))['model']
cap = cv2.VideoCapture(0)

# Set resolusi lebih besar
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)

# Variable untuk captured word
captured_word = ""
last_gesture = None
stable_count = 0
STABLE_FRAMES = 30  # 2 detik (30 fps x 2)
capture_effect = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip horizontal biar tidak kebalik
    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    x_, y_, data_aux = [], [], []
    H, W, _ = frame.shape
    
    predicted_label = None
    current_hand = "None"
    confidence = 0

    if results.multi_hand_landmarks:
        # UI: Deteksi tangan kiri/kanan
        if results.multi_handedness:
            current_hand = results.multi_handedness[0].classification[0].label
        
        # UI: Gambar lines/pipes di jari
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )
        
        # === CODE ASLI ANDA - TIDAK DIUBAH ===
        for lm in results.multi_hand_landmarks[0].landmark:
            x_.append(lm.x)
            y_.append(lm.y)

        for lm in results.multi_hand_landmarks[0].landmark:
            data_aux.append(lm.x - min(x_))
            data_aux.append(lm.y - min(y_))

        predicted_label = model.predict([np.asarray(data_aux)])[0]
        # === END CODE ASLI ===
        
        # UI: Dapatkan confidence
        try:
            probabilities = model.predict_proba([np.asarray(data_aux)])[0]
            confidence = max(probabilities) * 100
        except:
            confidence = 0
        
        # Auto-capture ketika stabil 2 detik
        if predicted_label == last_gesture:
            stable_count += 1
        else:
            stable_count = 0
            last_gesture = predicted_label
        
        # Jika sudah stabil 2 detik, auto-capture
        if stable_count == STABLE_FRAMES:
            if predicted_label == "SPACE":
                captured_word += " "
            elif predicted_label == "DELETE":
                captured_word = captured_word[:-1]
            else:
                captured_word += predicted_label
            capture_effect = 15
            stable_count = 0
        
        # UI: Kotak di sekitar tangan
        x_min = int(min(x_) * W) - 30
        y_min = int(min(y_) * H) - 50
        x_max = int(max(x_) * W) + 30
        y_max = int(max(y_) * H) + 30
        
        # Warna kotak berubah kuning saat hampir capture
        box_color = (0, 255, 255) if stable_count > STABLE_FRAMES * 0.7 else (0, 255, 0)
        box_thickness = 4 if stable_count > STABLE_FRAMES * 0.7 else 3
        
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, box_thickness)
        
        # Progress bar stabilitas
        if stable_count > 0:
            progress_width = int((stable_count / STABLE_FRAMES) * (x_max - x_min))
            cv2.rectangle(frame, (x_min, y_max + 5), (x_min + progress_width, y_max + 15), (0, 255, 255), -1)
            # Text percentage
            progress_pct = int((stable_count / STABLE_FRAMES) * 100)
            cv2.putText(frame, f"{progress_pct}%", (x_max + 10, y_max + 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # UI: Label gesture dengan persentase
        label_text = f"{predicted_label} ({confidence:.0f}%)"
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
        text_y = y_min - 15
        if text_y < text_size[1] + 10:
            text_y = y_max + text_size[1] + 25
        
        cv2.rectangle(frame, 
                     (x_min - 5, text_y - text_size[1] - 10),
                     (x_min + text_size[0] + 10, text_y + 5),
                     box_color, -1)
        
        cv2.putText(frame, label_text, (x_min, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    
    # UI: PANEL ATAS
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, 100), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    cv2.putText(frame, "ASL RECOGNITION", (20, 45), 
                cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 3)
    
    hand_color = (100, 200, 255) if current_hand == "Right" else (255, 150, 100) if current_hand == "Left" else (150, 150, 150)
    cv2.putText(frame, f"Hand: {current_hand}", (20, 75), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 2)
    
    gesture_text = predicted_label if predicted_label else "None"
    gesture_color = (0, 255, 0) if predicted_label else (150, 150, 150)
    cv2.putText(frame, f"Gesture: {gesture_text}", (250, 75), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, gesture_color, 2)
    
    # UI: PANEL BAWAH (diperbesar untuk word)
    cv2.rectangle(overlay, (0, H-90), (W, H), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    cv2.putText(frame, "Word:", (20, H-50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    
    word_display = captured_word if captured_word else "(empty)"
    word_color = (0, 255, 255) if captured_word else (100, 100, 100)
    cv2.putText(frame, word_display, (130, H-50), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, word_color, 3)
    
    cv2.putText(frame, "[SPACE] Manual Capture  |  [C] Clear  |  [Q] Quit  |  Auto: Hold 1s", (20, H-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    cv2.imshow("Hand Gesture Recognition Demo", frame)
    
    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' ') and predicted_label:
        # Handle special gestures
        if predicted_label == "SPACE":
            captured_word += " "  # Tambah spasi
        elif predicted_label == "DELETE":
            captured_word = captured_word[:-1]  # Hapus 1 karakter terakhir
        else:
            captured_word += predicted_label  # Tambah huruf/angka
    elif key == ord('c'):
        captured_word = ""

cap.release()
cv2.destroyAllWindows()