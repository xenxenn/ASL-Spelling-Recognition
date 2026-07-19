import pickle
import cv2
import mediapipe as mp
import numpy as np
from collections import deque

print("=" * 60)
print("LOADING MODELS...")
print("=" * 60)

# Load Static Model (Random Forest)
print("Loading static model (Random Forest)...")
static_model = pickle.load(open('./model.p', 'rb'))['model']
print("✓ Static model loaded")

# Load Dynamic Model (LSTM)
print("Loading dynamic model (LSTM)...")
try:
    from tensorflow import keras
    dynamic_model = keras.models.load_model('dynamic_model.h5')
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    print("✓ Dynamic model loaded")
    has_dynamic = True
except Exception as e:
    print(f"⚠ Dynamic model not found: {e}")
    print("Only SPELLING mode available.")
    has_dynamic = False

# Setup MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Variables
captured_text = ""
current_mode = "SPELLING"  # SPELLING or WORDS
SEQUENCE_LENGTH = 30
frame_buffer = deque(maxlen=SEQUENCE_LENGTH)

# Static mode variables
last_static_gesture = None
static_stable_count = 0
STATIC_STABLE_FRAMES = 60  # 2 detik
static_capture_effect = 0

# Dynamic mode variables
is_recording = False
dynamic_prediction = "None"
dynamic_confidence = 0

print("\n" + "=" * 60)
print("UNIFIED ASL RECOGNITION SYSTEM")
print("=" * 60)
print("MODES:")
print("  SPELLING - Detect letters/numbers (A-Z, 0-9)")
if has_dynamic:
    print("  WORDS    - Detect word gestures (HELLO, BOOK, etc)")
print("\nCONTROLS:")
print("  [1] - Switch to SPELLING mode")
if has_dynamic:
    print("  [2] - Switch to WORDS mode")
print("  [SPACE] - Manual capture (SPELLING mode)")
if has_dynamic:
    print("  [R] - Start/Stop recording (WORDS mode)")
print("  [C] - Clear text")
print("  [BACKSPACE] - Delete last character")
print("  [Q] - Quit")
print("=" * 60)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    H, W, _ = frame.shape
    x_, y_, data_aux = [], [], []
    
    predicted_label = None
    current_hand = "None"
    confidence = 0

    # ========== SPELLING MODE (STATIC) ==========
    if current_mode == "SPELLING":
        if results.multi_hand_landmarks:
            if results.multi_handedness:
                current_hand = results.multi_handedness[0].classification[0].label
            
            # Draw landmarks
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
                )
            
            # Extract features (STATIC)
            for lm in results.multi_hand_landmarks[0].landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            for lm in results.multi_hand_landmarks[0].landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            predicted_label = static_model.predict([np.asarray(data_aux)])[0]
            
            # Get confidence
            try:
                probabilities = static_model.predict_proba([np.asarray(data_aux)])[0]
                confidence = max(probabilities) * 100
            except:
                confidence = 95
            
            # Auto-capture logic
            if predicted_label == last_static_gesture:
                static_stable_count += 1
            else:
                static_stable_count = 0
                last_static_gesture = predicted_label
            
            if static_stable_count == STATIC_STABLE_FRAMES:
                if predicted_label == "SPACE":
                    captured_text += " "
                elif predicted_label == "DELETE":
                    captured_text = captured_text[:-1]
                else:
                    captured_text += predicted_label
                static_capture_effect = 15
                static_stable_count = 0
            
            # Draw box around hand
            x_min = int(min(x_) * W) - 30
            y_min = int(min(y_) * H) - 50
            x_max = int(max(x_) * W) + 30
            y_max = int(max(y_) * H) + 30
            
            box_color = (0, 255, 255) if static_stable_count > STATIC_STABLE_FRAMES * 0.7 else (0, 255, 0)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), box_color, 3)
            
            # Progress bar
            if static_stable_count > 0:
                progress_width = int((static_stable_count / STATIC_STABLE_FRAMES) * (x_max - x_min))
                cv2.rectangle(frame, (x_min, y_max + 5), (x_min + progress_width, y_max + 15), (0, 255, 255), -1)
            
            # Label
            label_text = f"{predicted_label} ({confidence:.0f}%)"
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            text_y = y_min - 15
            if text_y < text_size[1] + 10:
                text_y = y_max + text_size[1] + 25
            
            cv2.rectangle(frame, (x_min - 5, text_y - text_size[1] - 10),
                         (x_min + text_size[0] + 10, text_y + 5), box_color, -1)
            cv2.putText(frame, label_text, (x_min, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
        else:
            static_stable_count = 0
            last_static_gesture = None
        
        # Flash effect
        if static_capture_effect > 0:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (W, H), (255, 255, 255), -1)
            cv2.addWeighted(overlay, static_capture_effect / 30, frame, 1 - static_capture_effect / 30, 0, frame)
            static_capture_effect -= 1

    # ========== WORDS MODE (DYNAMIC) ==========
    elif current_mode == "WORDS" and has_dynamic:
        # Extract landmarks for both hands
        frame_landmarks = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)
                )
                
                hand_data = []
                for lm in hand_landmarks.landmark:
                    hand_data.extend([lm.x, lm.y, lm.z])
                frame_landmarks.append(hand_data)
        
        # Padding
        while len(frame_landmarks) < 2:
            frame_landmarks.append([0] * 63)
        
        frame_flat = []
        for hand in frame_landmarks:
            frame_flat.extend(hand)
        
        # Recording logic
        if is_recording:
            frame_buffer.append(frame_flat)
            
            if len(frame_buffer) == SEQUENCE_LENGTH:
                sequence = np.array([list(frame_buffer)])
                prediction = dynamic_model.predict(sequence, verbose=0)
                predicted_class = np.argmax(prediction[0])
                dynamic_confidence = prediction[0][predicted_class] * 100
                dynamic_prediction = label_encoder.inverse_transform([predicted_class])[0]
                
                if dynamic_confidence > 70:
                    captured_text += dynamic_prediction.upper() + " "
                    frame_buffer.clear()
                    is_recording = False
    
    # ========== UI PANEL ATAS ==========
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (W, 130), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    # Title
    cv2.putText(frame, "UNIFIED ASL RECOGNITION", (20, 40), 
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 3)
    
    # Mode indicator
    mode_color = (0, 255, 0) if current_mode == "SPELLING" else (255, 0, 255)
    mode_text = f"MODE: {current_mode}"
    cv2.putText(frame, mode_text, (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, mode_color, 2)
    
    # Status
    if current_mode == "SPELLING":
        status_text = f"Hand: {current_hand} | Gesture: {predicted_label if predicted_label else 'None'}"
        if static_stable_count > 0:
            status_text += f" | Stability: {int((static_stable_count/STATIC_STABLE_FRAMES)*100)}%"
    else:
        rec_status = "🔴 RECORDING" if is_recording else "⚪ READY"
        status_text = rec_status
        if is_recording:
            status_text += f" | Buffer: {int((len(frame_buffer)/SEQUENCE_LENGTH)*100)}%"
        if dynamic_prediction != "None":
            status_text += f" | Last: {dynamic_prediction} ({dynamic_confidence:.0f}%)"
    
    cv2.putText(frame, status_text, (20, 115), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
    # ========== UI PANEL BAWAH ==========
    cv2.rectangle(overlay, (0, H-100), (W, H), (40, 40, 40), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    cv2.putText(frame, "Text:", (20, H-60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    
    text_display = captured_text if captured_text else "(empty)"
    text_color = (0, 255, 255) if captured_text else (100, 100, 100)
    
    # Truncate jika terlalu panjang
    max_chars = 80
    if len(text_display) > max_chars:
        text_display = "..." + text_display[-(max_chars-3):]
    
    cv2.putText(frame, text_display, (120, H-60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
    
    # Controls
    if has_dynamic:
        controls = "[1] Spell | [2] Words | [SPACE/R] Capture | [C] Clear | [BACKSPACE] Del | [Q] Quit"
    else:
        controls = "[1] Spell | [SPACE] Capture | [C] Clear | [BACKSPACE] Del | [Q] Quit"
    
    cv2.putText(frame, controls, (20, H-20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    
    cv2.imshow("Unified ASL Recognition System", frame)
    
    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        break
    elif key == ord('1'):
        current_mode = "SPELLING"
        frame_buffer.clear()
        is_recording = False
    elif key == ord('2'):
        if has_dynamic:
            current_mode = "WORDS"
            static_stable_count = 0
        else:
            print("⚠ Dynamic model not available!")
    elif key == ord(' ') and current_mode == "SPELLING" and predicted_label:
        if predicted_label == "SPACE":
            captured_text += " "
        elif predicted_label == "DELETE":
            captured_text = captured_text[:-1]
        else:
            captured_text += predicted_label
    elif key == ord('r') and current_mode == "WORDS" and has_dynamic:
        is_recording = not is_recording
        if is_recording:
            frame_buffer.clear()
    elif key == ord('c'):
        captured_text = ""
    elif key == 8:  # BACKSPACE
        captured_text = captured_text[:-1]

cap.release()
cv2.destroyAllWindows()

print("\n✓ Program ended")
print(f"Final text: {captured_text}")