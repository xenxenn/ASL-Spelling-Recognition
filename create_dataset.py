import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'
data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    for img_name in os.listdir(os.path.join(DATA_DIR, dir_)):

        img_path = os.path.join(DATA_DIR, dir_, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            x_, y_ = [], []
            data_aux = []

            for lm in results.multi_hand_landmarks[0].landmark:
                x_.append(lm.x)
                y_.append(lm.y)

            for lm in results.multi_hand_landmarks[0].landmark:
                data_aux.append(lm.x - min(x_))
                data_aux.append(lm.y - min(y_))

            data.append(data_aux)
            labels.append(dir_)

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Dataset created!")
