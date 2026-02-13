# 

import cv2
import mediapipe as mp
import numpy as np
import os

# --- НАСТРОЙКИ ---
VIDEOS_PATH = 'raw_videos'   # Папка с видео
OUTPUT_X = 'X_data.npy'      # Куда сохраним координаты
OUTPUT_Y = 'y_data.npy'      # Куда сохраним названия букв
CLASSES_FILE = 'classes.npy' # Список классов (А, Б, В...)

mp_hands = mp.solutions.hands

def extract_keypoints(results):
    """Превращает результаты Mediapipe в массив из 126 чисел (2 руки)"""
    lh = np.zeros(21*3) # 63 нуля для левой
    rh = np.zeros(21*3) # 63 нуля для правой
    
    if results.multi_hand_landmarks:
        for idx, hand_handedness in enumerate(results.multi_handedness):
            # Определяем, какая это рука (Left/Right)
            label = hand_handedness.classification[0].label
            landmarks = results.multi_hand_landmarks[idx].landmark
            
            # Превращаем точки в плоский массив [x, y, z, x, y, z...]
            hand_data = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
            
            if label == 'Left':
                lh = hand_data
            else: # Right
                rh = hand_data
                
    # Склеиваем: сначала Левая (0-62), потом Правая (63-125)
    return np.concatenate([lh, rh])

def process_all_videos():
    X, y = [], []
   
    classes = sorted([d for d in os.listdir(VIDEOS_PATH) if os.path.isdir(os.path.join(VIDEOS_PATH, d))])
  
    np.save('classes.npy', classes)
    print(f"Список букв зафиксирован: {classes}")
   
    with mp_hands.Hands(
        static_image_mode=False, 
        max_num_hands=2,         
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        
        for idx, action_name in enumerate(classes):
            action_path = os.path.join(VIDEOS_PATH, action_name)
            if not os.path.isdir(action_path): continue
            
            video_files = os.listdir(action_path)
            print(f"Обработка жеста '{action_name}': найдено {len(video_files)} видео ")
            
            for video_name in video_files:
                video_path = os.path.join(action_path, video_name)
                cap = cv2.VideoCapture(video_path)
                
                frame_count = 0
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    # обработка кадра
                    # frame = cv2.flip(frame, 1) # для фронтальной камеры, если нужно
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image.flags.writeable = False
                    results = hands.process(image)
                    
                    if results.multi_hand_landmarks:
                        keypoints = extract_keypoints(results)
                        X.append(keypoints)
                        y.append(idx) 
                        frame_count += 1
                
                cap.release()
                print(f"   -> {video_name}: извлечено {frame_count} кадров")

    print(f"\nГОТОВО! Кол-во кадров: {len(X)}")
  
    
    np.save(OUTPUT_X, np.array(X))
    np.save(OUTPUT_Y, np.array(y))
    np.save(CLASSES_FILE, classes)
    print(f"Файлы сохранены: {OUTPUT_X}, {OUTPUT_Y}, {CLASSES_FILE}")

if __name__ == "__main__":
    process_all_videos()