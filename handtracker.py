import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mphands = mp.solutions.hands

cap = cv2.VideoCapture(0)
hands = mphands.Hands()

while True:
    data, image = cap.read()
    # flip the image
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            index_finger_tip = hand_landmarks.landmark[mphands.HandLandmark.INDEX_FINGER_TIP]
            # Convert normalized coordinates to pixel coordinates
            h, w, _ = image.shape
            x = int(index_finger_tip.x * w)
            y = int(index_finger_tip.y * h)

            # Draw a circle on the fingertip
            cv2.circle(image, (x, y), 10, (0, 255, 0), -1)

            # (Optional) Print coordinates
            print(f"Index fingertip position: x={x}, y={y}")
    cv2.imshow('indextracker', image)
    cv2.waitKey(1)
