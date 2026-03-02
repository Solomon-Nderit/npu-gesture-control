import cv2
import mediapipe as mp

# 1. Setup MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 'min_detection_confidence' ensures it doesn't jitter too much
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # MediaPipe works with RGB, OpenCV uses BGR
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 2. Run Inference
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the skeleton (21 points)
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 3. Extract Keypoints (Normalized 0.0 to 1.0)
            # Index Finger Tip is point 8
            index_tip = hand_landmarks.landmark[8]
            # Index Finger PIP (Knuckle) is point 6
            index_mcp = hand_landmarks.landmark[6] # Using MCP (base knuckle) for better stability
            
            # 4. The Logic: Is tip HIGHER than knuckle? 
            # Note: In screen coordinates, Y=0 is TOP. So "Smaller Y" means "Higher Up"
            if index_tip.y < index_mcp.y:
                cv2.putText(frame, "GESTURE: Index Up!", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # print("Command: 1") # This is where you'd send your mouse command later
            else:
                cv2.putText(frame, "State: Resting", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Hand Gesture Prototype', frame)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()