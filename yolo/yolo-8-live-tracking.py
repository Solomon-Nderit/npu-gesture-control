import cv2
from ultralytics import YOLO

# 1. Load the official Pose model
model = YOLO('./hand-trained.pt')  # This will download automatically on first run

# 2. Open Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # 3. Run Inference (Fast!)
    # stream=True makes it faster for video
    results = model(frame, stream=True, verbose=False)

    for r in results:
        # Visualize the skeleton on the frame
        frame = r.plot()
        
        # 4. Extract Keypoints for Logic
        # shape is (1, 21, 2) -> (Batch, Keypoints, XY)
        if r.keypoints and r.keypoints.data.shape[1] >= 21:
            keypoints = r.keypoints.data[0] 
            
            # Keypoint indices (Standard MediaPipe/YOLO format):
            # 0: Wrist, 8: Index Tip, 6: Index PIP (Knuckle)
            index_tip_y = keypoints[8][1]
            index_knuckle_y = keypoints[6][1]
            
            # Simple Logic: "Is the tip above the knuckle?" 
            # (Note: In image arrays, Y=0 is the top, so 'less than' means 'higher')
            if index_tip_y < index_knuckle_y: 
                cv2.putText(frame, "GESTURE: Index Up!", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "State: Resting", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('YOLOv8 Pose Control', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()