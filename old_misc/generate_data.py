import cv2
import os
import time

# 1. Create the folder
folder_name = "data"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

cap = cv2.VideoCapture(0)
print(f"Starting capture in 3 seconds... Wave your hand!")
time.sleep(3)

for i in range(20):
    ret, frame = cap.read()
    if not ret: break
    
    # Save the raw frame
    filename = f"{folder_name}/calib_{i}.jpg"
    cv2.imwrite(filename, frame)
    print(f"Captured {filename}")
    
    # Small delay so images are slightly different
    time.sleep(0.2) 

cap.release()
print("Done! You now have your calibration data.")