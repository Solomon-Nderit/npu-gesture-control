import cv2
import time

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


from action_taker import take_action


BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

#Open Default Camera

cam = cv2.VideoCapture(0)


latest_result = None  # Type: GestureRecognizerResult | None

#Create a gesture recognizer instance with live stream mode:
def print_result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    global latest_result
    latest_result = result


options = GestureRecognizerOptions(
    base_options=BaseOptions(model_asset_path='mediapipe_model/gesture_recognizer.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)

recognizer = GestureRecognizer.create_from_options(options)



#Return detection result object
def get_gestures(frame,timestamp):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = recognizer.recognize_async(mp_image, int(timestamp))
    return result


while True:
    ret, frame = cam.read()  # frame is already a numpy.ndarray
    timestamp_ms = cam.get(cv2.CAP_PROP_POS_MSEC)

    # Send for async recognition
    get_gestures(frame, timestamp_ms)
    
    # Draw the latest result if available
    if latest_result and latest_result.hand_landmarks:
        for hand_landmarks in latest_result.hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, 
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        gesture=latest_result.gestures[0][0].category_name

        #Finger tips coordinates
        index_finger_tip = latest_result.hand_landmarks[0][8]
        thumb_tip = latest_result.hand_landmarks[0][4]


        take_action(gesture, index_finger_tip, thumb_tip)

    
    cv2.imshow('Camera', frame)  # Display the frame with drawings
    if cv2.waitKey(1) == ord('q'):
        break

   

#Release capture object
cam.release()
cv2.destroyAllWindows()