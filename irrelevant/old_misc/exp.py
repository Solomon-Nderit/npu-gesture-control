import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = './mediapipe_model/gesture_recognizer.task'
mp_image = mp.Image.create_from_file('./images/thumbs_up.jpg')

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

base_options = python.BaseOptions(model_asset_path='mediapipe_model/gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)



image = mp.Image.create_from_file("./images/image.png")

result = recognizer.recognize(image)

# def get_position_of_index():

# print([attr for attr in dir(result) if not attr.startswith('__')])
# print(result.hand_landmarks[0][8])

index_finger_tip = result.hand_landmarks[0][8]

print(index_finger_tip.x)