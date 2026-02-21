import cv2
import mediapipe as mp
import time
from typing import Optional, Tuple

from core.interfaces import IPoseEngine
from core.types import HandSkeleton, FaceSkeleton, Point

# MediaPipe Configuration
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Hand Tracking
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult

# Face Tracking
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult

class MediaPipeEngine(IPoseEngine):
    def __init__(self, 
                 hand_model_path: str = 'mediapipe_model/gesture_recognizer.task',
                 face_model_path: str = 'mediapipe_model/face_landmarker.task'):
        self.hand_model_path = hand_model_path
        self.face_model_path = face_model_path
        
        self.recognizer = None
        self.face_landmarker = None
        
        self.latest_hand_result: Optional[GestureRecognizerResult] = None
        self.latest_face_result: Optional[FaceLandmarkerResult] = None
        
        # Initialize immediately
        self.initialize()

    def initialize(self) -> None:
        """Sets up the MediaPipe Gesture Recognizer and Face Landmarker."""
        # Initialize Hand Recognizer
        hand_options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=self.hand_model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self._hand_result_callback
        )
        self.recognizer = GestureRecognizer.create_from_options(hand_options)
        
        # Initialize Face Landmarker
        face_options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.face_model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self._face_result_callback,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.face_landmarker = FaceLandmarker.create_from_options(face_options)
        
        print(f"MediaPipe Engine initialized with models: {self.hand_model_path}, {self.face_model_path}")

    def _hand_result_callback(self, result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
        """Callback invoked by MediaPipe when a hand result is available."""
        self.latest_hand_result = result

    def _face_result_callback(self, result: FaceLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        """Callback invoked by MediaPipe when a face result is available."""
        self.latest_face_result = result

    def process_frame(self, frame, timestamp: int) -> Tuple[Optional[HandSkeleton], Optional[FaceSkeleton]]:
        """
        Processes a frame and returns standardized HandSkeleton and FaceSkeleton.
        """
        # MediaPipe expects RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Async calls - results will be updated in callbacks
        self.recognizer.recognize_async(mp_image, int(timestamp))
        self.face_landmarker.detect_async(mp_image, int(timestamp))
        
        # Return the latest available results
        hand_skeleton = self._convert_hand_result_to_skeleton(self.latest_hand_result, timestamp)
        face_skeleton = self._convert_face_result_to_skeleton(self.latest_face_result, timestamp)
        
        return hand_skeleton, face_skeleton

    def _convert_hand_result_to_skeleton(self, result: GestureRecognizerResult, timestamp: int) -> Optional[HandSkeleton]:
        if not result or not result.hand_landmarks:
            return None
            
        # We only take the first hand for now
        landmarks = result.hand_landmarks[0]
        
        # Convert MediaPipe landmarks to our Point class
        points = [Point(x=lm.x, y=lm.y, z=lm.z) for lm in landmarks]
        
        # Confidence extraction
        confidence = 0.0
        gesture_name = None
        if result.gestures and result.gestures[0]:
            confidence = result.gestures[0][0].score
            gesture_name = result.gestures[0][0].category_name

        return HandSkeleton(
            landmarks=points,
            confidence=confidence,
            timestamp=timestamp,
            gesture=gesture_name
        )

    def _convert_face_result_to_skeleton(self, result: FaceLandmarkerResult, timestamp: int) -> Optional[FaceSkeleton]:
        if not result or not result.face_landmarks:
            return None
            
        # We only take the first face for now
        landmarks = result.face_landmarks[0]
        
        # Convert MediaPipe landmarks to our Point class
        points = [Point(x=lm.x, y=lm.y, z=lm.z) for lm in landmarks]
        
        return FaceSkeleton(
            landmarks=points,
            timestamp=timestamp
        )

    def release(self) -> None:
        if self.recognizer:
            self.recognizer.close()
        if self.face_landmarker:
            self.face_landmarker.close()
