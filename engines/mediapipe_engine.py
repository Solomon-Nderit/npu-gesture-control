import cv2
import mediapipe as mp
import time
from typing import Optional

from core.interfaces import IPoseEngine
from core.types import HandSkeleton, Point

# MediaPipe Configuration
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

class MediaPipeEngine(IPoseEngine):
    def __init__(self, model_path: str = 'mediapipe_model/gesture_recognizer.task'):
        self.model_path = model_path
        self.recognizer = None
        self.latest_result: Optional[GestureRecognizerResult] = None
        
        # Initialize immediately
        self.initialize()

    def initialize(self) -> None:
        """Sets up the MediaPipe Gesture Recognizer."""
        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self._result_callback
        )
        self.recognizer = GestureRecognizer.create_from_options(options)
        print(f"MediaPipe Engine initialized with model: {self.model_path}")

    def _result_callback(self, result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
        """Callback invoked by MediaPipe when a result is available."""
        self.latest_result = result

    def process_frame(self, frame, timestamp: int) -> Optional[HandSkeleton]:
        """
        Processes a frame and returns a standardized HandSkeleton.
        """
        # MediaPipe expects RGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Async call - result will be updated in _result_callback
        self.recognizer.recognize_async(mp_image, int(timestamp))
        
        # Return the latest available result (might be from previous frame, which is fine for real-time)
        return self._convert_result_to_skeleton(self.latest_result, timestamp)

    def _convert_result_to_skeleton(self, result: GestureRecognizerResult, timestamp: int) -> Optional[HandSkeleton]:
        if not result or not result.hand_landmarks:
            return None
            
        # We only take the first hand for now
        landmarks = result.hand_landmarks[0]
        
        # Convert MediaPipe landmarks to our Point class
        points = [Point(x=lm.x, y=lm.y, z=lm.z) for lm in landmarks]
        
        # Confidence extraction (optional, MediaPipe gesture confidence is distinct from landmark confidence)
        confidence = 0.0
        if result.gestures and result.gestures[0]:
            confidence = result.gestures[0][0].score
            gesture_name = result.gestures[0][0].category_name
        else:
            gesture_name = None

        return HandSkeleton(
            landmarks=points,
            confidence=confidence,
            timestamp=timestamp,
            gesture=gesture_name
        )

    def release(self) -> None:
        if self.recognizer:
            self.recognizer.close()
