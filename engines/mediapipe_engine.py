import cv2
import mediapipe as mp
from typing import Optional, Tuple

from core.interfaces import IPoseEngine
from core.types import HandSkeleton, FaceSkeleton, Point

# MediaPipe Configuration
BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Hand Tracking only
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult

class MediaPipeEngine(IPoseEngine):
    def __init__(self, hand_model_path: str = 'mediapipe_model/gesture_recognizer.task'):
        self.hand_model_path = hand_model_path
        self.recognizer = None
        self.latest_hand_result: Optional[GestureRecognizerResult] = None
        self.initialize()

    def initialize(self) -> None:
        """Sets up the MediaPipe Gesture Recognizer (hands only)."""
        hand_options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=self.hand_model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self._hand_result_callback
        )
        self.recognizer = GestureRecognizer.create_from_options(hand_options)
        print(f"MediaPipe Engine initialized with model: {self.hand_model_path}")

    def _hand_result_callback(self, result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
        self.latest_hand_result = result

    def process_frame(self, frame, timestamp: int) -> Tuple[Optional[HandSkeleton], Optional[FaceSkeleton]]:
        """
        Processes a frame. Returns (HandSkeleton, None) — face tracking not used in this branch.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.recognizer.recognize_async(mp_image, int(timestamp))
        return self._convert_hand_result_to_skeleton(self.latest_hand_result, timestamp), None

    def _convert_hand_result_to_skeleton(self, result, timestamp: int) -> Optional[HandSkeleton]:
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

    def release(self) -> None:
        if self.recognizer:
            self.recognizer.close()
