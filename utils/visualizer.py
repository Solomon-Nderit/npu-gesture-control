import cv2
import numpy as np
from core.types import HandSkeleton, FaceSkeleton, Point

class Visualizer:
    def __init__(self):
        self.canvas = None
        self.prev_point = None
        self.drawing_color = (0, 255, 0) # Green
        self.thickness = 5
        self.mp_drawing = None

    def _init_mp_drawing(self):
        # Lazy import MediaPipe drawing utils only if needed for debugging skeleton
        try:
            import mediapipe as mp
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_hands = mp.solutions.hands
            self.mp_styles = mp.solutions.drawing_styles
        except ImportError:
            pass

    def draw_skeleton(self, frame, skeleton: HandSkeleton):
        """Draws the hand landmarks and connections on the frame."""
        if not skeleton or not skeleton.landmarks:
            return frame
        
        # Manually draw landmarks if MediaPipe utilities are tricky to use with custom objects
        # Or instantiate a dummy NormalizedLandmarkList for MediaPipe drawing util
        # For simplicity, let's draw circles for now or use MediaPipe if installed
        
        height, width, _ = frame.shape
        
        # Connect key points (very basic skeleton)
        # Wrist to Thumb
        self._draw_connection(frame, skeleton.landmarks[0], skeleton.landmarks[1], width, height)
        self._draw_connection(frame, skeleton.landmarks[1], skeleton.landmarks[2], width, height)
        self._draw_connection(frame, skeleton.landmarks[2], skeleton.landmarks[3], width, height)
        self._draw_connection(frame, skeleton.landmarks[3], skeleton.landmarks[4], width, height)
        
        # Wrist to Index
        self._draw_connection(frame, skeleton.landmarks[0], skeleton.landmarks[5], width, height)
        self._draw_connection(frame, skeleton.landmarks[5], skeleton.landmarks[6], width, height)
        self._draw_connection(frame, skeleton.landmarks[6], skeleton.landmarks[7], width, height)
        self._draw_connection(frame, skeleton.landmarks[7], skeleton.landmarks[8], width, height)
        
        # Draw points
        for lm in skeleton.landmarks:
            cx, cy = int(lm.x * width), int(lm.y * height)
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            
        return frame

    def draw_face_skeleton(self, frame, skeleton: FaceSkeleton):
        """Draws the face landmarks on the frame."""
        if not skeleton or not skeleton.landmarks:
            return frame
            
        height, width, _ = frame.shape
        
        # Draw points (just a few key ones to avoid clutter)
        # Nose tip is landmark 1
        nose_tip = skeleton.nose_tip
        cx, cy = int(nose_tip.x * width), int(nose_tip.y * height)
        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1) # Blue dot for nose tip
        
        return frame

    def _draw_connection(self, frame, p1: Point, p2: Point, w, h):
        cv2.line(frame, 
                (int(p1.x * w), int(p1.y * h)), 
                (int(p2.x * w), int(p2.y * h)), 
                (255, 255, 255), 2)

    def update_canvas(self, frame, x, y, is_active):
        """
        Updates the canvas with the new point.
        Args:
            frame: The current video frame.
            x, y: The current coordinates of the pointer.
            is_active: Boolean indicating if the 'pen' is down.
        Returns:
            The frame combined with the drawing canvas.
        """
        # Initialize canvas if it doesn't exist or if frame size changes
        if self.canvas is None or self.canvas.shape != frame.shape:
            self.canvas = np.zeros_like(frame)

        current_point = (int(x), int(y))

        # Check if we should draw
        if is_active and self.prev_point is not None:
             cv2.line(self.canvas, self.prev_point, current_point, self.drawing_color, self.thickness)
        
        # Reset previous point if we are not active, so we don't connect lines 
        # when the user lifts their finger and moves it elsewhere.
        if not is_active:
            self.prev_point = None
        else:
            self.prev_point = current_point

        # Merge the frame and the canvas
        # Create a mask of the drawing
        img_gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY_INV)
        
        # Mask out the drawing area from the original frame (make it black where drawing is)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame, img_inv)
        
        # Add the canvas (which has color) to the masked frame
        frame = cv2.bitwise_or(frame, self.canvas)

        return frame
