from collections import deque
import numpy as np
from core.types import Point

class CursorMapper:
    """
    Maps normalized hand coordinates to screen coordinates with smoothing.
    """
    def __init__(self, smoothing_factor=0.5, history_length=5):
        self.smoothing_factor = smoothing_factor
        self.history = deque(maxlen=history_length)
        self.prev_point = None

    def map(self, point: Point, screen_width: int, screen_height: int) -> tuple[int, int]:
        """
        Converts normalized coordinates (0-1) to pixel coordinates.
        Applies smoothing if history is available.
        """
        # Convert to pixels
        # Note: We assume the input point.x is already mirrored if needed by the engine 
        # or we handle mirroring here. The previous code mirrored the FRAME.
        # So the landmarks from MediaPipe on a flipped frame are already correct relative to that frame.
        
        target_x = int(point.x * screen_width)
        target_y = int(point.y * screen_height)

        # Simple Moving Average Smoothing
        self.history.append((target_x, target_y))
        
        avg_x = int(np.mean([p[0] for p in self.history]))
        avg_y = int(np.mean([p[1] for p in self.history]))

        return avg_x, avg_y
