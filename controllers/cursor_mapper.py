from collections import deque
import numpy as np
from core.types import Point

class CursorMapper:
    """
    Maps normalized head/nose coordinates to screen coordinates with smoothing.
    """
    def __init__(self, smoothing_factor=0.8, history_length=10, sensitivity=0.2):
        self.smoothing_factor = smoothing_factor
        self.history = deque(maxlen=history_length)
        
        # Define the active area in the camera frame that maps to the full screen.
        # A smaller area means higher sensitivity (less head movement required).
        # sensitivity=0.2 means the active zone is 0.5 +/- 0.2 = [0.3, 0.7]
        self.active_area = {
            'x_min': 0.5 - sensitivity, 'x_max': 0.5 + sensitivity,
            'y_min': 0.5 - sensitivity, 'y_max': 0.5 + sensitivity
        }

    def map_absolute(self, point: Point, screen_width: int, screen_height: int) -> tuple[int, int]:
        """
        Maps a normalized point within the active area to full screen coordinates.
        """
        # Clamp the point to the active area
        x = max(self.active_area['x_min'], min(point.x, self.active_area['x_max']))
        y = max(self.active_area['y_min'], min(point.y, self.active_area['y_max']))
        
        # Normalize within the active area (0.0 to 1.0)
        norm_x = (x - self.active_area['x_min']) / (self.active_area['x_max'] - self.active_area['x_min'])
        norm_y = (y - self.active_area['y_min']) / (self.active_area['y_max'] - self.active_area['y_min'])
        
        # Map to screen pixels
        target_x = int(norm_x * screen_width)
        target_y = int(norm_y * screen_height)

        # Exponential Moving Average Smoothing
        if not self.history:
            self.history.append((target_x, target_y))
            return target_x, target_y
            
        prev_x, prev_y = self.history[-1]
        
        # Apply smoothing factor (0.0 = no smoothing, 1.0 = infinite smoothing)
        # We use 1 - smoothing_factor as the weight for the new value
        alpha = 1.0 - self.smoothing_factor
        smoothed_x = int(prev_x * (1 - alpha) + target_x * alpha)
        smoothed_y = int(prev_y * (1 - alpha) + target_y * alpha)
        
        self.history.append((smoothed_x, smoothed_y))
        
        return smoothed_x, smoothed_y

