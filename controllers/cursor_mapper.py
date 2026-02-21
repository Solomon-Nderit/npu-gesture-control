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
        self._sensitivity = sensitivity
        
        # Default active area centred on frame — overridden by calibrate()
        self._set_active_area(0.5, 0.5)

    def calibrate(self, origin: Point) -> None:
        """
        Recentre the active area around the given nose position.
        Call this once when the system transitions to active.
        """
        self.history.clear()
        self._set_active_area(origin.x, origin.y)

    def _set_active_area(self, cx: float, cy: float) -> None:
        s = self._sensitivity
        self.active_area = {
            'x_min': cx - s, 'x_max': cx + s,
            'y_min': cy - s, 'y_max': cy + s,
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

