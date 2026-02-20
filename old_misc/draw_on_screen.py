import cv2
import numpy as np

class Drawer:
    def __init__(self):
        self.canvas = None
        self.prev_point = None
        self.drawing_color = (0, 255, 0) # Green
        self.thickness = 5

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
