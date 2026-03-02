import logging
import platform

class MouseDriver:
    """
    Handles system-level mouse interaction.
    Abstracts away the underlying library (pyautogui/pynput) and OS differences.
    """
    def __init__(self, screen_width=1920, screen_height=1080):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.driver_available = False
        
        try:
            import pyautogui
            # Disable PyAutoGUI's fail-safe (moving mouse to corner) if needed, 
            # though it's a good safety feature to keep on.
            pyautogui.FAILSAFE = False 
            self.pg = pyautogui
            self.driver_available = True
            
            # Get actual screen size
            self.screen_width, self.screen_height = pyautogui.size()
            logging.info(f"Mouse Driver initialized. Screen size: {self.screen_width}x{self.screen_height}")
            
        except ImportError:
            logging.warning("pyautogui not found. Mouse control will be disabled. Run 'pip install pyautogui' to enable.")

    def move_to(self, x: int, y: int):
        """
        Moves the mouse cursor to the specified coordinates.
        Coordinates are clamped to the screen dimensions.
        """
        if not self.driver_available:
            return

        # Clamp coordinates to screen bounds
        safe_x = max(0, min(x, self.screen_width - 1))
        safe_y = max(0, min(y, self.screen_height - 1))
        
        # Move without animation (duration=0) for responsiveness
        self.pg.moveTo(safe_x, safe_y, duration=0)

    def move_relative(self, dx: int, dy: int):
        """
        Moves the mouse cursor relative to its current position.
        Args:
            dx: Change in x (pixels).
            dy: Change in y (pixels).
        """
        if not self.driver_available:
            return
            
        # pyautogui.move() handles relative movement
        # _pause=False ensures it doesn't add delays
        self.pg.move(dx, dy, _pause=False)

    def set_button_state(self, state: str, button: str = 'left'):
        """
        Sets the mouse button state.
        Args:
            state: 'DOWN', 'UP'
            button: 'left', 'middle', 'right'
        """
        if not self.driver_available:
            return

        if state == "DOWN":
            self.pg.mouseDown(button=button)
        elif state == "UP":
            self.pg.mouseUp(button=button)
        
    def click(self, button='left'):
        """Performs a single click."""
        if self.driver_available:
            self.pg.click(button=button)
