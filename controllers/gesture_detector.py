from core.types import HandSkeleton, Point
import math

class GestureDetector:
    """
    Detects specific gesture intents from a hand skeleton, such as pinching.
    """
    def __init__(self, click_threshold: float = 0.05):
        """
        Args:
            click_threshold: The normalized distance between thumb and index tip 
                             to consider a 'click' (pinch). 
                             Default is 0.05 (5% of screen width approx).
        """
        self.click_threshold = click_threshold
        self.is_pinching = False
        self.was_pinching = False

    def detect_pinch(self, skeleton: HandSkeleton) -> bool:
        """
        Checks if the index finger and thumb are pinching.
        Returns True if the distance is below the threshold.
        """
        if not skeleton or len(skeleton.landmarks) <= 8:
            return False

        thumb_tip = skeleton.thumb_tip
        index_tip = skeleton.index_tip

        # Calculate Euclidean distance in normalized coordinates (3D distance including Z is better if available, 
        # but 2D is often sufficient for screen interaction)
        distance = math.sqrt(
            (thumb_tip.x - index_tip.x)**2 + 
            (thumb_tip.y - index_tip.y)**2
        )

        is_currently_pinching = distance < self.click_threshold
        
        # State tracking (useful for debouncing or edge detection if needed later)
        self.was_pinching = self.is_pinching
        self.is_pinching = is_currently_pinching
        
        return self.is_pinching

    def get_click_state(self) -> str:
        """
        Returns the simplified mouse button state based on pinch.
        Returns: 'DOWN', 'UP', or 'NONE' (no change)
        """
        if self.is_pinching and not self.was_pinching:
            return "DOWN"
        elif not self.is_pinching and self.was_pinching:
            return "UP"
        return "NONE"
