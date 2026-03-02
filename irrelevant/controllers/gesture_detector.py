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
        self.is_pinching = False # Deprecated: use specific dict below
        self.was_pinching = False
        
        # Track states for separate fingers
        # "index": Drag
        # "middle": Left Click
        # "ring": Right Click
        self.finger_states = {
            "index": {"is_pinching": False, "was_pinching": False},
            "middle": {"is_pinching": False, "was_pinching": False},
            "ring": {"is_pinching": False, "was_pinching": False}
        }

    def detect_pinches(self, skeleton: HandSkeleton):
        """
        Updates the pinch state for index, middle, and ring fingers.
        """
        if not skeleton or len(skeleton.landmarks) <= 16:
            return

        thumb_tip = skeleton.thumb_tip
        
        fingers = {
            "index": skeleton.index_tip,
            "middle": skeleton.middle_tip,
            "ring": skeleton.ring_tip
        }

        for finger_name, finger_tip in fingers.items():
             # Calculate Euclidean distance
            distance = math.sqrt(
                (thumb_tip.x - finger_tip.x)**2 + 
                (thumb_tip.y - finger_tip.y)**2
            )
            
            is_currently_pinching = distance < self.click_threshold
            
            # Update state
            self.finger_states[finger_name]["was_pinching"] = self.finger_states[finger_name]["is_pinching"]
            self.finger_states[finger_name]["is_pinching"] = is_currently_pinching

    def get_action_state(self, finger_name: str) -> str:
        """
        Returns the simplified button state based on pinch ('DOWN', 'UP', 'NONE').
        Args:
            finger_name: "index", "middle", or "ring"
        """
        state = self.finger_states.get(finger_name)
        if not state: 
            return "NONE"

        if state["is_pinching"] and not state["was_pinching"]:
            return "DOWN"
        elif not state["is_pinching"] and state["was_pinching"]:
            return "UP"
        return "NONE"

    # Deprecated methods for backward compatibility
    def detect_pinch(self, skeleton: HandSkeleton) -> bool:
        self.detect_pinches(skeleton)
        return self.finger_states["index"]["is_pinching"]
    
    def get_click_state(self) -> str:
        return self.get_action_state("index")
