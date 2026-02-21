from core.types import HandSkeleton
import math


class GestureDetector:
    """
    Detects gesture intents from a hand skeleton.

    Roles:
      - Index + Thumb pinch  → joystick gate (cursor moves while held)
      - Middle + Thumb tap   → Left Click
      - Ring + Thumb tap     → Right Click
    """

    def __init__(self, joystick_threshold: float = 0.04, click_threshold: float = 0.04):
        """
        Args:
            joystick_threshold: Distance for index+thumb joystick activation.
            click_threshold:    Distance for middle/ring+thumb click taps.
        """
        self.joystick_threshold = joystick_threshold
        self.click_threshold = click_threshold

        # Click tap states (middle and ring only — index is the joystick gate)
        self.click_states = {
            "middle": {"is_pinching": False, "was_pinching": False},
            "ring":   {"is_pinching": False, "was_pinching": False},
        }

        # Joystick gate state
        self._joystick_pinching = False

    # ------------------------------------------------------------------
    # Joystick gate
    # ------------------------------------------------------------------

    def update(self, skeleton: HandSkeleton) -> None:
        """Update all gesture states from the latest skeleton."""
        if not skeleton or len(skeleton.landmarks) <= 16:
            self._joystick_pinching = False
            for s in self.click_states.values():
                s["was_pinching"] = s["is_pinching"]
                s["is_pinching"] = False
            return

        thumb = skeleton.thumb_tip

        # Joystick gate: index + thumb
        self._joystick_pinching = self._dist(thumb, skeleton.index_tip) < self.joystick_threshold

        # Click taps
        click_fingers = {
            "middle": skeleton.middle_tip,
            "ring":   skeleton.ring_tip,
        }
        for name, tip in click_fingers.items():
            dist = self._dist(thumb, tip)
            self.click_states[name]["was_pinching"] = self.click_states[name]["is_pinching"]
            self.click_states[name]["is_pinching"] = dist < self.click_threshold

    @property
    def is_joystick_pinching(self) -> bool:
        return self._joystick_pinching

    # ------------------------------------------------------------------
    # Click events
    # ------------------------------------------------------------------

    def get_click_event(self, finger: str) -> str:
        """
        Returns the button transition for a click finger.
        'DOWN' on pinch start, 'UP' on release, 'NONE' otherwise.

        Args:
            finger: 'middle' (left click) or 'ring' (right click).
        """
        state = self.click_states.get(finger)
        if not state:
            return "NONE"
        if state["is_pinching"] and not state["was_pinching"]:
            return "DOWN"
        if not state["is_pinching"] and state["was_pinching"]:
            return "UP"
        return "NONE"

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    @staticmethod
    def _dist(a, b) -> float:
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

