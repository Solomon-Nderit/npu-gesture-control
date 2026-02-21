import time
import math
from core.types import HandSkeleton, Point


class PinchJoystick:
    """
    Joystick controller driven by pinch position.

    When the user pinches index+thumb, the midpoint at that moment becomes the
    joystick origin. Deflecting the pinch from that origin generates cursor
    velocity proportional to the deflection magnitude (after a deadzone),
    with a quadratic ramp up to a configurable max speed.

    Release and re-pinch resets the origin to wherever the hand currently is.
    """

    def __init__(self,
                 deadzone: float = 0.04,
                 max_radius: float = 0.25,
                 max_speed: float = 25.0,
                 smoothing: float = 0.7):
        """
        Args:
            deadzone:    Normalized deflection radius with zero output (absorbs tremor).
            max_radius:  Normalized deflection at which max_speed is reached.
            max_speed:   Max pixels-per-frame at full deflection.
            smoothing:   EMA coefficient for velocity smoothing (0=none, 1=max).
        """
        self.deadzone = deadzone
        self.max_radius = max_radius
        self.max_speed = max_speed
        self.smoothing = smoothing

        # Joystick origin (set on pinch-start)
        self.origin: Point | None = None

        # Current pinch midpoint (for visualisation)
        self.current: Point | None = None

        # Smoothed velocity
        self._vel_x: float = 0.0
        self._vel_y: float = 0.0

        # Click-lock: freeze joystick output for a short window after a click
        self._lock_until_ms: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, skeleton: HandSkeleton, is_pinching: bool) -> tuple[float, float]:
        """
        Call every frame. Returns (vel_x, vel_y) in pixels/frame.

        Args:
            skeleton:    Current hand skeleton.
            is_pinching: Whether index+thumb are currently pinched.
        """
        if not is_pinching or not skeleton:
            # Reset when hand lifts or unpinches
            self.origin = None
            self.current = None
            self._vel_x = 0.0
            self._vel_y = 0.0
            return 0.0, 0.0

        midpoint = self._pinch_midpoint(skeleton)
        self.current = midpoint

        if self.origin is None:
            # First frame of this pinch — set origin
            self.origin = midpoint
            return 0.0, 0.0

        # Check click-lock
        if time.time() * 1000 < self._lock_until_ms:
            return 0.0, 0.0

        # Deflection from origin (normalized coords)
        dx = midpoint.x - self.origin.x
        dy = midpoint.y - self.origin.y
        dist = math.sqrt(dx * dx + dy * dy)

        if dist < self.deadzone or dist == 0:
            raw_vx, raw_vy = 0.0, 0.0
        else:
            # Amount past deadzone, scaled to [0, 1] over the active range
            effective = min(dist - self.deadzone, self.max_radius - self.deadzone)
            active_range = max(self.max_radius - self.deadzone, 1e-6)
            t = effective / active_range          # 0→1
            speed = (t ** 2) * self.max_speed     # Quadratic ramp

            raw_vx = (dx / dist) * speed
            raw_vy = (dy / dist) * speed

        # EMA smoothing
        alpha = 1.0 - self.smoothing
        self._vel_x = self._vel_x * (1 - alpha) + raw_vx * alpha
        self._vel_y = self._vel_y * (1 - alpha) + raw_vy * alpha

        return self._vel_x, self._vel_y

    def lock(self, duration_ms: float) -> None:
        """Freeze joystick output for duration_ms (call when a click fires)."""
        self._lock_until_ms = time.time() * 1000 + duration_ms

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pinch_midpoint(self, skeleton: HandSkeleton) -> Point:
        t = skeleton.thumb_tip
        i = skeleton.index_tip
        return Point(x=(t.x + i.x) / 2, y=(t.y + i.y) / 2)

