from dataclasses import dataclass, field
from typing import List, Optional
import time

@dataclass
class Point:
    """Represents a 3D point in normalized coordinates (0.0 to 1.0)."""
    x: float
    y: float
    z: float = 0.0

@dataclass
class HandSkeleton:
    """
    Standardized representation of a detected hand.
    Independent of the underlying engine (MediaPipe, YOLO, etc).
    """
    landmarks: List[Point]
    confidence: float
    timestamp: int  # in milliseconds
    gesture: Optional[str] = None
    
    # Helper properties for common landmarks
    @property
    def wrist(self) -> Point:
        return self.landmarks[0]

    @property
    def thumb_tip(self) -> Point:
        return self.landmarks[4]

    @property
    def index_tip(self) -> Point:
        return self.landmarks[8]

    @property
    def middle_tip(self) -> Point:
        return self.landmarks[12]
