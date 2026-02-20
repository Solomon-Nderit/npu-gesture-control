from abc import ABC, abstractmethod
from typing import Optional, Any
from .types import HandSkeleton

class IPoseEngine(ABC):
    """
    Interface that all pose estimation engines must implement.
    """
    
    @abstractmethod
    def initialize(self) -> None:
        """Sets up the model and resources."""
        pass

    @abstractmethod
    def process_frame(self, frame: Any, timestamp: int) -> Optional[HandSkeleton]:
        """
        Processes a single video frame and returns a HandSkeleton if a hand is detected.
        Args:
            frame: The image frame (usually numpy array from OpenCV).
            timestamp: The timestamp of the frame in milliseconds.
        Returns:
            HandSkeleton object or None if no hand is detected.
        """
        pass

    @abstractmethod
    def release(self) -> None:
        """Releases resources."""
        pass


class IGestureController(ABC):
    """
    Interface for the logic layer that translates skeletons into system actions.
    """
    @abstractmethod
    def process_skeleton(self, skeleton: Optional[HandSkeleton], frame_shape: tuple) -> None:
        pass
