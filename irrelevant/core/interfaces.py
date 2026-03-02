from abc import ABC, abstractmethod
from typing import Optional, Any, Tuple
from .types import HandSkeleton, FaceSkeleton

class IPoseEngine(ABC):
    """
    Interface that all pose estimation engines must implement.
    """
    
    @abstractmethod
    def initialize(self) -> None:
        """Sets up the model and resources."""
        pass

    @abstractmethod
    def process_frame(self, frame: Any, timestamp: int) -> Tuple[Optional[HandSkeleton], Optional[FaceSkeleton]]:
        """
        Processes a single video frame and returns skeletons if detected.
        Args:
            frame: The image frame (usually numpy array from OpenCV).
            timestamp: The timestamp of the frame in milliseconds.
        Returns:
            Tuple of (HandSkeleton, FaceSkeleton). Either can be None.
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
