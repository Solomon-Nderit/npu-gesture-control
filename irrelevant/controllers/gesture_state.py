from collections import deque
from enum import Enum
import logging
from typing import Optional
from core.types import HandSkeleton

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MachineState(Enum):
    ACTIVE = "active"
    PASSIVE = "passive"
    UNKNOWN = "unknown"

class GestureStateMachine:
    def __init__(self, time_window_ms=2000):
        """
        Initializes the state machine.
        :param time_window_ms: The duration (in ms) of consistent gestures required to switch states.
        """
        # Store tuples of (timestamp, gesture_name)
        self.gesture_buffer = deque() 
        self.time_window_ms = time_window_ms
        self.current_state = MachineState.PASSIVE
        
        # Configuration - using standard names
        self.target_gesture_active = "Open_Palm"
        self.target_gesture_passive = "Closed_Fist"

    def process_skeleton(self, skeleton: Optional[HandSkeleton]) -> str:
        """
        Updates the buffer with the new gesture and determines the current state.
        Returns the current state string ('active' or 'passive').
        """
        if not skeleton:
            return self.current_state.value
            
        gesture_name = skeleton.gesture
        timestamp_ms = skeleton.timestamp
        
        # If no gesture recognized, we might want to hold state or treat as None
        if not gesture_name:
            return self.current_state.value

        self._add_to_buffer(gesture_name, timestamp_ms)
        self._prune_buffer(timestamp_ms)
        self._evaluate_state()
        
        return self.current_state.value

    def _add_to_buffer(self, gesture, timestamp):
        self.gesture_buffer.append((timestamp, gesture))

    def _prune_buffer(self, current_time):
        # Remove gestures older than the time window
        threshold = current_time - self.time_window_ms
        while self.gesture_buffer and self.gesture_buffer[0][0] < threshold:
            self.gesture_buffer.popleft()

    def _evaluate_state(self):
        if not self.gesture_buffer:
            return
        
        # If we have very little data (start of program), don't switch yet
        if len(self.gesture_buffer) < 5: 
            return

        gestures_in_window = [g[1] for g in self.gesture_buffer]
        
        if all(g == self.target_gesture_active for g in gestures_in_window):
            self._transition_to(MachineState.ACTIVE)
        elif all(g == self.target_gesture_passive for g in gestures_in_window):
            self._transition_to(MachineState.PASSIVE)

    def _transition_to(self, new_state):
        if self.current_state != new_state:
            logging.info(f"State transition: {self.current_state.name} -> {new_state.name}")
            self.current_state = new_state
