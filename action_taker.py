
from collections import deque
from enum import Enum
import logging

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
        
        # Configuration
        self.target_gesture_active = "Open_Palm"
        self.target_gesture_passive = "Closed_Fist"

    def update(self, gesture_name, timestamp_ms):
        """
        Updates the buffer with the new gesture and determines the current state.
        Returns the current state string ('active' or 'passive').
        """
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

        # Check mostly consistent gestures? Or strictly consistent?
        # Sticking to strict consistency for now as per original logic.
        
        # If we have very little data (start of program), don't switch yet to avoid glitches
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

# Singleton instance to maintain state across function calls
_state_machine = GestureStateMachine(time_window_ms=2000)

def take_action(gesture, index_finger_coordinates, thumb_tip, time_and_gesture):
    """
    Main entry point for gesture processing.
    
    Args:
        gesture (str): The name of the detected gesture.
        index_finger_coordinates (NormalizedLandmark): The coordinates of the index finger tip.
        thumb_tip (NormalizedLandmark): The coordinates of the thumb tip.
        time_and_gesture (dict): A dictionary of timestamp -> gesture. 
                                 (Legacy argument, largely ignored now in favor of internal tracking 
                                  except for extracting the latest timestamp if needed).
    
    Returns:
        str: The current state of the machine ('active' or 'passive').
    """
    
    # Extract the latest timestamp. 
    # Since the caller appends to the dict right before calling us, the last item is current.
    if time_and_gesture:
        # Get last key
        latest_timestamp = next(reversed(time_and_gesture))
    else:
        # Fallback if dict is empty (shouldn't happen in normal flow)
        import time
        latest_timestamp = time.time() * 1000

    current_state = _state_machine.update(gesture, latest_timestamp)
    return current_state




