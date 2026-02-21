# Configuration parameters

# Engine Selection: 'MEDIAPIPE' or 'RYZEN_AI' (Future)
ACTIVE_ENGINE = 'MEDIAPIPE'

# Mode Selection: 'MOUSE' or 'DRAW'
SYSTEM_MODE = 'MOUSE' 

# Camera Settings
CAMERA_INDEX = 0

# Gesture Thresholds
PINCH_THRESHOLD = 0.04 # Normalized distance
ACTIVATION_TIME_MS = 2000  # Time to hold gesture to trigger state change

# Smoothing
SMOOTHING_FACTOR = 0.8

# Head Tracking Sensitivity
# Controls the active area of the camera frame mapped to the full screen.
# Smaller value = higher sensitivity (less head movement needed).
# Range: 0.05 (very high) to 0.45 (very low). Default: 0.2
HEAD_SENSITIVITY = 0.02
