# Configuration parameters

# Engine Selection: 'MEDIAPIPE' or 'RYZEN_AI' (Future)
ACTIVE_ENGINE = 'MEDIAPIPE'

# Mode Selection: 'MOUSE' or 'DRAW'
SYSTEM_MODE = 'MOUSE' 

# Camera Settings
CAMERA_INDEX = 0

# Gesture Thresholds
PINCH_THRESHOLD = 0.03       # Normalized distance for index+thumb joystick pinch
CLICK_THRESHOLD = 0.04        # Normalized distance for click taps (middle/ring + thumb)
ACTIVATION_TIME_MS = 2000     # Time to hold gesture to trigger state change
CLICK_LOCK_MS = 200           # Freeze joystick for this long after a click tap fires

# Smoothing
SMOOTHING_FACTOR = 0.7        # EMA smoothing on velocity (0=none, 1=max)

# Joystick Settings
JOYSTICK_DEADZONE = 0.02      # Normalized radius with no movement (absorbs tremor)
JOYSTICK_MAX_RADIUS = 0.15    # Normalized radius at which max speed is reached
JOYSTICK_MAX_SPEED = 25.0     # Max pixels moved per frame at full deflection
