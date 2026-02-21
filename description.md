# NPU Gesture Control

## Overview
NPU Gesture Control is a modular, touchless PC control interface designed to allow users to manipulate their computer entirely using gestures. The project aims to provide a power-efficient, low-latency solution that can eventually run on a Ryzen NPU using YOLOv8.

## Control Paradigm: Head Pointing + Micro-Gestures
After experimenting with various control paradigms (absolute finger tracking, joystick relative movement), the project settled on a "Head Pointing + Micro-Gestures" approach to solve the "Midas Touch" problem (where clicking moves the cursor) and "Gorilla Arm" (fatigue from holding the arm up).

- **Cursor Movement**: Controlled by tracking the tip of the user's nose. A small active area in the center of the camera frame is mapped to the full screen, allowing for precise cursor control with minimal head movement.
- **Clicking**: Controlled by micro-gestures (pinches) from the user's hand, which can rest comfortably on the desk or armrest.
  - **Left Click**: Thumb + Index Finger pinch.
  - **Right Click**: Thumb + Middle Finger pinch.
- **State Management**: The system has an Active and Passive state to prevent accidental inputs.
  - **Activate**: Hold an "Open Palm" gesture for 2 seconds.
  - **Deactivate**: Hold a "Closed Fist" gesture for 2 seconds.

## Architecture
The project is built with a strict Interface-based Object-Oriented Programming (OOP) architecture to decouple the AI engine from the control logic. This ensures that the underlying AI models can be swapped out (e.g., moving from MediaPipe to a custom YOLOv8 model on an NPU) without breaking the rest of the application.

### Directory Structure
- `core/`: Contains the foundational data structures and interfaces.
  - `types.py`: Defines standardized data packets (`Point`, `HandSkeleton`, `FaceSkeleton`).
  - `interfaces.py`: Defines the `IPoseEngine` contract that all AI engines must implement.
- `engines/`: Contains the AI model wrappers.
  - `mediapipe_engine.py`: Implements `IPoseEngine` using MediaPipe's `GestureRecognizer` and `FaceLandmarker`.
- `controllers/`: Contains the logic for interpreting the skeleton data.
  - `cursor_mapper.py`: Maps the nose tip coordinates to screen pixels with exponential moving average smoothing.
  - `gesture_detector.py`: Detects pinches by calculating the Euclidean distance between finger tips.
  - `gesture_state.py`: Manages the Active/Passive state machine based on sustained gestures.
- `system/`: Contains OS-level integrations.
  - `mouse_driver.py`: Wraps `pyautogui` to execute mouse movements and clicks.
- `utils/`: Contains utility scripts.
  - `visualizer.py`: Handles drawing the skeletons and UI overlays on the camera feed.
- `main.py`: The orchestrator that ties all the components together.
- `config.py`: Centralized configuration settings (thresholds, smoothing factors, modes).

## Future Roadmap
The ultimate goal is to port the AI processing to a Ryzen NPU using a custom-trained YOLOv8 model for pose estimation. The modular architecture ensures that this transition will only require creating a new `RyzenAIEngine` that implements `IPoseEngine`, leaving the control logic and system integrations untouched.
