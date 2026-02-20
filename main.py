import cv2
import time
import logging

from engines.mediapipe_engine import MediaPipeEngine
from controllers.gesture_state import GestureStateMachine, MachineState
from controllers.cursor_mapper import CursorMapper
from controllers.gesture_detector import GestureDetector
from system.mouse_driver import MouseDriver
import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def main():
    print(f"Starting Gesture Control Application in {config.SYSTEM_MODE} mode...")

    # 1. Initialize Components
    engine = MediaPipeEngine()
    
    # Logic Controllers
    state_machine = GestureStateMachine(time_window_ms=config.ACTIVATION_TIME_MS)
    mapper = CursorMapper(smoothing_factor=config.SMOOTHING_FACTOR)
    gesture_detector = GestureDetector(click_threshold=config.PINCH_THRESHOLD)
    
    # System / Output
    mouse_driver = MouseDriver() if config.SYSTEM_MODE == 'MOUSE' else None
    # visualizer = Visualizer()

    # 2. Start Camera
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        logging.error("Cannot open camera")
        return

    print("System Ready.")
    print("- Hold 'Open Palm' to activate.")
    print("- Hold 'Closed Fist' to deactivate.")
    if config.SYSTEM_MODE == 'MOUSE':
        print("- Pinch to Click.")
    
    current_state = MachineState.PASSIVE.value

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Pre-processing (Mirroring)
            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape
            timestamp_ms = time.time() * 1000 # Use system time as fallback or cap.get(cv2.CAP_PROP_POS_MSEC)

            # 3. Get Skeleton (The "Input")
            skeleton = engine.process_frame(frame, timestamp_ms)

            is_active = False 
            cursor_x, cursor_y = 0, 0

            if skeleton:
                # 4. Update State (The "Logic")
                current_state = state_machine.process_skeleton(skeleton)
                is_active = (current_state == "active")
                
                # 5. Map Coordinates
                if len(skeleton.landmarks) > 8:
                    index_tip = skeleton.landmarks[8]
                    cursor_x, cursor_y = mapper.map(index_tip, width, height)
                    
                    # 6. Execute Action based on Mode
                    if is_active:
                        if config.SYSTEM_MODE == 'MOUSE' and mouse_driver:
                             # Move Mouse
                            screen_x = int((cursor_x / width) * mouse_driver.screen_width)
                            screen_y = int((cursor_y / height) * mouse_driver.screen_height)
                            mouse_driver.move_to(screen_x, screen_y)
                            
                            # Handle Clicks (Pinch)
                            if gesture_detector.detect_pinch(skeleton):
                                click_state = gesture_detector.get_click_state()
                                if click_state != "NONE":
                                    mouse_driver.set_button_state(click_state)
                        
                        # In DRAW mode, visualizer logic below handles it via is_active flag
            
            # 7. Draw Visuals (Always draw for feedback)
            # If we are in MOUSE mode, we might still want to see the "cursor" on the camera feed
            # frame = visualizer.update_canvas(frame, cursor_x, cursor_y, is_active and config.SYSTEM_MODE == 'DRAW')

            # Overlay State Text
            color = (0, 255, 0) if is_active else (0, 0, 255)
            cv2.putText(frame, f"State: {current_state}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Display
            cv2.imshow('Gesture Control', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        engine.release()

if __name__ == "__main__":
    main()
