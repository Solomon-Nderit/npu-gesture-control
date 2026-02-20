import cv2
import time
import logging

from engines.mediapipe_engine import MediaPipeEngine
from controllers.gesture_state import GestureStateMachine, MachineState
from controllers.cursor_mapper import CursorMapper
from controllers.gesture_detector import GestureDetector
from system.mouse_driver import MouseDriver
from utils.visualizer import Visualizer
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
    visualizer = Visualizer()

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
                    # Switch to Index Knuckle (Landmark 5) for stable tracking
                    # But need to check if landmarks array is big enough
                    tracking_point = skeleton.index_knuckle
                    
                    cursor_x, cursor_y = mapper.map(tracking_point, width, height)
                    
                    # 6. Execute Action based on Mode
                    if is_active:
                        # Update all pinch states
                        gesture_detector.detect_pinches(skeleton)
                        
                        # Check Index Pinch (The "Clutch" / "Grab")
                        # "DOWN" means we just started pinching, "NONE" means state unchanged (still pinching or still released)
                        # We need to know if it IS pinching currently, not just the transition.
                        is_grabbing = gesture_detector.finger_states["index"]["is_pinching"]

                        if config.SYSTEM_MODE == 'MOUSE' and mouse_driver:
                            # Move Mouse ONLY if "Grabbing" (Pinching Index)
                            if is_grabbing:
                                screen_x = int((cursor_x / width) * mouse_driver.screen_width)
                                screen_y = int((cursor_y / height) * mouse_driver.screen_height)
                                mouse_driver.move_to(screen_x, screen_y)
                            
                            # Left Click (Middle Finger Pinch)
                            left_click = gesture_detector.get_action_state("middle")
                            if left_click != "NONE":
                                mouse_driver.set_button_state(left_click, button='left')
                                
                            # Right Click (Ring Finger Pinch)
                            right_click = gesture_detector.get_action_state("ring")
                            if right_click != "NONE":
                                mouse_driver.set_button_state(right_click, button='right')
                        
                        # In DRAW mode, visualizer logic below handles it via is_active flag
            
            # 7. Draw Visuals (Always draw for feedback)
            # Draw a circle on the knuckle to show tracking point
            if is_active:
                cv2.circle(frame, (cursor_x, cursor_y), 8, (0, 255, 255), -1) # Yellow dot for cursor

            # If we are in MOUSE mode, we might still want to see the "cursor" on the camera feed
            frame = visualizer.update_canvas(frame, cursor_x, cursor_y, is_active and config.SYSTEM_MODE == 'DRAW')

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
