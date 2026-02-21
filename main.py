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
    mapper = CursorMapper(smoothing_factor=config.SMOOTHING_FACTOR, sensitivity=config.HEAD_SENSITIVITY)
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

            # 3. Get Skeletons (The "Input")
            hand_skeleton, face_skeleton = engine.process_frame(frame, timestamp_ms)

            is_active = False 
            cursor_x, cursor_y = 0, 0

            # 4. Update State (The "Logic")
            # We use the hand to determine active/passive state
            if hand_skeleton:
                current_state = state_machine.process_skeleton(hand_skeleton)
                is_active = (current_state == "active")
            else:
                current_state = "passive"
                is_active = False
                
            # 5. Map Coordinates (Head Tracking)
            if face_skeleton and is_active:
                tracking_point = face_skeleton.nose_tip
                cursor_x, cursor_y = mapper.map_absolute(tracking_point, width, height)
                
                if config.SYSTEM_MODE == 'MOUSE' and mouse_driver:
                    # Move Mouse using Absolute Position
                    # We map to screen resolution, not frame resolution
                    screen_w, screen_h = mouse_driver.screen_width, mouse_driver.screen_height
                    mouse_x, mouse_y = mapper.map_absolute(tracking_point, screen_w, screen_h)
                    mouse_driver.move_to(mouse_x, mouse_y)

            # 6. Execute Action based on Mode (Hand Gestures)
            if hand_skeleton and is_active:
                # Update all pinch states
                gesture_detector.detect_pinches(hand_skeleton)
                
                if config.SYSTEM_MODE == 'MOUSE' and mouse_driver:
                    # Left Click (Index Finger Pinch)
                    left_click = gesture_detector.get_action_state("index")
                    if left_click != "NONE":
                        mouse_driver.set_button_state(left_click, button='left')
                        
                    # Right Click (Middle Finger Pinch)
                    right_click = gesture_detector.get_action_state("middle")
                    if right_click != "NONE":
                        mouse_driver.set_button_state(right_click, button='right')
                        
            # 7. Draw Visuals (Always draw for feedback)
            # Draw Active Area for Head Tracking
            active_area = mapper.active_area
            x_min = int(active_area['x_min'] * width)
            x_max = int(active_area['x_max'] * width)
            y_min = int(active_area['y_min'] * height)
            y_max = int(active_area['y_max'] * height)
            
            # Active Area Visual (White Rectangle)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 255, 255), 1)

            # Draw a circle on the nose to show tracking point
            if is_active and face_skeleton:
                cv2.circle(frame, (cursor_x, cursor_y), 8, (0, 255, 255), -1) # Yellow dot for cursor

            # Draw skeletons for feedback
            if hand_skeleton:
                frame = visualizer.draw_skeleton(frame, hand_skeleton)
            if face_skeleton:
                frame = visualizer.draw_face_skeleton(frame, face_skeleton)

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
