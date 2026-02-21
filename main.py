import cv2
import time
import logging

from engines.mediapipe_engine import MediaPipeEngine
from controllers.gesture_state import GestureStateMachine, MachineState
from controllers.cursor_mapper import PinchJoystick
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

    state_machine = GestureStateMachine(time_window_ms=config.ACTIVATION_TIME_MS)
    joystick = PinchJoystick(
        deadzone=config.JOYSTICK_DEADZONE,
        max_radius=config.JOYSTICK_MAX_RADIUS,
        max_speed=config.JOYSTICK_MAX_SPEED,
        smoothing=config.SMOOTHING_FACTOR,
    )
    gesture_detector = GestureDetector(
        joystick_threshold=config.PINCH_THRESHOLD,
        click_threshold=config.CLICK_THRESHOLD,
    )

    mouse_driver = MouseDriver() if config.SYSTEM_MODE == 'MOUSE' else None
    visualizer = Visualizer()

    # 2. Start Camera
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        logging.error("Cannot open camera")
        return

    print("System Ready.")
    print("- Hold 'Open Palm' for 2 s to activate.")
    print("- Hold 'Closed Fist' for 2 s to deactivate.")
    print("- Pinch index+thumb to move cursor (joystick).")
    print("- Tap middle+thumb for Left Click.")
    print("- Tap ring+thumb for Right Click.")

    current_state = MachineState.PASSIVE.value

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape
            timestamp_ms = time.time() * 1000

            # 3. Get skeleton
            hand_skeleton, _ = engine.process_frame(frame, timestamp_ms)

            # 4. Update activation state machine
            if hand_skeleton:
                current_state = state_machine.process_skeleton(hand_skeleton)
            else:
                current_state = MachineState.PASSIVE.value

            is_active = current_state == MachineState.ACTIVE.value

            # 5. Update gesture detector
            if hand_skeleton and is_active:
                gesture_detector.update(hand_skeleton)
            else:
                gesture_detector.update(None)

            is_pinching = gesture_detector.is_joystick_pinching

            # 6. Run joystick
            vel_x, vel_y = joystick.update(hand_skeleton if is_active else None, is_pinching)

            # 7. Handle clicks (middle=left, ring=right)
            if is_active and hand_skeleton and config.SYSTEM_MODE == 'MOUSE' and mouse_driver:
                left_event = gesture_detector.get_click_event("middle")
                right_event = gesture_detector.get_click_event("ring")

                if left_event != "NONE":
                    mouse_driver.set_button_state(left_event, button='left')
                    if left_event == "DOWN":
                        joystick.lock(config.CLICK_LOCK_MS)

                if right_event != "NONE":
                    mouse_driver.set_button_state(right_event, button='right')
                    if right_event == "DOWN":
                        joystick.lock(config.CLICK_LOCK_MS)

            # 8. Move mouse
            if is_active and (vel_x != 0 or vel_y != 0) and config.SYSTEM_MODE == 'MOUSE' and mouse_driver:
                mouse_driver.move_relative(int(vel_x), int(vel_y))

            # 9. Draw Visuals
            if hand_skeleton:
                frame = visualizer.draw_skeleton(frame, hand_skeleton)

            frame = visualizer.draw_joystick(frame, joystick, is_active)

            color = (0, 255, 0) if is_active else (0, 0, 255)
            cv2.putText(frame, f"State: {current_state}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.imshow('Gesture Control', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()
        engine.release()

if __name__ == "__main__":
    main()

