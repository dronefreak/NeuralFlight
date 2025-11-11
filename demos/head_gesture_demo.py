#!/usr/bin/env python3
"""Head Gesture Controlled Drone Demo.

Control a simulated drone using head movements detected via webcam.
No EEG hardware required!

Controls:
- Look down: Move forward
- Look up: Move backward
- Turn head left/right: Rotate
- Tilt head left/right: Strafe sideways
- SPACE: Takeoff/Land
- ESC: Exit
"""

import cv2

from neuralflight.controllers.drone_controller import DroneController
from neuralflight.gestures.head_detector import HeadGestureDetector
from neuralflight.simulator.drone_sim import DroneSimulator
from neuralflight.utils.config_loader import load_config


def main():
    print("=" * 60)
    print("🧠 HEAD GESTURE DRONE CONTROL")
    print("=" * 60)
    print()
    print("This demo uses your webcam to detect head movements")
    print("and control a simulated drone. No EEG hardware needed!")
    print()
    print("CONTROLS:")
    print("  👇 Look DOWN    → Drone moves FORWARD")
    print("  👆 Look UP      → Drone moves BACKWARD")
    print("  👈 Turn LEFT    → Drone rotates LEFT")
    print("  👉 Turn RIGHT   → Drone rotates RIGHT")
    print("  ↖️  Tilt LEFT    → Drone strafes LEFT")
    print("  ↗️  Tilt RIGHT   → Drone strafes RIGHT")
    print("  ⎵  SPACE       → Takeoff/Land")
    print("  ⎋  ESC         → Exit")
    print()
    print("=" * 60)
    print()

    # Load configurations
    print("Loading configurations...")
    try:
        drone_config = load_config("drone_config")
        gesture_config = load_config("gesture_config")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Make sure you're running from the project root directory")
        return

    # Initialize simulator
    print("Initializing drone simulator...")
    simulator = DroneSimulator(drone_config)

    # Initialize controller
    controller = DroneController(simulator)

    # Initialize gesture detector
    print("Initializing webcam and gesture detector...")
    print("(This may take a few seconds...)")
    try:
        detector = HeadGestureDetector(gesture_config)
    except Exception as e:
        print(f"Error initializing webcam: {e}")
        print("Make sure your webcam is connected and not in use")
        simulator.close()
        return

    print()
    print("✓ Ready! Press SPACE to takeoff, then move your head to fly.")
    print()

    try:
        running = True
        while running:
            # Get frame and detected gesture
            frame, command = detector.get_frame_and_command()

            if frame is None:
                print("Warning: Could not read from camera")
                break

            # Display webcam feed
            cv2.imshow("Head Gesture Control", frame)

            # Send command to drone if detected
            if command and controller.is_flying:
                controller.move(command, intensity=0.8)
            elif controller.is_flying:
                controller.hover()

            # Update simulator
            running = simulator.update()

            # Handle OpenCV window events
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                running = False

    except KeyboardInterrupt:
        print("\nInterrupted by user")

    finally:
        print("\nCleaning up...")
        detector.release()
        simulator.close()
        print("Done!")


if __name__ == "__main__":
    main()
