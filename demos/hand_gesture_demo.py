#!/usr/bin/env python3
"""Hand Gesture Controlled Drone Demo.

Control a simulated drone by moving your fist!
Simple and intuitive - like dragging a cursor.

Gestures:
- ✊ Make a fist and move it → Drone follows your fist!
  - Move fist to upper screen → Drone moves forward
  - Move fist to lower screen → Drone moves backward
  - Move fist to left → Drone strafes left
  - Move fist to right → Drone strafes right
  - Keep fist in center → Drone hovers
- ✋ Open palm (5 fingers) → Takeoff/Land
- SPACE: Takeoff/Land (keyboard backup)
- ESC: Exit

Tips:
- Start with fist in center of screen
- Move fist slowly at first
- The further from center, the faster the drone moves
"""

import time

import cv2

from neuralflight.controllers.drone_controller import DroneController
from neuralflight.gestures.hand_detector import HandGestureDetector
from neuralflight.simulator.drone_sim import DroneSimulator
from neuralflight.utils.config_loader import load_config

# Add src to path


def main():
    print("=" * 60)
    print("✊ HAND GESTURE DRONE CONTROL")
    print("=" * 60)
    print()
    print("Control a drone by moving your FIST!")
    print("Simple and intuitive - like dragging a cursor.")
    print()
    print("GESTURES:")
    print("  ✊ Make a fist → Drone follows your fist position!")
    print("     📍 Move fist around screen to control drone")
    print("     🎯 Center = hover, edges = move faster")
    print()
    print("  ✋ Open palm (5 fingers) → Takeoff/Land")
    print()
    print("KEYBOARD:")
    print("  ⎵  SPACE → Takeoff/Land")
    print("  ⎋  ESC   → Exit")
    print()
    print("=" * 60)
    print()

    # Load configurations
    print("Loading configurations...")
    try:
        drone_config = load_config("drone_config")
        hand_config = load_config("hand_config")
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
    print("Initializing webcam and hand tracking...")
    print("(This may take a few seconds...)")
    try:
        detector = HandGestureDetector(hand_config)
    except Exception as e:
        print(f"Error initializing webcam: {e}")
        print("Make sure your webcam is connected and not in use")
        simulator.close()
        return

    print()
    print("✓ Ready!")
    print("✓ Show open palm (5 fingers) to takeoff")
    print("✓ Then make a fist and move it around!")
    print()

    # Track last land gesture time to prevent rapid toggling
    last_land_time = 0
    land_cooldown = 1.0  # seconds

    try:
        running = True
        while running:
            # Get frame and detected gesture
            frame, command, intensity = detector.get_frame_and_command()

            if frame is None:
                print("Warning: Could not read from camera")
                break

            # Display webcam feed
            cv2.imshow("Hand Gesture Control - Move your FIST!", frame)

            # Handle land/takeoff command with cooldown
            if command == "land":
                current_time = time.time()
                if current_time - last_land_time > land_cooldown:
                    if controller.is_flying:
                        controller.land()
                        print("Landing...")
                    else:
                        controller.takeoff()
                        print("Taking off... Now make a fist and move it around!")
                    last_land_time = current_time

            # Send other commands to drone with intensity
            elif command and controller.is_flying:
                controller.move(command, intensity=max(0.3, intensity * 0.8))
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
