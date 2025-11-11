"""
Hand gesture detection using Mediapipe Hands - FIST FOLLOWING MODE.
"""

from collections import deque
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


class HandGestureDetector:
    """Detects hand gestures using Mediapipe Hands for drone control.

    Simplified: Fist follows mode - move your fist, drone follows!
    """

    def __init__(self, config: dict):
        self.config = config

        # Initialize Mediapipe
        mp_config = config["mediapipe"]
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=mp_config["min_detection_confidence"],
            min_tracking_confidence=mp_config["min_tracking_confidence"],
            max_num_hands=mp_config["max_num_hands"],
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Initialize camera
        camera_config = config["camera"]
        self.cap = cv2.VideoCapture(camera_config["device_id"])
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config["width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config["height"])
        self.cap.set(cv2.CAP_PROP_FPS, camera_config["fps"])

        # Gesture thresholds
        self.thresholds = config["gestures"]
        self.commands = config["commands"]

        # Filtering
        filter_config = config["filtering"]
        self.filtering_enabled = filter_config["enabled"]
        self.window_size = filter_config["window_size"]

        # Smoothing buffers
        self.x_buffer = deque(maxlen=self.window_size)
        self.y_buffer = deque(maxlen=self.window_size)

        # Screen center for reference
        self.screen_center = (
            camera_config["width"] // 2,
            camera_config["height"] // 2,
        )

    def _is_fist(self, hand_landmarks) -> bool:
        """Detect if hand is making a fist.

        A fist is when all fingertips are close to the palm.
        """
        # Get wrist position
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]

        # Get all fingertips
        fingertips = [
            hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP],
        ]

        # Calculate average distance from wrist to fingertips
        total_distance = 0
        for tip in fingertips:
            dx = tip.x - wrist.x
            dy = tip.y - wrist.y
            distance = (dx**2 + dy**2) ** 0.5
            total_distance += distance

        avg_distance = total_distance / len(fingertips)

        # If average distance is small, it's a fist
        # Threshold: 0.15 (tune this if needed)
        return avg_distance < 0.15

    def _is_open_palm(self, hand_landmarks) -> bool:
        """Detect if hand is open (all fingers extended)."""
        # Get wrist position
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]

        # Get all fingertips
        fingertips = [
            hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP],
            hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP],
        ]

        # All fingertips should be far from wrist (extended)
        for tip in fingertips:
            dx = tip.x - wrist.x
            dy = tip.y - wrist.y
            distance = (dx**2 + dy**2) ** 0.5

            # If any finger is too close, not an open palm
            if distance < 0.2:
                return False

        return True

    def _get_hand_position(self, hand_landmarks) -> Tuple[float, float]:
        """Get normalized hand position relative to screen center. Uses WRIST position
        so it works even when fist is closed!

        Returns:
            Tuple of (x_offset, y_offset) normalized to -1 to 1
        """
        # Use WRIST as reference point (works for both fist and open hand)
        wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]

        # Convert to pixel coordinates
        x_pixel = wrist.x * self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        y_pixel = wrist.y * self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # Calculate offset from center, normalize
        x_offset = (x_pixel - self.screen_center[0]) / self.screen_center[0]
        y_offset = (y_pixel - self.screen_center[1]) / self.screen_center[1]

        return x_offset, y_offset

    def _smooth_position(self, x: float, y: float) -> Tuple[float, float]:
        """Apply temporal smoothing to position."""
        if not self.filtering_enabled:
            return x, y

        self.x_buffer.append(x)
        self.y_buffer.append(y)

        return np.mean(self.x_buffer), np.mean(self.y_buffer)

    def _position_to_command(
        self, x_offset: float, y_offset: float, is_fist: bool, is_palm: bool
    ) -> Tuple[Optional[str], float]:
        """Convert hand position to drone command. Now detects movement in BOTH axes
        independently!

        Returns:
            Tuple of (command, intensity)
        """
        pos_thresh = self.thresholds["position"]["threshold"]

        # Open palm = takeoff/land
        if is_palm:
            return self.commands["open_palm"], 0.0

        # Fist = follow mode - drone follows fist position
        if is_fist:
            # Check BOTH vertical and horizontal independently
            # Pick the direction with stronger displacement

            y_intensity = abs(y_offset)
            x_intensity = abs(x_offset)

            # Determine primary command based on which is stronger
            if y_intensity > pos_thresh and y_intensity >= x_intensity:
                # Vertical movement is dominant
                if y_offset < 0:
                    # Fist in upper part → forward
                    return self.commands["fist_up"], min(y_intensity, 1.0)
                else:
                    # Fist in lower part → backward
                    return self.commands["fist_down"], min(y_intensity, 1.0)

            elif x_intensity > pos_thresh and x_intensity > y_intensity:
                # Horizontal movement is dominant
                if x_offset < 0:
                    # Fist on left → strafe left
                    return self.commands["fist_left"], min(x_intensity, 1.0)
                else:
                    # Fist on right → strafe right
                    return self.commands["fist_right"], min(x_intensity, 1.0)

            else:
                # Fist near center = hover
                return self.commands["fist_center"], 0.0

        # Neither fist nor palm = ignore
        return None, 0.0

    def get_frame_and_command(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[str], float]:
        """Get camera frame and detected command.

        Returns:
            Tuple of (frame, command, intensity)
        """
        ret, frame = self.cap.read()
        if not ret:
            return None, None, 0.0

        # Flip for mirror effect
        frame = cv2.flip(frame, 1)

        # Convert to RGB for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        command = None
        intensity = 0.0
        gesture_name = "None"

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                    self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2),
                )

                # Detect gesture type
                is_fist = self._is_fist(hand_landmarks)
                is_palm = self._is_open_palm(hand_landmarks)

                if is_fist:
                    gesture_name = "FIST"
                elif is_palm:
                    gesture_name = "OPEN PALM"
                else:
                    gesture_name = "Other"

                # Get hand position
                x_offset, y_offset = self._get_hand_position(hand_landmarks)

                # Smooth position
                x_offset, y_offset = self._smooth_position(x_offset, y_offset)

                # Convert to command
                command, intensity = self._position_to_command(
                    x_offset, y_offset, is_fist, is_palm
                )

                # Draw info
                cv2.putText(
                    frame,
                    f"Gesture: {gesture_name}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Position: ({x_offset:.2f}, {y_offset:.2f})",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                if command:
                    cv2.putText(
                        frame,
                        f"Command: {command} ({intensity:.2f})",
                        (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )

                # Draw center crosshair and quadrants
                h, w = frame.shape[:2]
                cv2.drawMarker(
                    frame,
                    self.screen_center,
                    (255, 0, 0),
                    cv2.MARKER_CROSS,
                    30,
                    2,
                )
                cv2.line(frame, (w // 2, 0), (w // 2, h), (100, 100, 100), 1)
                cv2.line(frame, (0, h // 2), (w, h // 2), (100, 100, 100), 1)

                break  # Only process first hand

        return frame, command, intensity

    def release(self):
        """Release camera resources."""
        self.cap.release()
        cv2.destroyAllWindows()
