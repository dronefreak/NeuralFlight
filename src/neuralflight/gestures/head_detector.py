"""Head gesture detection using Mediapipe Face Mesh."""

from collections import deque
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np


class HeadGestureDetector:
    """Detects head gestures (pitch, yaw, roll) using Mediapipe Face Mesh."""

    def __init__(self, config: dict):
        self.config = config

        # Initialize Mediapipe
        mp_config = config["mediapipe"]
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            min_detection_confidence=mp_config["min_detection_confidence"],
            min_tracking_confidence=mp_config["min_tracking_confidence"],
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
        self.min_confidence = filter_config["min_confidence"]

        # Smoothing buffers
        self.pitch_buffer = deque(maxlen=self.window_size)
        self.yaw_buffer = deque(maxlen=self.window_size)
        self.roll_buffer = deque(maxlen=self.window_size)

        # Key landmark indices for pose estimation
        self.NOSE_TIP = 1
        self.CHIN = 152
        self.LEFT_EYE = 33
        self.RIGHT_EYE = 263
        self.LEFT_MOUTH = 61
        self.RIGHT_MOUTH = 291

    def _calculate_head_pose(
        self, landmarks, image_shape
    ) -> Tuple[float, float, float]:
        """Calculate head pose angles (pitch, yaw, roll).

        Returns:
            Tuple of (pitch, yaw, roll) in degrees
        """
        h, w = image_shape[:2]

        # Get 2D landmark positions
        nose = np.array(
            [
                landmarks[self.NOSE_TIP].x * w,
                landmarks[self.NOSE_TIP].y * h,
            ]
        )
        chin = np.array([landmarks[self.CHIN].x * w, landmarks[self.CHIN].y * h])
        left_eye = np.array(
            [
                landmarks[self.LEFT_EYE].x * w,
                landmarks[self.LEFT_EYE].y * h,
            ]
        )
        right_eye = np.array(
            [
                landmarks[self.RIGHT_EYE].x * w,
                landmarks[self.RIGHT_EYE].y * h,
            ]
        )

        # Calculate pitch (nodding up/down)
        # Positive = looking up, Negative = looking down
        pitch = np.arctan2(chin[1] - nose[1], chin[0] - nose[0])
        pitch = np.degrees(pitch) - 90

        # Calculate yaw (turning left/right)
        # Use eye positions
        face_center_x = (left_eye[0] + right_eye[0]) / 2
        yaw = (nose[0] - face_center_x) / w * 60

        # Calculate roll (tilting head)
        eye_angle = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
        roll = np.degrees(eye_angle)

        return pitch, yaw, roll

    def _smooth_angles(
        self, pitch: float, yaw: float, roll: float
    ) -> Tuple[float, float, float]:
        """Apply temporal smoothing to angles."""
        if not self.filtering_enabled:
            return pitch, yaw, roll

        self.pitch_buffer.append(pitch)
        self.yaw_buffer.append(yaw)
        self.roll_buffer.append(roll)

        return (
            np.mean(self.pitch_buffer),
            np.mean(self.yaw_buffer),
            np.mean(self.roll_buffer),
        )

    def _angles_to_command(
        self, pitch: float, yaw: float, roll: float
    ) -> Optional[str]:
        """Convert head pose angles to drone command.

        Args:
            pitch: Pitch angle in degrees
            yaw: Yaw angle in degrees
            roll: Roll angle in degrees

        Returns:
            Command string or None
        """
        # Check pitch (nodding)
        pitch_thresh = self.thresholds["pitch"]
        if pitch < pitch_thresh["forward_threshold"]:
            return self.commands["pitch_down"]
        elif pitch > pitch_thresh["backward_threshold"]:
            return self.commands["pitch_up"]

        # Check yaw (turning)
        yaw_thresh = self.thresholds["yaw"]
        if yaw < yaw_thresh["left_threshold"]:
            return self.commands["yaw_left"]
        elif yaw > yaw_thresh["right_threshold"]:
            return self.commands["yaw_right"]

        # Check roll (tilting)
        roll_thresh = self.thresholds["roll"]
        if roll < roll_thresh["left_threshold"]:
            return self.commands["roll_left"]
        elif roll > roll_thresh["right_threshold"]:
            return self.commands["roll_right"]

        return None

    def get_frame_and_command(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Get camera frame and detected command.

        Returns:
            Tuple of (frame, command) where command can be None
        """
        ret, frame = self.cap.read()
        if not ret:
            return None, None

        # Flip for mirror effect
        frame = cv2.flip(frame, 1)

        # Convert to RGB for Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        command = None

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Calculate head pose
                pitch, yaw, roll = self._calculate_head_pose(
                    face_landmarks.landmark, frame.shape
                )

                # Smooth angles
                pitch, yaw, roll = self._smooth_angles(pitch, yaw, roll)

                # Convert to command
                command = self._angles_to_command(pitch, yaw, roll)

                # Draw face mesh
                self.mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    self.mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=(0, 255, 0), thickness=1
                    ),
                )

                # Draw angles on frame
                cv2.putText(
                    frame,
                    f"Pitch: {pitch:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Yaw: {yaw:.1f}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Roll: {roll:.1f}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                if command:
                    cv2.putText(
                        frame,
                        f"Command: {command}",
                        (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                    )

        return frame, command

    def release(self):
        """Release camera resources."""
        self.cap.release()
        cv2.destroyAllWindows()
