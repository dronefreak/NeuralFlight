"""High-level drone controller that abstracts command sending."""

from enum import Enum


class DroneCommand(Enum):
    """Available drone commands."""

    FORWARD = "forward"
    BACKWARD = "backward"
    STRAFE_LEFT = "strafe_left"
    STRAFE_RIGHT = "strafe_right"
    ROTATE_LEFT = "rotate_left"
    ROTATE_RIGHT = "rotate_right"
    HOVER = "hover"
    TAKEOFF = "takeoff"
    LAND = "land"


class DroneController:
    """High-level interface for controlling drones.

    Supports both simulated and real drones.
    """

    def __init__(self, drone):
        """Initialize controller with a drone instance.

        Args:
            drone: Drone instance (simulator or real drone)
        """
        self.drone = drone
        self._is_flying = False

    def move(self, direction: str, intensity: float = 1.0, duration: float = 0.1):
        """Move the drone in a direction.

        Args:
            direction: Direction to move (forward, backward, left, right, etc.)
            intensity: Movement intensity (0-1)
            duration: Command duration in seconds (for future use)
        """
        # Validate command
        try:
            cmd = DroneCommand(direction)
        except ValueError:
            print(f"Invalid command: {direction}")
            return

        # Send to drone
        self.drone.send_command(cmd.value, intensity)

    def takeoff(self):
        """Takeoff the drone."""
        if not self._is_flying:
            self.drone.takeoff()
            self._is_flying = True

    def land(self):
        """Land the drone."""
        if self._is_flying:
            self.drone.land()
            self._is_flying = False

    def hover(self):
        """Make drone hover in place."""
        self.move(DroneCommand.HOVER.value)

    def emergency_stop(self):
        """Emergency stop - immediately land."""
        self.land()

    @property
    def is_flying(self) -> bool:
        """Check if drone is flying."""
        return self._is_flying
