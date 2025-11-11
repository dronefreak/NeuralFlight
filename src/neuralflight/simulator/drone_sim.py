"""Drone simulator with real-time visualization using pygame."""

import math
from dataclasses import dataclass

import numpy as np
import pygame


@dataclass
class DroneState:
    """Current state of the drone."""

    x: float
    y: float
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    rotation: float = 0.0  # Degrees
    is_flying: bool = False


class DroneSimulator:
    """Simulates a drone with physics and pygame visualization."""

    def __init__(self, config: dict):
        self.config = config
        pygame.init()

        # Display setup
        window_config = config["window"]
        self.width = window_config["width"]
        self.height = window_config["height"]
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(window_config["title"])
        self.clock = pygame.time.Clock()
        self.fps = window_config["fps"]

        # Drone physics
        drone_config = config["drone"]
        initial_pos = drone_config["initial_position"]
        self.state = DroneState(x=initial_pos[0], y=initial_pos[1])
        self.max_speed = drone_config["max_speed"]
        self.acceleration = drone_config["acceleration"]
        self.drag = drone_config["drag"]
        self.rotation_speed = drone_config["rotation_speed"]

        # Control settings
        control_config = config["control"]
        self.smoothing = control_config["smoothing"]

        # Visual settings
        self.colors = config["colors"]
        self.font = pygame.font.Font(None, 24)

        # Trail for visualization
        self.trail = []
        self.max_trail_length = 50

        # Current command
        self.current_command = None

    def send_command(self, command: str, intensity: float = 1.0):
        """Send a command to the drone.

        Args:
            command: One of 'forward', 'backward', 'strafe_left', 'strafe_right',
                    'rotate_left', 'rotate_right', 'hover', 'takeoff', 'land'
            intensity: Command intensity (0-1)
        """
        self.current_command = (command, intensity)

    def _apply_physics(self):
        """Update drone state based on current command and physics."""
        if not self.state.is_flying:
            return

        command, intensity = self.current_command or ("hover", 0.0)
        intensity = np.clip(intensity, 0.0, 1.0)

        # Calculate acceleration based on command
        accel_x, accel_y = 0.0, 0.0

        if command == "forward":
            accel_y = -self.acceleration * intensity
        elif command == "backward":
            accel_y = self.acceleration * intensity
        elif command == "strafe_left":
            accel_x = -self.acceleration * intensity
        elif command == "strafe_right":
            accel_x = self.acceleration * intensity
        elif command == "rotate_left":
            self.state.rotation -= self.rotation_speed * intensity
        elif command == "rotate_right":
            self.state.rotation += self.rotation_speed * intensity

        # Update velocity
        self.state.velocity_x += accel_x
        self.state.velocity_y += accel_y

        # Apply drag
        self.state.velocity_x *= self.drag
        self.state.velocity_y *= self.drag

        # Clamp to max speed
        speed = math.sqrt(self.state.velocity_x**2 + self.state.velocity_y**2)
        if speed > self.max_speed:
            scale = self.max_speed / speed
            self.state.velocity_x *= scale
            self.state.velocity_y *= scale

        # Update position
        self.state.x += self.state.velocity_x
        self.state.y += self.state.velocity_y

        # Keep within bounds (with wrapping)
        self.state.x = self.state.x % self.width
        self.state.y = self.state.y % self.height

        # Keep rotation in 0-360 range
        self.state.rotation = self.state.rotation % 360

        # Add to trail
        if len(self.trail) == 0 or (
            abs(self.trail[-1][0] - self.state.x) > 2
            or abs(self.trail[-1][1] - self.state.y) > 2
        ):
            self.trail.append((int(self.state.x), int(self.state.y)))
            if len(self.trail) > self.max_trail_length:
                self.trail.pop(0)

    def _draw_grid(self):
        """Draw background grid."""
        grid_color = tuple(self.colors["grid"])
        grid_size = 50

        for x in range(0, self.width, grid_size):
            pygame.draw.line(self.screen, grid_color, (x, 0), (x, self.height), 1)
        for y in range(0, self.height, grid_size):
            pygame.draw.line(self.screen, grid_color, (0, y), (self.width, y), 1)

    def _draw_trail(self):
        """Draw drone movement trail."""
        if len(self.trail) < 2:
            return

        trail_color = tuple(self.colors["trail"][:3])
        for i in range(len(self.trail) - 1):
            start_pos = self.trail[i]
            # pygame doesn't support alpha in lines easily, so we draw circles
            pygame.draw.circle(self.screen, trail_color, start_pos, 2)

    def _draw_drone(self):
        """Draw the drone."""
        if not self.state.is_flying:
            return

        drone_color = tuple(self.colors["drone"])
        pos = (int(self.state.x), int(self.state.y))

        # Draw drone body (circle)
        pygame.draw.circle(self.screen, drone_color, pos, 10)

        # Draw direction indicator (triangle)
        angle_rad = math.radians(self.state.rotation)
        front_x = pos[0] + 15 * math.cos(angle_rad)
        front_y = pos[1] + 15 * math.sin(angle_rad)
        pygame.draw.line(self.screen, drone_color, pos, (front_x, front_y), 3)

    def _draw_hud(self):
        """Draw heads-up display with drone info."""
        text_color = tuple(self.colors["text"])

        # Status
        status = "FLYING" if self.state.is_flying else "LANDED"
        status_text = self.font.render(f"Status: {status}", True, text_color)
        self.screen.blit(status_text, (10, 10))

        # Position
        pos_text = self.font.render(
            f"Position: ({int(self.state.x)}, {int(self.state.y)})",
            True,
            text_color,
        )
        self.screen.blit(pos_text, (10, 35))

        # Speed
        speed = math.sqrt(self.state.velocity_x**2 + self.state.velocity_y**2)
        speed_text = self.font.render(f"Speed: {speed:.1f}", True, text_color)
        self.screen.blit(speed_text, (10, 60))

        # Current command
        cmd = self.current_command[0] if self.current_command else "None"
        cmd_text = self.font.render(f"Command: {cmd}", True, text_color)
        self.screen.blit(cmd_text, (10, 85))

    def update(self) -> bool:
        """Update simulation by one frame.

        Returns:
            True if window is still open, False if closed
        """
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False
                elif event.key == pygame.K_SPACE:
                    self.state.is_flying = not self.state.is_flying
                    if not self.state.is_flying:
                        self.state.velocity_x = 0
                        self.state.velocity_y = 0

        # Update physics
        self._apply_physics()

        # Render
        bg_color = tuple(self.colors["background"])
        self.screen.fill(bg_color)
        self._draw_grid()
        self._draw_trail()
        self._draw_drone()
        self._draw_hud()

        pygame.display.flip()
        self.clock.tick(self.fps)

        return True

    def takeoff(self):
        """Takeoff the drone."""
        self.state.is_flying = True

    def land(self):
        """Land the drone."""
        self.state.is_flying = False
        self.state.velocity_x = 0
        self.state.velocity_y = 0

    def close(self):
        """Clean up resources."""
        pygame.quit()
