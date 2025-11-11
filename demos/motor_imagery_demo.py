#!/usr/bin/env python3
"""Motor Imagery EEG Drone Control Demo.

Uses real EEG motor imagery data to control a simulated drone.
Demonstrates classification of left/right hand motor imagery.

Controls:
- SPACE: Takeoff/Land
- ESC: Exit
- R: Load random EEG epoch and classify
"""

import time

import numpy as np
import pygame
import torch

from neuralflight.controllers.drone_controller import DroneController
from neuralflight.eeg.dataset import PhysioNetDataset, preprocess_eeg
from neuralflight.models.eegnet import EEGClassifier, EEGNet
from neuralflight.simulator.drone_sim import DroneSimulator
from neuralflight.utils.config_loader import get_project_root, load_config


class MotorImageryController:
    """Controls drone using motor imagery predictions."""

    def __init__(self, classifier, command_mapping):
        self.classifier = classifier
        self.command_mapping = command_mapping
        self.last_prediction = None
        self.last_confidence = None

    def predict_and_command(self, eeg_epoch):
        """Predict motor imagery class and return drone command.

        Args:
            eeg_epoch: EEG data (channels, time_samples)

        Returns:
            Tuple of (command, confidence)
        """
        # Convert to tensor
        X = torch.FloatTensor(eeg_epoch)

        # Predict
        pred_class, probs = self.classifier.predict(X)
        pred_class = pred_class[0]
        confidence = probs[0, pred_class]

        # Map to command
        command = self.command_mapping.get(pred_class, "hover")

        self.last_prediction = pred_class
        self.last_confidence = confidence

        return command, confidence


def load_test_data(config):
    """Load test EEG data."""
    print("Loading test EEG data...")

    dataset_config = config["dataset"]
    preprocess_config = config["preprocessing"]

    # Use a subject not in training set for testing
    test_subject = 6
    data_dir = get_project_root() / "data" / "raw" / "physionet"
    dataset = PhysioNetDataset(str(data_dir))

    # Download if needed
    dataset.download_subject(test_subject, dataset_config["runs"])

    # Load data
    X, y = dataset.load_subject(
        test_subject,
        dataset_config["runs"],
        preprocess_config["channels"],
    )

    # Filter to keep only left hand (T1) and right hand (T2)
    mask = (y == 1) | (y == 2)
    X = X[mask]
    y = y[mask]

    # Remap labels
    y = y - 1

    # Apply preprocessing
    X = preprocess_eeg(
        X,
        lowcut=preprocess_config["lowcut"],
        highcut=preprocess_config["highcut"],
        fs=preprocess_config["sampling_rate"],
    )

    print(f"Loaded {len(X)} test epochs")
    return X, y


def main():
    print("=" * 60)
    print("🧠 MOTOR IMAGERY EEG DRONE CONTROL")
    print("=" * 60)
    print()
    print("This demo uses REAL EEG motor imagery data to control")
    print("a simulated drone. The model classifies brain signals")
    print("from imagined left/right hand movements.")
    print()
    print("CONTROLS:")
    print("  ⎵  SPACE  → Takeoff/Land")
    print("  R  R      → Load random EEG epoch and classify")
    print("  ⎋  ESC    → Exit")
    print()
    print("MOTOR IMAGERY MAPPING:")
    print("  🤚 Left hand  → Drone strafes LEFT")
    print("  🤚 Right hand → Drone strafes RIGHT")
    print()
    print("=" * 60)
    print()

    # Load configs
    drone_config = load_config("drone_config")
    eeg_config = load_config("eeg_config")

    # Initialize simulator
    print("Initializing drone simulator...")
    simulator = DroneSimulator(drone_config)
    controller = DroneController(simulator)

    # Load model
    print("Loading trained model...")
    model_path = (
        get_project_root()
        / eeg_config["training"]["savedir"]
        / eeg_config["training"]["savename"]
    )

    if not model_path.exists():
        print()
        print("ERROR: Model not found!")
        print("Please run 'python demos/train_model.py' first to train the model.")
        print()
        simulator.close()
        return

    # Load checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(model_path, map_location=device)
    model_cfg = checkpoint["model_config"]

    model = EEGNet(
        n_channels=model_cfg["n_channels"],
        n_classes=model_cfg["n_classes"],
        n_samples=model_cfg["n_samples"],
    )
    classifier = EEGClassifier(model, device)
    classifier.load(str(model_path))

    print(f"✓ Model loaded (device: {device})")

    # Load test data
    try:
        X_test, y_test = load_test_data(eeg_config)
    except Exception as e:
        print(f"Error loading test data: {e}")
        simulator.close()
        return

    # Initialize controller
    command_mapping = eeg_config["class_to_command"]
    mi_controller = MotorImageryController(classifier, command_mapping)

    print()
    print("✓ Ready! Press SPACE to takeoff, then press R to classify EEG data.")
    print()

    # UI overlay
    font = pygame.font.Font(None, 32)
    small_font = pygame.font.Font(None, 24)

    running = True
    current_epoch_idx = 0

    # Persistent display state
    current_prediction = None
    current_confidence = None
    current_ground_truth = None

    # Movement state
    movement_command = None
    movement_start_time = None
    movement_duration = 2.0  # seconds

    clock = pygame.time.Clock()

    try:
        while running:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        if controller.is_flying:
                            controller.land()
                        else:
                            controller.takeoff()
                    elif event.key == pygame.K_r:
                        # Classify random epoch
                        current_epoch_idx = np.random.randint(len(X_test))
                        epoch = X_test[current_epoch_idx]
                        true_label = y_test[current_epoch_idx]

                        command, confidence = mi_controller.predict_and_command(epoch)

                        # Store for persistent display
                        current_prediction = mi_controller.last_prediction
                        current_confidence = confidence
                        current_ground_truth = true_label

                        class_names = {0: "Left Hand", 1: "Right Hand"}
                        print(
                            f"Classified: {class_names[current_prediction]} "
                            f"(confidence: {confidence:.2%}) | "
                            f"True: {class_names[true_label]} | "
                            f"Command: {command}"
                        )

                        # Start movement if flying
                        if controller.is_flying:
                            movement_command = command
                            movement_start_time = time.time()
                            print(f"→ Executing {command} for {movement_duration}s...")

            # Execute ongoing movement command
            if movement_command is not None and movement_start_time is not None:
                elapsed = time.time() - movement_start_time
                if elapsed < movement_duration:
                    # Still executing command
                    controller.move(movement_command, intensity=0.6)
                else:
                    # Movement complete, hover
                    controller.hover()
                    movement_command = None
                    movement_start_time = None
                    print("→ Movement complete, hovering")
            elif controller.is_flying:
                # No active command, just hover
                controller.hover()

            # Update simulator (this clears screen and redraws drone)
            running = simulator.update()

            # Draw persistent EEG info overlay AFTER simulator update
            if current_prediction is not None:
                class_names = {0: "Left Hand", 1: "Right Hand"}

                # Background panel for better visibility
                panel_rect = pygame.Rect(5, 115, 350, 120)
                panel_surface = pygame.Surface((panel_rect.width, panel_rect.height))
                panel_surface.set_alpha(200)
                panel_surface.fill((20, 20, 40))
                simulator.screen.blit(panel_surface, panel_rect.topleft)

                # Prediction info
                pred_text = f"Prediction: {class_names[current_prediction]}"
                conf_text = f"Confidence: {current_confidence:.1%}"
                true_text = f"Ground Truth: {class_names[current_ground_truth]}"

                # Show active command
                if movement_command:
                    cmd_text = f"Executing: {movement_command.upper()}"
                else:
                    cmd_text = "Status: Hovering"

                # Color code: green if correct, red if wrong
                pred_color = (
                    (0, 255, 0)
                    if current_prediction == current_ground_truth
                    else (255, 100, 100)
                )

                # Render text
                pred_surface = font.render(pred_text, True, pred_color)
                conf_surface = small_font.render(conf_text, True, (0, 255, 255))
                true_surface = small_font.render(true_text, True, (200, 200, 200))
                cmd_surface = small_font.render(cmd_text, True, (255, 200, 0))

                simulator.screen.blit(pred_surface, (15, 125))
                simulator.screen.blit(conf_surface, (15, 160))
                simulator.screen.blit(true_surface, (15, 185))
                simulator.screen.blit(cmd_surface, (15, 210))

            # Draw instructions at bottom
            inst_surface = small_font.render(
                "Press R to classify EEG | SPACE to takeoff/land | ESC to exit",
                True,
                (200, 200, 200),
            )
            simulator.screen.blit(inst_surface, (10, simulator.height - 30))

            # Update display
            pygame.display.flip()

            # Control frame rate (60 FPS)
            clock.tick(60)

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        print("\nCleaning up...")
        simulator.close()
        print("Done!")


if __name__ == "__main__":
    main()
