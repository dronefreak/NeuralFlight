# Architecture Overview

## System Design

The NeuralFlight Drone Control framework follows a modular, layered architecture:

```
┌─────────────────────────────────────────────────┐
│            User Interface Layer                  │
│  (Demos: head_gesture_demo.py, motor_imagery_   │
│   demo.py)                                       │
└─────────────────┬───────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────┐
│         Control Abstraction Layer                │
│  (DroneController - unified command interface)   │
└─────────────────┬───────────────────────────────┘
                  │
         ┌────────┴────────┐
         │                 │
┌────────▼─────┐  ┌────────▼─────────────────────┐
│  Input Layer │  │   Output Layer                │
│              │  │  (DroneSimulator/Real Drone)  │
│  • Gestures  │  └───────────────────────────────┘
│  • EEG       │
│  • Other     │
└──────────────┘
```

## Core Components

### 1. Simulator (`src/neuralflight/simulator/`)

**Purpose:** Provides a physics-based drone simulation with real-time visualization.

**Key Classes:**

- `DroneSimulator`: Main simulator class
  - Pygame-based rendering
  - Physics engine (velocity, drag, acceleration)
  - Command processing
  - HUD overlay

**Features:**

- Realistic drone physics
- Trail visualization
- Real-time status display
- Configurable via YAML

### 2. Controllers (`src/neuralflight/controllers/`)

**Purpose:** High-level abstraction for drone control.

**Key Classes:**

- `DroneController`: Unified control interface
  - Command validation
  - State management (flying/landed)
  - Works with both simulated and real drones

**Design Pattern:** Adapter pattern - same interface works for different drone types.

### 3. Gesture Input (`src/neuralflight/gestures/`)

**Purpose:** Detect head movements using computer vision.

**Key Classes:**

- `HeadGestureDetector`: Mediapipe-based head tracking
  - Face mesh detection (468 landmarks)
  - Pose estimation (pitch, yaw, roll)
  - Temporal smoothing
  - Gesture classification

**Pipeline:**

```
Camera → Face Detection → Landmark Extraction →
Pose Calculation → Smoothing → Gesture Classification → Command
```

### 4. EEG Processing (`src/neuralflight/eeg/`)

**Purpose:** Load, preprocess, and classify EEG motor imagery data.

**Key Classes:**

- `PhysioNetDataset`: Dataset handler
  - Automatic downloading via MNE
  - Multi-subject loading
  - Event extraction
- `preprocess_eeg()`: Signal processing
  - Bandpass filtering (8-30 Hz)
  - Artifact removal (future work)

**Pipeline:**

```
Raw EDF → MNE Loader → Channel Selection →
Epoching → Filtering → Feature Extraction → Classification
```

### 5. Models (`src/neuralflight/models/`)

**Purpose:** Deep learning models for EEG classification.

**Key Classes:**

- `EEGNet`: Compact CNN architecture
  - Temporal convolution (frequency learning)
  - Depthwise spatial convolution (channel learning)
  - Separable convolution (pattern learning)
  - ~8K parameters
- `EEGClassifier`: Training and inference wrapper
  - Training loop
  - Evaluation metrics
  - Model persistence

**Architecture Details:**

```
Input: (batch, channels, time_samples)
  ↓
[Temporal Conv 1D] → [Batch Norm] → [ELU]
  ↓
[Depthwise Conv] → [Batch Norm] → [ELU] → [Avg Pool] → [Dropout]
  ↓
[Separable Conv] → [Batch Norm] → [ELU] → [Avg Pool] → [Dropout]
  ↓
[Flatten] → [Dense] → Output: (batch, n_classes)
```

### 6. Utilities (`src/neuralflight/utils/`)

**Purpose:** Shared utilities and helpers.

**Key Functions:**

- `load_config()`: YAML configuration loader
- `get_project_root()`: Path resolution
- Logging utilities (future)
- Visualization helpers (future)

## Data Flow

### Head Gesture Control

```
Webcam Frame
  ↓
Mediapipe Face Mesh
  ↓
Head Pose (pitch, yaw, roll)
  ↓
Temporal Smoothing
  ↓
Gesture Classification
  ↓
Drone Command
  ↓
Simulator Update
  ↓
Display
```

### Motor Imagery Control

```
EEG Dataset (PhysioNet)
  ↓
Preprocessing (filtering, epoching)
  ↓
Training Data → EEGNet Model → Trained Checkpoint
  ↓
Test Epoch → Model Inference → Predicted Class
  ↓
Command Mapping
  ↓
Drone Command
  ↓
Simulator Update
```

## Configuration Management

All components are configurable via YAML files in `config/`:

- **drone_config.yaml**: Simulator physics, display settings
- **gesture_config.yaml**: Head tracking thresholds, camera setup
- **eeg_config.yaml**: Signal processing params, model hyperparameters

Benefits:

- Easy parameter tuning without code changes
- Separate configs for different scenarios
- Version control friendly

## Extensibility

### Adding New Input Methods

1. Create new module in `src/neuralflight/`
2. Implement input detector class
3. Return commands in standard format
4. Plug into `DroneController`

Example:

```python
class VoiceCommandDetector:
    def get_command(self):
        # Voice recognition logic
        return "forward", confidence
```

### Adding Real Drone Support

1. Create adapter in `src/neuralflight/adapters/`
2. Implement same interface as `DroneSimulator`
3. Handle hardware communication
4. Use with existing `DroneController`

Example:

```python
class TelloDrone:
    def send_command(self, cmd, intensity):
        # Send to real Tello drone
        self.tello.send_rc_control(...)
```

## Testing Strategy

### Current State

- Manual testing via demos
- Visual inspection of simulator
- Model accuracy metrics

### Future Work

- Unit tests for each module
- Integration tests for full pipeline
- Mock objects for hardware
- Continuous integration

## Performance Considerations

### Head Gesture Detection

- **Latency**: ~30ms per frame (30 FPS)
- **CPU Usage**: Moderate (Mediapipe is optimized)
- **Bottleneck**: Camera capture

### EEG Classification

- **Latency**: ~5ms per inference (CPU)
- **Model Size**: 32KB (EEGNet checkpoint)
- **Training Time**: ~5-10 minutes (CPU), ~2 minutes (GPU)

### Simulator

- **Frame Rate**: 60 FPS target
- **CPU Usage**: Low (Pygame is efficient)
- **Bottleneck**: None for single drone

## Security Considerations

- No network communication required
- All processing done locally
- No PII stored (EEG data is anonymized)
- Config files don't contain secrets

## Future Architecture Improvements

1. **Plugin System**: Dynamic loading of input methods
2. **Web Interface**: Flask/FastAPI backend + React frontend
3. **Distributed Training**: Multi-GPU support for larger models
4. **Real-time Streaming**: Support for live EEG devices
5. **Swarm Control**: Multiple drones from single BCI
6. **Cloud Deployment**: Containerized with Docker

## References

- **EEGNet**: Lawhern et al. (2018) - Compact CNN for BCI
- **Mediapipe**: Google's ML solutions for face tracking
- **MNE-Python**: Standard library for EEG processing
- **PyTorch**: Deep learning framework
