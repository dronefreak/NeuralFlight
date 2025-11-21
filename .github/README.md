# NeuralFlight

**From neurons to flight paths**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

An accessible framework for controlling drones using computer vision and brain signals. Control drones with hand gestures, head movements, or real EEG motor imagery. No expensive hardware required - just a webcam and Python.

> _Started as a BCI research project in 2018 (MATLAB + $800 EEG headset + $300 drone). Rebuilt from the ground up in 2025 with modern ML so anyone can try it with just a webcam._

**Original 2018 version:** [brain-computer-interface-for-drones](https://github.com/dronefreak/brain-computer-interface-for-drones)

---

## 🎮 Quick Demo (No Hardware Needed!)

### 🥇 Hand Gesture Control (Most Intuitive!)

Control a drone by **moving your fist** - like dragging a cursor!

```bash
python demos/hand_gesture_demo.py
```

<!-- ![Hand Gesture Demo](../docs/hand_gesture_demo.gif) -->

_✊ Move your fist around the screen, drone follows!_

**Gestures:**

- ✊ **Make a fist and move it** → Drone follows your fist position!
  - Move fist to **upper screen** → Drone moves forward
  - Move fist to **lower screen** → Drone moves backward
  - Move fist to **left side** → Drone strafes left
  - Move fist to **right side** → Drone strafes right
  - Keep fist in **center** → Drone hovers
- ✋ **Open palm** (5 fingers) → Takeoff/Land

**Tips:**

- Start with fist in center of screen
- Move fist slowly at first
- The further from center, the faster the drone moves

**How it works:** Uses [Mediapipe Hands](https://google.github.io/mediapipe/) to track your wrist position in real-time. Most natural and intuitive control method - no need to think about specific gestures!

---

### 🎭 Head Gesture Control (Hands-Free!)

Control the drone with **head movements** - no hands needed!

```bash
python demos/head_gesture_demo.py
```

<!-- ![Head Gesture Demo](docs/head_gesture_demo.gif) -->

_🗣️ Look down to fly forward, tilt head to strafe - hands-free control_

**Controls:**

- 👇 **Look down** → Drone moves forward
- 👆 **Look up** → Drone moves backward
- 👈 **Turn left** → Drone rotates left
- 👉 **Turn right** → Drone rotates right
- ↖️ **Tilt left** → Drone strafes left
- ↗️ **Tilt right** → Drone strafes right
- ⎵ **SPACE** → Takeoff/Land
- ⎋ **ESC** → Exit

**How it works:** Uses Mediapipe Face Mesh to track head pose (pitch/yaw/roll). Like BCI, but using neck muscles instead of reading brain signals. Quirky, effective, and great for accessibility.

---

### 🧠 Motor Imagery Control (Research-Oriented)

Train a PyTorch model on **real EEG data** and control the drone with imagined hand movements.

```bash
# 1. Download dataset and train model (takes ~10 minutes)
python demos/train_model.py

# 2. Run the demo
python demos/motor_imagery_demo.py
```

![Motor Imagery Demo](../docs/motor_imagery_demo.gif)

_🧠 Real brain signals controlling a drone - no physical movement required_

**Controls:**

- ⎵ **SPACE** → Takeoff/Land
- **R** → Classify random EEG epoch and execute command
- ⎋ **ESC** → Exit

**How it works:** Uses the [PhysioNet Motor Movement/Imagery Dataset](https://physionet.org/content/eegmmidb/1.0.0/) to train an EEGNet model with residual connections that classifies motor imagery (imagined left/right hand movements) from EEG signals. Achieves **73% cross-subject accuracy** with 17 channels.

---

## ✨ Features

- **🎯 Three Control Modes:**
  - **Hand Gesture**: Fist-following control (most intuitive)
  - **Head Gesture**: Hands-free control (accessibility)
  - **Motor Imagery**: Real EEG classification (research-grade, 73% accuracy)

- **🤖 Modern ML Stack:**
  - PyTorch 2.0+ with CUDA support
  - EEGNet with residual connections for BCI
  - Real-time inference (60 FPS)

- **🎮 Simulated Drone:**
  - Physics-based movement
  - Real-time visualization
  - No hardware required for testing

- **📊 Production Ready:**
  - Clean, modular architecture
  - Type hints throughout
  - Pre-commit hooks with Ruff
  - Comprehensive configs
  - Proper package installation (setup.py)

---

## 🚀 Installation

### Prerequisites

- Python 3.10 or higher
- Webcam (for hand/head gesture demos)
- ~2GB disk space (for EEG dataset)
- GPU recommended for EEG training (optional)

### Quick Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/NeuralFlight.git
cd NeuralFlight

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package (recommended!)
pip install -e .

# Or just install dependencies
pip install -r requirements.txt

# Install pre-commit hooks (for development)
pre-commit install
```

**That's it!** The package is now installed and you can run demos from anywhere.

See [INSTALLATION.md](INSTALLATION.md) for more details.

---

## 📖 Usage

### Method 1: Using Installed Commands

After `pip install -e .`, you can run from anywhere:

```bash
# Hand gesture control
neuralflight-hand

# Head gesture control
neuralflight-head

# Motor imagery EEG control
neuralflight-eeg

# Train EEG model
neuralflight-train
```

### Method 2: Using Python Scripts

```bash
# Hand gesture control
python demos/hand_gesture_demo.py

# Head gesture control
python demos/head_gesture_demo.py

# Train EEG model
python demos/train_model.py

# Motor imagery demo
python demos/motor_imagery_demo.py
```

---

## 🏗️ Architecture

```
NeuralFlight/
├── src/neuralflight/
│   ├── simulator/       # Pygame-based drone simulator
│   ├── gestures/        # Hand & head gesture detection (Mediapipe)
│   ├── eeg/            # EEG data loading and preprocessing
│   ├── models/         # PyTorch neural networks (EEGNet + Residual)
│   ├── controllers/    # High-level drone control abstraction
│   └── utils/          # Config loading, logging, etc.
├── demos/              # Runnable demos
├── config/             # YAML configuration files
├── notebooks/          # Jupyter notebooks for analysis
└── tests/              # Unit tests
```

### Key Components

**Simulator** (`src/neuralflight/simulator/drone_sim.py`)

- Physics-based drone movement
- Real-time pygame visualization
- Supports all standard drone commands

**Hand Gesture Detector** (`src/neuralflight/gestures/hand_detector.py`)

- Mediapipe Hands for tracking
- Wrist position tracking (works with closed fist!)
- Fist detection via fingertip-to-wrist distance
- Open palm detection for takeoff/land

**Head Gesture Detector** (`src/neuralflight/gestures/head_detector.py`)

- Mediapipe Face Mesh for pose estimation
- Temporal smoothing for stable control
- Configurable gesture thresholds

**EEG Pipeline** (`src/neuralflight/eeg/`)

- PhysioNet dataset downloader
- Bandpass filtering (8-30 Hz for motor imagery)
- MNE-Python integration
- Subject-level train/val split (prevents overfitting)

**EEGNet with Residuals** (`src/neuralflight/models/eegnet_residual.py`)

- Compact CNN for EEG classification
- Based on Lawhern et al. (2018) + residual connections
- ~10K parameters, trains in 10-15 minutes
- 73% cross-subject accuracy (17 channels)

---

## 🎓 Background

This project is a complete modernization of my original 2018 BCI drone controller, which used:

- MATLAB for signal processing
- Node.js for drone control
- Emotiv EPOC+ headset ($800)
- AR Parrot drone ($300)

**What's changed in 2025:**

- ✅ Pure Python (no MATLAB)
- ✅ PyTorch for deep learning
- ✅ Modern EEG processing with MNE
- ✅ Accessible demos without expensive hardware
- ✅ Clean, maintainable architecture
- ✅ Open-source datasets
- ✅ Residual connections for better accuracy
- ✅ Proper package installation

---

## 🧪 Technical Details

### Motor Imagery Classification

**Dataset:** [PhysioNet Motor Movement/Imagery](https://physionet.org/content/eegmmidb/1.0.0/)

- 109 subjects, 64-channel EEG
- Motor imagery tasks: left hand, right hand, feet, fists
- 160 Hz sampling rate

**Preprocessing:**

- Bandpass filter: 8-30 Hz (alpha/beta bands)
- Channels: 17 (FC3-FC4, C5-C6, CP3-CP4 - full motor cortex coverage)
- Epoch length: 3 seconds
- Subject-level split (19 train, 5 validation)

**Model:** EEGNet with Residual Connections

- Architecture: Temporal conv → Depthwise spatial conv → Separable conv (with skip connection) → FC layers
- Parameters: ~10,000
- Training: ~100 epochs, Adam optimizer
- **Accuracy: 73% cross-subject validation** (excellent for 17 channels!)
- Training time: ~15 minutes on GPU, ~45 minutes on CPU

**Why 73% is good:**

- Cross-subject (not person-specific) = harder task
- 17 channels (not full 64-channel cap) = less data
- Motor imagery (imagined movement) = weak signals
- Comparable to published research papers

### Hand Gesture Detection

**Tracking:** Mediapipe Hands

- 21 hand landmarks
- Tracks wrist position (not just fingertips!)

**Gesture Recognition:**

- **Fist detection**: Measures average fingertip-to-wrist distance
  - Distance < 0.15 = fist detected
- **Open palm detection**: All fingers extended far from wrist
  - Distance > 0.2 for all fingers = palm detected

**Control Mapping:**

- Wrist position relative to screen center
- Deadzone threshold: 0.10 (configurable)
- Movement intensity scales with distance from center

**Filtering:**

- Temporal smoothing (5-frame window)
- Adaptive dimension matching for edge cases

### Head Gesture Detection

**Tracking:** Mediapipe Face Mesh

- 468 facial landmarks
- Key points: nose, eyes, chin
- Calculates pitch, yaw, roll angles

**Gesture Mapping:**

- Pitch (±10°) → Forward/Backward
- Yaw (±15°) → Rotate Left/Right
- Roll (±10°) → Strafe Left/Right

**Filtering:**

- Temporal smoothing (5-frame window)
- Configurable dead zones

---

## 🎯 Use Cases

This framework is useful for:

- **Research:** Rapid prototyping of BCI algorithms
- **Education:** Teaching ML, signal processing, and robotics
- **Accessibility:** Alternative control methods for users with motor impairments
- **Autonomous Systems:** Intent detection, attention monitoring
- **Portfolio:** Demonstrating ML/robotics/neuroscience skills
- **Hackathons:** Quick BCI demos and prototypes

---

## 🛠️ Configuration

All settings are in YAML files under `config/`:

- `drone_config.yaml` - Simulator physics, display settings
- `hand_config.yaml` - Hand gesture thresholds, camera settings
- `gesture_config.yaml` - Head gesture thresholds
- `eeg_config.yaml` - Signal processing, model hyperparameters

Example: Adjust hand gesture sensitivity

```yaml
# config/hand_config.yaml
gestures:
  position:
    threshold: 0.10 # Lower = more sensitive (was 0.15)
```

Example: Use more subjects for training

```yaml
# config/eeg_config.yaml
dataset:
  train_subjects: [1, 2, 3, 4, 6, 7, 8, 9, 10, ..., 40] # More subjects
  val_subjects: [41, 42, 43, 44, 45]
```

---

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

This project uses:

- **Ruff** for linting and formatting
- **Pre-commit** hooks for code quality
- **PyTorch** for deep learning
- **Type hints** throughout

```bash
# Set up development environment
pip install -e ".[dev]"
pre-commit install

# Format code
black src/
ruff check src/
```

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before contributing.

---

## 🔒 Security

For security concerns, please see [SECURITY.md](SECURITY.md) for our vulnerability reporting process.

**Safety Note:** NeuralFlight is designed for research and education. Always test in controlled environments. Do not use for safety-critical applications without extensive validation.

---

## 📚 References

- Lawhern et al. (2018). "[EEGNet: A Compact Convolutional Network for EEG-based Brain-Computer Interfaces](https://arxiv.org/abs/1611.08024)"
- Schalk et al. (2004). "BCI2000: A General-Purpose Brain-Computer Interface System"
- Goldberger et al. (2000). "PhysioBank, PhysioToolkit, and PhysioNet"
- He et al. (2016). "Deep Residual Learning for Image Recognition" (Residual connections)

---

## 📄 License

Apache License 2.0 - see [LICENSE](LICENSE) file for details.

This means you can:

- ✅ Use commercially
- ✅ Modify and distribute
- ✅ Patent use
- ✅ Private use

With conditions:

- ⚠️ Include license and copyright notice
- ⚠️ State changes made
- ⚠️ Include NOTICE file if present

---

## 👨‍💻 Author

**Saumya Saksena**
Originally created in 2018, modernized in 2025.

- [GitHub](https://github.com/dronefreak)
- [Email](mailto:kumaar324@gmail.com)

---

If you find this project useful, consider giving it a star! It helps others discover the project.

---

## 🔮 Future Work

- [ ] Support for real drone hardware (DJI Tello, CrazyFlie)
- [ ] Multi-class motor imagery (4+ classes for 3D control)
- [ ] Real-time EEG streaming from consumer headsets (Muse, OpenBCI)
- [ ] Web dashboard for remote control
- [ ] Reinforcement learning for autonomous navigation
- [ ] Multi-modal control (voice + BCI + gestures)
- [ ] Mobile app for gesture control
- [ ] VR/AR visualization
- [ ] Multi-drone swarm control

---

## 🙏 Acknowledgments

- PhysioNet for the EEG dataset
- Google Mediapipe team for computer vision tools
- PyTorch team for the deep learning framework
- Original EEGNet paper authors
- All contributors and users!

---
