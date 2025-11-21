# Contributing to NeuralFlight

Thank you for your interest in contributing to NeuralFlight! 🧠✈️

We welcome contributions of all kinds: bug reports, feature requests, documentation improvements, and code contributions.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Submitting Changes](#submitting-changes)
- [Community](#community)

## Code of Conduct

This project adheres to a Code of Conduct (see [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)). By participating, you are expected to uphold this code.

## How Can I Contribute?

### 🐛 Reporting Bugs

Before creating a bug report:

- Check the [existing issues](https://github.com/dronefreak/NeuralFlight/issues) to avoid duplicates
- Try the latest version to see if the issue persists

**Good bug reports include:**

- Clear, descriptive title
- Steps to reproduce
- Expected vs. actual behavior
- System information (OS, Python version, GPU/CPU)
- Error messages and logs
- Screenshots/GIFs if applicable

**Example:**

```markdown
**Title**: Motor imagery demo crashes on macOS with M1

**Description**:
When running `neuralflight-eeg`, the demo crashes immediately after loading the model.

**Steps to reproduce**:

1. Install on macOS 13.0 (M1 chip)
2. Run `pip install -e .`
3. Run `neuralflight-eeg`
4. Program crashes with error: ...

**Expected**: Demo should start and display window
**Actual**: Crashes with PyTorch MPS error

**System**:

- OS: macOS 13.0
- Python: 3.10.8
- PyTorch: 2.0.1
- Chip: Apple M1
```

### 💡 Suggesting Features

We love new ideas! Before suggesting:

- Check if it's already been proposed
- Consider if it fits the project's scope (neural control for autonomous systems)

**Good feature requests include:**

- Clear use case
- Why this feature is valuable
- Proposed implementation (if applicable)
- Alternatives you've considered

**Example:**

```markdown
**Feature**: Add support for 4-class motor imagery (left/right hand + feet/rest)

**Use Case**: Enable more complex drone maneuvers (forward/back + left/right)

**Why**: Current 2-class only allows lateral movement. 4-class would enable:

- Forward/backward with feet imagery
- Left/right with hand imagery
- More natural 3D control

**Implementation Ideas**:

- Modify EEGNet output layer to 4 classes
- Update command mapping in config
- Add training option for runs [5,6,9,10,13,14]

**Alternatives**:

- Use head tracking for forward/backward (but defeats BCI-only purpose)
```

### 📝 Improving Documentation

Documentation improvements are always welcome:

- Fix typos or unclear explanations
- Add examples or tutorials
- Improve API documentation
- Translate documentation

Small fixes can be submitted directly. Larger changes should be discussed in an issue first.

### 🔬 Contributing Code

We welcome code contributions! Areas where we need help:

- Adding new gesture recognition methods
- Improving EEG preprocessing
- Supporting more EEG datasets
- Real drone adapters (DJI Tello, etc.)
- Performance optimizations
- Cross-platform testing
- Unit tests

## Development Setup

### 1. Fork and Clone

```bash
# Fork on GitHub, then clone your fork
git clone https://github.com/dronefreak/NeuralFlight.git
cd NeuralFlight
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install in Development Mode

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### 4. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

Branch naming conventions:

- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Adding tests

## Coding Standards

### Python Style

We follow **PEP 8** with some modifications:

- Line length: 88 characters (Black default)
- Use type hints for function signatures
- Docstrings for all public functions/classes

### Code Quality Tools

We use these tools (run automatically with pre-commit):

```bash
# Format code
black src/

# Lint code
ruff check src/

# Type checking (optional but encouraged)
mypy src/
```

### Docstring Format

Use **Google-style** docstrings:

```python
def predict_command(eeg_epoch: np.ndarray) -> Tuple[str, float]:
    """
    Predict drone command from EEG epoch.

    Args:
        eeg_epoch: EEG data of shape (n_channels, n_samples)

    Returns:
        Tuple of (command, confidence) where command is a string
        like "strafe_left" and confidence is 0-1

    Raises:
        ValueError: If eeg_epoch has wrong shape

    Example:
        >>> epoch = np.random.randn(3, 480)
        >>> cmd, conf = predict_command(epoch)
        >>> print(f"Command: {cmd}, Confidence: {conf:.2%}")
    """
    # Implementation
```

### Testing

While we don't have comprehensive tests yet (contributions welcome!), ensure:

- Your code runs without errors
- Existing demos still work
- Add manual test instructions in your PR

Future test structure:

```bash
pytest tests/              # Run all tests
pytest tests/test_eeg.py   # Run specific test file
pytest --cov=neuralflight  # With coverage
```

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): short description

Longer description if needed.

Fixes #123
```

**Types:**

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, no logic change)
- `refactor`: Code restructuring
- `perf`: Performance improvement
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples:**

```bash
feat(eeg): add 4-class motor imagery support

- Modified EEGNet to handle 4 output classes
- Updated training script for feet/rest imagery
- Added command mapping for forward/backward

Closes #45

---

fix(hand): resolve fist detection on low-light conditions

The distance threshold was too strict for dim lighting.
Increased threshold from 0.15 to 0.18.

Fixes #67

---

docs(readme): add troubleshooting section for macOS M1

Added common PyTorch MPS issues and workarounds.
```

## Submitting Changes

### Pull Request Process

1. **Update documentation** for any new features
2. **Test thoroughly** on your system
3. **Update CHANGELOG.md** if applicable
4. **Fill out the PR template** completely
5. **Request review** from maintainers

### PR Template

When opening a PR, include:

```markdown
## Description

Brief description of changes

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing

How did you test this?

- [ ] Tested locally
- [ ] Tested on Linux/Mac/Windows
- [ ] All demos work
- [ ] No regressions

## Checklist

- [ ] Code follows project style
- [ ] Documentation updated
- [ ] Pre-commit hooks pass
- [ ] No breaking changes (or documented)

## Screenshots (if applicable)

Add screenshots or GIFs showing the change

## Related Issues

Fixes #123
Related to #456
```

### Review Process

- Maintainers will review within 3-5 business days
- Address review comments promptly
- Be open to suggestions and feedback
- Once approved, we'll merge your PR!

### After Your PR is Merged

- Your contribution will be acknowledged in release notes
- You'll be added to the contributors list
- Thank you for making NeuralFlight better! 🎉

## Community

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: Questions, ideas, general chat
- **Email**: kumaar324@gmail.com for private matters

### Getting Help

Stuck? Here's how to get help:

1. Check [documentation](README.md)
2. Search [existing issues](https://github.com/dronefreak/NeuralFlight/issues)
3. Open a GitHub Discussion
4. Ask in your PR if related to your contribution

## Recognition

We value all contributions! Contributors are recognized:

- In release notes
- In the project README
- As GitHub contributors

## Project Priorities

Current focus areas:

1. **Stability**: Bug fixes and reliability
2. **Documentation**: Clear guides and examples
3. **Performance**: Optimization and efficiency
4. **Features**: New control methods and hardware support

## Questions?

Not sure where to start? Open a GitHub Discussion or reach out to maintainers. We're happy to help new contributors!

---

**Thank you for contributing to NeuralFlight! Together, we're making neural control accessible to everyone.** 🧠✈️
