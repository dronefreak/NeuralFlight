#!/usr/bin/env python3
"""Setup script for NeuralFlight package."""

from pathlib import Path

from setuptools import find_packages, setup

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip()
        for line in requirements_file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="neuralflight",
    version="2.0.0",
    description="Neural control framework for drones - hand gestures, head tracking, and EEG",  # noqa E501
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Likith Reddy",
    author_email="likith012@gmail.com",
    url="https://github.com/dronefreak/NeuralFlight",
    project_urls={
        "Documentation": "https://github.com/dronefreak/NeuralFlight#readme",
        "Source": "https://github.com/dronefreak/NeuralFlight",
        "Original Project": "https://github.com/dronefreak/brain-computer-interface-for-drones",  # noqa E501
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "pre-commit>=3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "neuralflight-hand=neuralflight.demos.hand_gesture_demo:main",
            "neuralflight-head=neuralflight.demos.head_gesture_demo:main",
            "neuralflight-eeg=neuralflight.demos.motor_imagery_demo:main",
            "neuralflight-train=neuralflight.demos.train_model:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Human Machine Interfaces",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords=[
        "bci",
        "brain-computer-interface",
        "eeg",
        "motor-imagery",
        "drone-control",
        "gesture-recognition",
        "computer-vision",
        "deep-learning",
        "pytorch",
        "mediapipe",
    ],
    include_package_data=True,
    zip_safe=False,
)
