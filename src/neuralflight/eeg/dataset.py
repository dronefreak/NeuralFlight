"""Download and prepare PhysioNet Motor Movement/Imagery Dataset."""

from pathlib import Path
from typing import List

import mne
import numpy as np
from tqdm import tqdm


class PhysioNetDataset:
    """PhysioNet Motor Movement/Imagery Dataset handler.

    Dataset info:
    - 109 subjects
    - 64-channel EEG
    - Motor imagery tasks: left hand, right hand, both feet, both fists

    Run structure (per subject, 14 runs total):
    - Runs 1-2: Baseline (eyes open/closed)
    - Runs 3,7,11: ACTUAL left/right fist movement
    - Runs 4,8,12: IMAGINED left/right fist movement (USE THESE!)
    - Runs 5,9,13: ACTUAL both fists/feet movement
    - Runs 6,10,14: IMAGINED both fists/feet movement

    For left vs right hand motor imagery, use runs [4, 8, 12]

    Event codes:
    - T0: Rest
    - T1: Left fist (runs 3,4,7,8,11,12) or both fists (other runs)
    - T2: Right fist (runs 3,4,7,8,11,12) or both feet (other runs)
    """

    def __init__(self, data_dir: str = "data/raw/physionet"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Event codes
        self.event_id = {
            "T0": 0,  # Rest
            "T1": 1,  # Left fist
            "T2": 2,  # Right fist
        }

    def download_subject(self, subject_id: int, runs: List[int] = None) -> bool:
        """Download data for a single subject.

        Args:
            subject_id: Subject number (1-109)
            runs: List of run numbers to download (default: motor imagery runs)

        Returns:
            True if successful
        """
        if runs is None:
            runs = [4, 8, 12]  # Motor IMAGERY of left/right hand

        print(f"Downloading subject {subject_id:03d}...")

        try:
            for run in runs:
                # MNE will handle the download automatically
                mne.datasets.eegbci.load_data(
                    subject_id, runs=[run], path=str(self.data_dir.parent)
                )
            return True
        except Exception as e:
            print(f"Error downloading subject {subject_id}: {e}")
            return False

    def load_subject(
        self,
        subject_id: int,
        runs: List[int] = None,
        channels: List[str] = None,
    ) -> tuple:
        """Load and preprocess data for a subject.

        Args:
            subject_id: Subject number
            runs: Runs to load
            channels: Specific channels to use (default: C3, Cz, C4)

        Returns:
            Tuple of (epochs, labels)
        """
        if runs is None:
            runs = [4, 8, 12]  # Motor IMAGERY runs

        if channels is None:
            channels = ["C3", "Cz", "C4"]  # Motor cortex channels

        # Load raw data
        raw_files = mne.datasets.eegbci.load_data(
            subject_id, runs=runs, path=str(self.data_dir.parent)
        )

        # Read and concatenate runs
        raw_list = [
            mne.io.read_raw_edf(f, preload=True, verbose=False) for f in raw_files
        ]
        raw = mne.concatenate_raws(raw_list)

        # Standardize channel names
        mne.datasets.eegbci.standardize(raw)

        # Pick motor cortex channels
        raw.pick_channels(channels, ordered=True)

        # Find events - let MNE extract the actual event IDs
        events, event_id = mne.events_from_annotations(raw, verbose=False)

        # Print available events for debugging
        print(f"  Available events: {event_id}")

        # We want T1 (left fist/hand) and T2 (right fist/hand)
        # Filter event_id to only include T1 and T2
        motor_imagery_events = {k: v for k, v in event_id.items() if k in ["T1", "T2"]}

        if not motor_imagery_events:
            raise ValueError(
                f"No T1 or T2 events found. Available: {list(event_id.keys())}"
            )

        # Create epochs (3 second windows)
        epochs = mne.Epochs(
            raw,
            events,
            event_id=motor_imagery_events,
            tmin=0.0,
            tmax=3.0,
            baseline=None,
            preload=True,
            verbose=False,
        )

        # Get data and labels
        X = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
        y = epochs.events[:, -1]  # Event codes

        return X, y

    def download_multiple_subjects(
        self, subject_ids: List[int], runs: List[int] = None
    ) -> int:
        """Download data for multiple subjects.

        Args:
            subject_ids: List of subject IDs
            runs: Runs to download

        Returns:
            Number of successfully downloaded subjects
        """
        success_count = 0
        for subject_id in tqdm(subject_ids, desc="Downloading subjects"):
            if self.download_subject(subject_id, runs):
                success_count += 1
        return success_count


def preprocess_eeg(
    X: np.ndarray, lowcut: float = 8.0, highcut: float = 30.0, fs: float = 160.0
) -> np.ndarray:
    """Apply bandpass filter to EEG data.

    Args:
        X: EEG data (n_epochs, n_channels, n_times)
        lowcut: Low cutoff frequency (Hz)
        highcut: High cutoff frequency (Hz)
        fs: Sampling rate (Hz)

    Returns:
        Filtered EEG data
    """
    from scipy.signal import butter, filtfilt

    # Design bandpass filter
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(5, [low, high], btype="band")

    # Apply filter to each epoch and channel
    X_filtered = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X_filtered[i, j, :] = filtfilt(b, a, X[i, j, :])

    return X_filtered
