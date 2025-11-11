#!/usr/bin/env python3
"""Train EEGNet model on PhysioNet Motor Imagery dataset."""

import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from neuralflight.eeg.dataset import PhysioNetDataset, preprocess_eeg
from neuralflight.models.eegnet import EEGClassifier, EEGNet
from neuralflight.models.eegnet_residual import EEGNetResidual
from neuralflight.utils.config_loader import get_project_root, load_config


def prepare_data(config: dict):
    """Download and prepare dataset with SUBJECT-LEVEL split."""
    print("=" * 60)
    print("PREPARING DATASET")
    print("=" * 60)

    dataset_config = config["dataset"]
    train_subjects = dataset_config["train_subjects"]
    val_subjects = dataset_config["val_subjects"]
    runs = dataset_config["runs"]

    print("\nSubject split (CRITICAL for preventing overfitting!):")
    print(f"  Training subjects: {train_subjects}")
    print(f"  Validation subjects: {val_subjects}")
    print(f"  Runs per subject: {runs}")

    # Initialize dataset
    data_dir = get_project_root() / "data" / "raw" / "physionet"
    dataset = PhysioNetDataset(str(data_dir))

    preprocess_config = config["preprocessing"]
    channels = preprocess_config["channels"]

    # Load training subjects
    print(f"\n{'='*60}")
    print("LOADING TRAINING SUBJECTS")
    print("=" * 60)
    train_X, train_y = [], []

    for subject_id in tqdm(train_subjects, desc="Training subjects"):
        try:
            # Download if needed
            dataset.download_subject(subject_id, runs)

            # Load data
            X, y = dataset.load_subject(subject_id, runs, channels)

            if X is None or len(X) == 0:
                print(f"  Subject {subject_id}: No data returned")
                continue

            unique_events = np.unique(y)
            print(f"  Subject {subject_id}: Event IDs {unique_events}")

            if len(unique_events) != 2:
                print(
                    f"  Subject {subject_id}:"
                    f" Expected 2 events, got {len(unique_events)}"
                )
                continue

            # Remap to 0, 1
            label_map = {unique_events[0]: 0, unique_events[1]: 1}
            y = np.array([label_map[label] for label in y])

            # Apply bandpass filter
            X = preprocess_eeg(
                X,
                lowcut=preprocess_config["lowcut"],
                highcut=preprocess_config["highcut"],
                fs=preprocess_config["sampling_rate"],
            )

            train_X.append(X)
            train_y.append(y)
            print(f"  Subject {subject_id}: ✓ {len(X)} epochs")

        except Exception as e:
            print(f"  Subject {subject_id}: ✗ {e}")
            continue

    # Load validation subjects
    print(f"\n{'='*60}")
    print("LOADING VALIDATION SUBJECTS")
    print("=" * 60)
    val_X, val_y = [], []

    for subject_id in tqdm(val_subjects, desc="Validation subjects"):
        try:
            # Download if needed
            dataset.download_subject(subject_id, runs)

            # Load data
            X, y = dataset.load_subject(subject_id, runs, channels)

            if X is None or len(X) == 0:
                print(f"  Subject {subject_id}: No data returned")
                continue

            unique_events = np.unique(y)
            print(f"Subject {subject_id}: Event IDs {unique_events}")

            if len(unique_events) != 2:
                print(
                    f"Subject {subject_id}: Expected 2 events, got {len(unique_events)}"
                )
                continue

            # Remap to 0, 1
            label_map = {unique_events[0]: 0, unique_events[1]: 1}
            y = np.array([label_map[label] for label in y])

            # Apply bandpass filter
            X = preprocess_eeg(
                X,
                lowcut=preprocess_config["lowcut"],
                highcut=preprocess_config["highcut"],
                fs=preprocess_config["sampling_rate"],
            )

            val_X.append(X)
            val_y.append(y)
            print(f"  Subject {subject_id}: ✓ {len(X)} epochs")

        except Exception as e:
            print(f"  Subject {subject_id}: ✗ {e}")
            continue

    # Check if we have data
    if len(train_X) == 0 or len(val_X) == 0:
        print("\n❌ ERROR: Need data for both training and validation!")
        print(f"Training subjects loaded: {len(train_X)}")
        print(f"Validation subjects loaded: {len(val_X)}")
        return None, None, None, None

    # Concatenate
    X_train = np.concatenate(train_X, axis=0)
    y_train = np.concatenate(train_y, axis=0)
    X_val = np.concatenate(val_X, axis=0)
    y_val = np.concatenate(val_y, axis=0)

    print(f"\n{'='*60}")
    print("DATASET SUMMARY")
    print("=" * 60)
    print("Training:")
    print(f"  Subjects: {len(train_X)} successfully loaded")
    print(f"  Epochs: {len(X_train)}")
    print(f"  Shape: {X_train.shape}")
    print(f"  Class distribution: {np.bincount(y_train)}")

    print("\nValidation:")
    print(f"  Subjects: {len(val_X)} successfully loaded")
    print(f"  Epochs: {len(X_val)}")
    print(f"  Shape: {X_val.shape}")
    print(f"  Class distribution: {np.bincount(y_val)}")

    print(f"\nTotal: {len(X_train) + len(X_val)} epochs")
    print("Classes: 0=left hand, 1=right hand")

    return X_train, y_train, X_val, y_val


def train_model(config: dict, X_train, y_train, X_val, y_val):
    """Train the model."""
    print("\n" + "=" * 60)
    print("TRAINING MODEL")
    print("=" * 60)

    print(f"\nTrain samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.LongTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.LongTensor(y_val)

    # Create data loaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset, batch_size=config["training"]["batch_size"], shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["training"]["batch_size"], shuffle=False
    )

    # Initialize model
    model_config = config["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Use residual version for better performance
    use_residual = model_config.get("use_residual", True)
    use_attention = model_config.get("use_attention", False)

    if use_residual:
        print(f"Using EEGNetResidual (attention: {use_attention})")
        model = EEGNetResidual(
            n_channels=model_config["input_channels"],
            n_classes=model_config["num_classes"] - 2,  # Only left/right
            n_samples=X_train.shape[2],
            dropout=model_config["dropout"],
            kernel_length=model_config["kernel_length"],
            use_attention=use_attention,
        )
    else:
        print("Using original EEGNet")
        model = EEGNet(
            n_channels=model_config["input_channels"],
            n_classes=model_config["num_classes"] - 2,
            n_samples=X_train.shape[2],
            dropout=model_config["dropout"],
            kernel_length=model_config["kernel_length"],
        )

    model = EEGNet(
        n_channels=model_config["input_channels"],
        n_classes=model_config["num_classes"] - 2,  # Only left/right for now
        n_samples=X_train.shape[2],
        dropout=model_config["dropout"],
        kernel_length=model_config["kernel_length"],
    )

    classifier = EEGClassifier(model, device)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    patience = config["training"]["early_stopping_patience"]

    for epoch in range(config["training"]["num_epochs"]):
        # Training
        train_losses, train_accs = [], []
        for X_batch, y_batch in train_loader:
            loss, acc = classifier.train_step(X_batch, y_batch, optimizer, criterion)
            train_losses.append(loss)
            train_accs.append(acc)

        # Validation
        val_losses, val_accs = [], []
        for X_batch, y_batch in val_loader:
            loss, acc, _ = classifier.eval_step(X_batch, y_batch, criterion)
            val_losses.append(loss)
            val_accs.append(acc)

        train_loss = np.mean(train_losses)
        train_acc = np.mean(train_accs)
        val_loss = np.mean(val_losses)
        val_acc = np.mean(val_accs)

        print(
            f"Epoch {epoch+1:02d}/{config['training']['num_epochs']} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
        )

        # Early stopping and checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            # Save best model
            save_path = (
                get_project_root()
                / config["training"]["savedir"]
                / config["training"]["savename"]
            )
            os.makedirs(save_path.parent, exist_ok=True)
            classifier.save(str(save_path))
            print(f"  → Saved best model (val_acc: {val_acc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping after {epoch+1} epochs")
                break

    print(f"\n✓ Training complete! Best validation accuracy: {best_val_acc:.4f}")
    return classifier


def main():
    print("\n🧠 EEG MOTOR IMAGERY CLASSIFIER TRAINING\n")

    # Load config
    config = load_config("eeg_config")

    # Prepare data
    X_train, y_train, X_val, y_val = prepare_data(config)

    # Check if data loading failed
    if X_train is None or X_val is None:
        print("\nCannot proceed without data. Exiting.")
        return

    # Train model
    _ = train_model(config, X_train, y_train, X_val, y_val)

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    save_path = (
        get_project_root()
        / config["training"]["savedir"]
        / config["training"]["savename"]
    )
    print(f"\nModel saved to: {save_path}")
    print("Run 'python demos/motor_imagery_demo.py' to test it!")


if __name__ == "__main__":
    main()
