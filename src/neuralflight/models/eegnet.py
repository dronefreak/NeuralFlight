"""EEGNet-inspired model for motor imagery classification.

Based on: Lawhern et al. (2018) "EEGNet: A Compact Convolutional Network
for EEG-based Brain-Computer Interfaces"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    """Compact CNN for EEG motor imagery classification.

    Architecture:
    1. Temporal convolution (learns frequency filters)
    2. Depthwise convolution (learns spatial filters per frequency)
    3. Separable convolution (learns temporal patterns)
    4. Classification head
    """

    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 3,
        n_samples: int = 480,
        dropout: float = 0.5,
        kernel_length: int = 64,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
    ):
        """Initialize EEGNet.

        Args:
            n_channels: Number of EEG channels
            n_classes: Number of classes to predict
            n_samples: Number of time samples per epoch
            dropout: Dropout rate
            kernel_length: Length of temporal convolution kernel
            F1: Number of temporal filters
            D: Depth multiplier for spatial filters
            F2: Number of pointwise filters
        """
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_samples = n_samples

        # Layer 1: Temporal convolution
        self.conv1 = nn.Conv2d(
            1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(F1)

        # Layer 2: Depthwise spatial convolution
        self.conv2 = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.pooling1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout)

        # Layer 3: Separable convolution
        self.conv3 = nn.Conv2d(
            F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False
        )
        self.conv4 = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.pooling2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout)

        # Calculate flattened size
        self._to_linear = self._get_conv_output_size()

        # Classification head
        self.fc = nn.Linear(self._to_linear, n_classes)

    def _get_conv_output_size(self):
        """Calculate the size after convolutions."""
        with torch.no_grad():
            x = torch.zeros(1, 1, self.n_channels, self.n_samples)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.pooling1(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.pooling2(x)
            return x.numel()

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, channels, time_samples)

        Returns:
            Logits of shape (batch, n_classes)
        """
        # Add "image" dimension: (batch, channels, time) -> (batch, 1, channels, time)
        x = x.unsqueeze(1)

        # Block 1: Temporal convolution
        x = self.conv1(x)
        x = self.batchnorm1(x)

        # Block 2: Depthwise spatial convolution
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = self.pooling1(x)
        x = self.dropout1(x)

        # Block 3: Separable convolution
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.batchnorm3(x)
        x = F.elu(x)
        x = self.pooling2(x)
        x = self.dropout2(x)

        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class EEGClassifier:
    """Wrapper for training and inference with EEGNet."""

    def __init__(self, model: EEGNet, device: str = "cpu"):
        self.model = model.to(device)
        self.device = device

    def train_step(self, X_batch, y_batch, optimizer, criterion):
        """Single training step."""
        self.model.train()

        X_batch = X_batch.to(self.device)
        y_batch = y_batch.to(self.device)

        # Forward pass
        optimizer.zero_grad()
        outputs = self.model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == y_batch).sum().item()
        accuracy = correct / y_batch.size(0)

        return loss.item(), accuracy

    def eval_step(self, X_batch, y_batch, criterion):
        """Single evaluation step."""
        self.model.eval()

        X_batch = X_batch.to(self.device)
        y_batch = y_batch.to(self.device)

        with torch.no_grad():
            outputs = self.model(X_batch)
            loss = criterion(outputs, y_batch)

            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == y_batch).sum().item()
            accuracy = correct / y_batch.size(0)

        return loss.item(), accuracy, predicted.cpu().numpy()

    def predict(self, X):
        """Predict class for input.

        Args:
            X: Input tensor (batch, channels, time_samples) or (channels, time_samples)

        Returns:
            Predicted class indices and probabilities
        """
        self.model.eval()

        # Handle single sample
        if X.dim() == 2:
            X = X.unsqueeze(0)

        X = X.to(self.device)

        with torch.no_grad():
            outputs = self.model(X)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

        return predicted.cpu().numpy(), probabilities.cpu().numpy()

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_config": {
                    "n_channels": self.model.n_channels,
                    "n_classes": self.model.n_classes,
                    "n_samples": self.model.n_samples,
                },
            },
            path,
        )

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
