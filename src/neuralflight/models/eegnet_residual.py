"""EEGNet with Residual Connections for improved motor imagery classification.

Based on: Lawhern et al. (2018) "EEGNet: A Compact Convolutional Network
for EEG-based Brain-Computer Interfaces"

Enhanced with:
- Residual connections for better gradient flow
- Optional attention mechanism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNetResidual(nn.Module):
    """EEGNet with residual connections for improved performance.

    Residual connections added:
    1. Around temporal block (if dimensions match)
    2. Around separable convolution block
    3. Optional attention mechanism on features
    """

    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 2,
        n_samples: int = 480,
        dropout: float = 0.5,
        kernel_length: int = 64,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        use_attention: bool = False,
    ):
        """Initialize EEGNet with residual connections.

        Args:
            n_channels: Number of EEG channels
            n_classes: Number of classes to predict
            n_samples: Number of time samples per epoch
            dropout: Dropout rate
            kernel_length: Length of temporal convolution kernel
            F1: Number of temporal filters
            D: Depth multiplier for spatial filters
            F2: Number of pointwise filters
            use_attention: Whether to use attention mechanism
        """
        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.use_attention = use_attention

        # Block 1: Temporal convolution
        self.conv1 = nn.Conv2d(
            1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False
        )
        self.batchnorm1 = nn.BatchNorm2d(F1)

        # Block 2: Depthwise spatial convolution
        self.conv2 = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D)
        self.pooling1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout)

        # Block 3: Separable convolution
        self.conv3 = nn.Conv2d(
            F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False
        )
        self.conv4 = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(F2)
        self.pooling2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout)

        # Residual projection for Block 3 (match dimensions)
        # Don't pool in the projection - we'll add before pooling
        self.residual_proj = nn.Sequential(
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False), nn.BatchNorm2d(F2)
        )

        # Optional attention mechanism
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(F2, F2 // 4), nn.ReLU(), nn.Linear(F2 // 4, F2), nn.Sigmoid()
            )

        # Calculate flattened size
        self._to_linear = self._get_conv_output_size()

        # Classification head with additional FC layer
        self.fc1 = nn.Linear(self._to_linear, self._to_linear // 2)
        self.dropout3 = nn.Dropout(dropout * 0.5)  # Less aggressive dropout
        self.fc2 = nn.Linear(self._to_linear // 2, n_classes)

    def _get_conv_output_size(self):
        """Calculate the size after convolutions."""
        with torch.no_grad():
            x = torch.zeros(1, 1, self.n_channels, self.n_samples)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.pooling1(x)

            # Separable conv with residual
            identity = x
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.batchnorm3(x)
            identity = self.residual_proj(identity)

            # Match dimensions if needed
            if x.shape[3] != identity.shape[3]:
                min_size = min(x.shape[3], identity.shape[3])
                x = x[:, :, :, :min_size]
                identity = identity[:, :, :, :min_size]

            x = x + identity  # Add before pooling
            x = self.pooling2(x)

            return x.numel()

    def forward(self, x):
        """Forward pass with residual connections.

        Args:
            x: Input tensor of shape (batch, channels, time_samples)

        Returns:
            Logits of shape (batch, n_classes)
        """
        # Add "image" dimension
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

        # Block 3: Separable convolution WITH RESIDUAL CONNECTION
        identity = x

        # Main path
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.batchnorm3(x)

        # Residual path (project identity to match channel dimensions)
        identity = self.residual_proj(identity)

        # Match temporal dimensions if needed (handle padding differences)
        if x.shape[3] != identity.shape[3]:
            # Crop the larger one to match the smaller
            min_size = min(x.shape[3], identity.shape[3])
            x = x[:, :, :, :min_size]
            identity = identity[:, :, :, :min_size]

        # Add residual connection BEFORE activation and pooling
        x = x + identity
        x = F.elu(x)

        # Now pool
        x = self.pooling2(x)
        x = self.dropout2(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Optional attention
        if self.use_attention:
            # Global average pooling for attention
            attn_weights = self.attention(x)
            x = x * attn_weights

        # Classification head with residual-like structure
        x = self.fc1(x)
        x = F.elu(x)
        x = self.dropout3(x)
        x = self.fc2(x)

        return x


class EEGNet(nn.Module):
    """Original EEGNet (kept for backwards compatibility).

    Use EEGNetResidual for better performance.
    """

    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 2,
        n_samples: int = 480,
        dropout: float = 0.5,
        kernel_length: int = 64,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
    ):
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
        # For kernel_size=16, use padding=7 for 'same' padding (maintains size)
        # With padding=(16-1)//2 = 7, output size = input size
        self.conv3 = nn.Conv2d(
            F1 * D, F1 * D, (1, 16), padding=(0, 7), groups=F1 * D, bias=False
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
        """Forward pass."""
        x = x.unsqueeze(1)

        # Block 1
        x = self.conv1(x)
        x = self.batchnorm1(x)

        # Block 2
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = self.pooling1(x)
        x = self.dropout1(x)

        # Block 3
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

    def __init__(self, model: nn.Module, device: str = "cpu"):
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
