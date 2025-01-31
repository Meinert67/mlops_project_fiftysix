import pytorch_lightning as pl
import torch
from torch import nn
from loguru import logger


class MyAwesomeModel(pl.LightningModule):
    """Model for cifar-10"""

    def __init__(self, dropout_rate=0.5) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1)  # Input channels updated to 3
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.dropout = nn.Dropout(dropout_rate)

        # Compute the flattened dimension dynamically
        self.example_input_array = torch.randn(1, 3, 32, 32)  # Example input
        conv_out = self._get_conv_output(self.example_input_array)
        self.fc1 = nn.Linear(conv_out, 10)

        self.loss_fn = nn.CrossEntropyLoss()

    def _get_conv_output(self, x: torch.Tensor) -> int:
        """Compute the output dimension of the convolutional layers."""
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2, 2)
        return x.flatten(1).shape[1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc1(x)

    def training_step(self, batch):
        """Training step."""
        img, target = batch
        y_pred = self(img)
        return self.loss_fn(y_pred, target)


if __name__ == "__main__":
    model = MyAwesomeModel()
    logger.info(f"Model architecture: {model}")
    logger.info(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 3, 32, 32)
    output = model(dummy_input)
    logger.info(f"Output shape: {output.shape}")
