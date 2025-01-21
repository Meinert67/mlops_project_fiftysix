import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from src.project.model import MyAwesomeModel


def test_model():
    model = MyAwesomeModel()
    x = torch.randn(1, 3, 32, 32)
    y = model(x)
    assert y.shape == (1, 10), "Wrong output shape, should be (1, 10)"


if __name__ == "__main__":
    test_model()
