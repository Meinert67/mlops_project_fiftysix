import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from torch.utils.data import Dataset
from src.project.data import preprocess


def test_my_dataset():
    """Test the MyDataset class."""
    dataset = preprocess()
    assert isinstance(dataset, Dataset)

    train = dataset.train_set
    test = dataset.test_set

    assert len(train) == 50000, "Train Dataset did not have the correct number of samples (50.000)"
    assert len(test) == 10000, "Test Dataset did not have the correct number of samples (10.000)"
    for dataset in [train, test]:
        for x, y in dataset:
            assert x.shape == (3, 32, 32), "Dataset shape not correct size"
            assert y in range(10)
    train_targets = set([y for _, y in train])
    assert train_targets == set(range(10))

    test_targets = set([y for _, y in test])
    assert test_targets == set(range(10))


if __name__ == "__main__":
    test_my_dataset()
