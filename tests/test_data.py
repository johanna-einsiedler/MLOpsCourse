import os

import pytest
import torch

from test_project.data import corrupt_mnist

file_path = "/Users/htr365/Documents/PhD/MLOps/MLOpsCourse/jae_repo_1/data/processed/"
@pytest.mark.skipif(not os.path.exists(file_path), reason="Data files not found")
def test_data():

    train, test = corrupt_mnist(file_path)
    assert len(train) == 30000
    assert len(test) == 5000
    for dataset in [train, test]:
        for x, y in dataset:
            assert x.shape == (1, 28, 28)
            assert y in range(10)
    train_targets = torch.unique(train.tensors[1])
    assert (train_targets == torch.arange(0,10)).all()
    test_targets = torch.unique(test.tensors[1])
    assert (test_targets == torch.arange(0,10)).all()