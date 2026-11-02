"""`deep._ShuffledTensorBatches` replaces ``DataLoader(TensorDataset, shuffle=True)``
for speed and must draw the same batches. It reproduces DataLoader's use of the
generator (one int64 base-seed draw at iterator creation, then RandomSampler's
two ``randperm`` calls), so a PyTorch release that changes that sequence fails
here instead of silently reordering every deep fit's batches."""
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from pysimlr.deep import _ShuffledTensorBatches, _drop_last


@pytest.mark.parametrize("n,bs", [(97, 32), (64, 32), (50, 7)])
def test_shuffled_batches_match_dataloader(n, bs):
    mats = [torch.arange(n, dtype=torch.float32)[:, None], torch.randn(n, 3)]
    ds = TensorDataset(*mats)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=_drop_last(ds, bs),
                    generator=torch.Generator().manual_seed(5))
    sb = _ShuffledTensorBatches(mats, bs, 5)
    assert len(sb) == len(dl)
    for _ in range(3):
        for a, b in zip(dl, sb, strict=True):
            for x, y in zip(a, b):
                assert torch.equal(x, y)


def test_both_drop_last_cases_are_exercised():
    assert {_drop_last(torch.zeros(n, 1), bs) for n, bs in [(97, 32), (64, 32), (50, 7)]} == {True, False}
