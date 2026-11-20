"""Epoch shuffling with a cursor shared by all proposals."""

from random import Random

from gepa.core.data_loader import DataLoader
from gepa.core.state import GEPAState
from gepa.strategies.batch_sampler import EpochShuffledBatchSampler


class CursorEpochShuffledBatchSampler(EpochShuffledBatchSampler):
    """Advance one batch per proposal, retaining upstream shuffling and padding."""

    def __init__(self, minibatch_size: int, rng: Random):
        super().__init__(minibatch_size=minibatch_size, rng=rng)
        self.cursor = 0

    def next_minibatch_ids(self, loader: DataLoader, state: GEPAState) -> list:
        if not len(loader):
            raise ValueError("Cannot sample a minibatch from an empty loader")
        if (
            not self.shuffled_ids
            or len(loader) != self.last_trainset_size
            or self.cursor >= len(self.shuffled_ids)
        ):
            self.epoch += 1
            self._update_shuffled(loader)
            self.cursor = 0
        start = self.cursor
        self.cursor += self.minibatch_size
        return self.shuffled_ids[start : self.cursor]
