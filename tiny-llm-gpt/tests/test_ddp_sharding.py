import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from train_ddp import shard_offset_bounds


def test_shards_partition_the_full_range_with_no_gaps_or_overlap():
    usable_len = 1000
    world_size = 4
    bounds = [shard_offset_bounds(usable_len, world_size, rank) for rank in range(world_size)]

    assert bounds[0][0] == 0
    assert bounds[-1][1] == usable_len
    for (start, end), (next_start, _) in zip(bounds, bounds[1:]):
        assert end == next_start
        assert start < end


def test_uneven_division_folds_the_remainder_into_the_last_shard():
    usable_len = 1001  # not evenly divisible by 4
    world_size = 4
    bounds = [shard_offset_bounds(usable_len, world_size, rank) for rank in range(world_size)]

    sizes = [end - start for start, end in bounds]
    assert sizes[:-1] == [250, 250, 250]
    assert sizes[-1] == 251  # remainder folded into the last rank
    assert bounds[-1][1] == usable_len


def test_single_rank_gets_the_entire_range():
    start, end = shard_offset_bounds(1000, world_size=1, rank=0)
    assert (start, end) == (0, 1000)


@pytest.mark.parametrize("rank,world_size", [(-1, 4), (4, 4), (0, 0)])
def test_invalid_rank_or_world_size_raises(rank, world_size):
    with pytest.raises(ValueError):
        shard_offset_bounds(1000, world_size=world_size, rank=rank)


def test_usable_len_smaller_than_world_size_raises():
    with pytest.raises(ValueError):
        shard_offset_bounds(2, world_size=4, rank=0)
