# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Test for data/replay_buffer.py"""

from dataclasses import dataclass

import pytest
import pytest_asyncio
from forge.actors.replay_buffer import ReplayBuffer


@dataclass
class TestEpisode:
    """
    Dummy Episode containing just a policy version

    ReplayBuffer expects any construct (typically an Episode) that contains a
    `policy_version`.

    TODO: Replaced with a unified interface in the future.
    """

    policy_version: int


class TestReplayBuffer:
    @pytest_asyncio.fixture
    async def replay_buffer(self) -> ReplayBuffer:
        replay_buffer = await ReplayBuffer.options(procs=1, with_gpus=False).as_actor(
            batch_size=2, max_policy_age=1
        )
        await replay_buffer.setup.call()
        return replay_buffer

    @pytest.mark.asyncio
    async def test_add(self, replay_buffer: ReplayBuffer) -> None:
        episode = TestEpisode(policy_version=0)
        await replay_buffer.add.call_one(episode)
        assert replay_buffer._numel.call_one().get() == 1
        assert replay_buffer._getitem.call_one(0).get() == episode
        replay_buffer.clear.call_one().get()

    @pytest.mark.asyncio
    async def test_add_multiple(self, replay_buffer) -> None:
        episode_0 = TestEpisode(policy_version=0)
        episode_1 = TestEpisode(policy_version=1)
        await replay_buffer.add.call_one(episode_0)
        await replay_buffer.add.call_one(episode_1)
        assert replay_buffer._numel.call_one().get() == 2
        assert replay_buffer._getitem.call_one(0).get() == episode_0
        assert replay_buffer._getitem.call_one(1).get() == episode_1
        replay_buffer.clear.call_one().get()

    @pytest.mark.asyncio
    async def test_state_dict_save_load(self, replay_buffer) -> None:
        episode = TestEpisode(policy_version=0)
        await replay_buffer.add.call_one(episode)
        state_dict = replay_buffer.state_dict.call_one().get()
        replay_buffer.clear.call_one().get()
        assert replay_buffer._numel.call_one().get() == 0
        await replay_buffer.load_state_dict.call_one(state_dict)
        assert replay_buffer._numel.call_one().get() == 1
        replay_buffer.clear.call_one().get()

    @pytest.mark.asyncio
    async def test_evict(self, replay_buffer) -> None:
        episode_0 = TestEpisode(policy_version=0)
        episode_1 = TestEpisode(policy_version=1)
        await replay_buffer.add.call_one(episode_0)
        await replay_buffer.add.call_one(episode_1)
        assert replay_buffer._numel.call_one().get() == 2
        await replay_buffer.evict.call_one(curr_policy_version=2)
        assert replay_buffer._numel.call_one().get() == 1
        replay_buffer.clear.call_one().get()

    @pytest.mark.asyncio
    async def test_sample(self, replay_buffer) -> None:
        episode_0 = TestEpisode(policy_version=0)
        episode_1 = TestEpisode(policy_version=1)
        await replay_buffer.add.call_one(episode_0)
        await replay_buffer.add.call_one(episode_1)
        assert replay_buffer._numel.call_one().get() == 2

        # Test a simple sampling
        samples = await replay_buffer.sample.call_one(curr_policy_version=1)
        assert samples is not None
        assert len(samples[0]) == 2
        assert replay_buffer._numel.call_one().get() == 2

        # Test sampling (not enough samples in buffer, returns None)
        await replay_buffer.add.call_one(episode_0)
        samples = await replay_buffer.sample.call_one(curr_policy_version=1)
        assert samples is None
        replay_buffer.clear.call_one().get()

    @pytest.mark.asyncio
    async def test_sample_with_evictions(self, replay_buffer) -> None:
        episode_0 = TestEpisode(policy_version=0)
        episode_1 = TestEpisode(policy_version=1)
        episode_2 = TestEpisode(policy_version=2)
        await replay_buffer.add.call_one(episode_0)
        await replay_buffer.add.call_one(episode_1)
        await replay_buffer.add.call_one(episode_2)
        assert replay_buffer._numel.call_one().get() == 3
        samples = await replay_buffer.sample.call_one(
            curr_policy_version=2,
        )
        assert samples is not None
        assert len(samples[0]) == 2
        assert samples[0][0].policy_version > 0
        assert samples[0][1].policy_version > 0
        assert replay_buffer._numel.call_one().get() == 2
        replay_buffer.clear.call_one().get()

    @pytest.mark.asyncio
    async def test_sample_dp_size(self) -> None:
        """Test that len(samples) == dp_size when sampling."""
        # Create replay buffer with dp_size=3
        replay_buffer = await ReplayBuffer.options(procs=1, with_gpus=False).as_actor(
            batch_size=2, max_policy_age=1, dp_size=3
        )
        await replay_buffer.setup.call()

        # Add enough trajectories to sample
        for i in range(10):
            episode = TestEpisode(policy_version=0)
            await replay_buffer.add.call_one(episode)

        # Sample and verify len(samples) == dp_size
        samples = await replay_buffer.sample.call_one(curr_policy_version=0)
        assert samples is not None
        assert len(samples) == 3  # dp_size
        # Each sub-list should have batch_size samples
        for dp_samples in samples:
            assert len(dp_samples) == 2  # batch_size

        replay_buffer.clear.call_one().get()

    @pytest.mark.asyncio
    async def test_collect(self) -> None:
        """Test _collect method"""
        local_rb = ReplayBuffer(batch_size=1)
        await local_rb.setup._method(local_rb)
        for i in range(1, 6):
            local_rb.buffer.append(i)
        values = local_rb._collect([2, 0, -1])
        assert values == [3, 1, 5]
        values = local_rb._collect([1, 3])
        assert values == [2, 4]
        assert local_rb.buffer[0] == 1
