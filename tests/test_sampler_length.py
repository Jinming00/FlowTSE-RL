"""The epoch-length optimization must preserve the old data plan and RNG."""
import random
import unittest
from unittest.mock import patch

from torch.utils.data import DataLoader

from flowtse_rl.data import FrameBatchSampler


class VariableFrames:
    frames = [3, 8, 2, 7, 5, 11, 4, 9, 1, 6, 8, 2, 10, 5, 3, 12, 4]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        return index

    def estimated_frames(self, index, hop_length, mixture_copies=1):
        return self.frames[index]


# Captured from the unmodified sampler in the frozen pre-fix source. These
# include world-size truncation and both shuffles, before rank striding.
OLD_PLANS = {
    0: [[0, 4, 1], [10, 9, 8], [3, 13, 2], [15, 11]],
    1: [[13, 15], [2, 8, 10], [0, 6, 16], [1, 12]],
    2: [[4, 10, 14], [11, 8, 0, 6], [1, 7], [12, 3]],
}


class SamplerLengthTests(unittest.TestCase):
    def make(self, **kwargs):
        settings = dict(max_frames=24, max_samples=4, hop_length=1,
                        shuffle=True, seed=13, world_size=4)
        settings.update(kwargs)
        return FrameBatchSampler(VariableFrames(), **settings)

    def test_repeated_loader_lengths_keep_old_rank_plans_and_global_rng(self):
        state = random.getstate()
        for rank in range(4):
            sampler = self.make(rank=rank)
            loader = DataLoader(VariableFrames(), batch_sampler=sampler, num_workers=0)
            for epoch, expected in OLD_PLANS.items():
                sampler.set_epoch(epoch)
                for _ in range(50):
                    self.assertEqual(len(loader), len(expected[rank::4]))
                self.assertEqual([list(batch.numpy()) for batch in loader], expected[rank::4])
        self.assertEqual(random.getstate(), state)

    def test_length_queries_build_once_and_iteration_still_builds_original_plan(self):
        sampler = self.make()
        with patch.object(sampler, '_global_batches', wraps=sampler._global_batches) as build:
            for _ in range(100):
                len(sampler)
            self.assertEqual(build.call_count, 1)
            sampler.set_epoch(0)
            len(sampler)
            self.assertEqual(build.call_count, 1)
            self.assertEqual([batch for batch in sampler], OLD_PLANS[0][::4])
            self.assertEqual(build.call_count, 2)
            sampler.set_epoch(1)
            len(sampler)
            self.assertEqual(build.call_count, 3)
            self.assertEqual([batch for batch in sampler], OLD_PLANS[1][::4])

    def test_configuration_changes_and_epoch_change_invalidate_scalar_length(self):
        sampler = self.make(world_size=1)
        with patch.object(sampler, '_global_batches', wraps=sampler._global_batches) as build:
            len(sampler)
            for name, value in [('epoch', 1), ('max_samples', 3), ('max_frames', 30),
                                ('shuffle', False), ('seed', 7), ('world_size', 2)]:
                before = build.call_count
                setattr(sampler, name, value)
                measured = len(sampler)
                self.assertEqual(build.call_count, before + 1)
                self.assertEqual(measured, sum(1 for _ in sampler))

    def test_unsized_rank_budget_error_is_not_hidden_by_cache(self):
        sampler = self.make(world_size=100)
        for _ in range(2):
            with self.assertRaisesRegex(ValueError, 'Fewer training batches than ranks'):
                len(sampler)
        sampler.world_size = 1
        self.assertEqual(len(sampler), sum(1 for _ in sampler))


if __name__ == '__main__':
    unittest.main()
