import unittest

import torch

from flowtse_rl.rl import (LoRALinear, inject_lora, reference_policy, group_advantages,
                              sample_window_sde, grpo_loss, sde_stats)
from test_baseline import tiny_model


class RLTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_lora_initialization_preserves_model_and_freezes_base(self):
        model = tiny_model().eval()
        mixture, state, enrollment = torch.randn(2, 4, 9), torch.randn(2, 4, 9), torch.randn(2, 4, 5)
        lengths, elengths, time = torch.tensor([9, 6]), torch.tensor([5, 2]), torch.tensor([0.3, 0.7])
        baseline = model(state, time, enrollment, mixture, lengths, elengths).detach()
        names = inject_lora(model, rank=4, alpha=8)
        self.assertEqual(len(names), 3 * 6)
        actual = model(state, time, enrollment, mixture, lengths, elengths)
        torch.testing.assert_close(actual, baseline)
        actual.square().mean().backward()
        self.assertTrue(all(not p.requires_grad for n,p in model.named_parameters() if 'lora_' not in n))
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for n,p in model.named_parameters() if 'lora_b' in n))
        with torch.no_grad():
            for layer in model.modules():
                if isinstance(layer, LoRALinear):
                    layer.lora_b.add_(0.01)
        with reference_policy(model):
            torch.testing.assert_close(model(state, time, enrollment, mixture, lengths, elengths), baseline)

    def test_sde_replay_ratio_one_kl_zero_and_backward(self):
        model = tiny_model().eval()
        inject_lora(model, rank=4, alpha=8)
        mixture, enrollment = torch.randn(2, 4, 9), torch.randn(2, 4, 5)
        lengths, elengths = torch.tensor([9, 6]), torch.tensor([5, 2])
        output, records = sample_window_sde(model, mixture, enrollment, lengths, elengths, steps=5,
                                            generator=torch.Generator().manual_seed(29))
        self.assertEqual(len(records), 2)
        self.assertEqual(output[1, :, 6:].count_nonzero(), 0)
        loss, diagnostic = grpo_loss(model, records, torch.tensor([1.,-1.]), mixture, enrollment, lengths, elengths)
        torch.testing.assert_close(diagnostic['ratio'], torch.tensor(1.))
        torch.testing.assert_close(diagnostic['reference_kl'], torch.tensor(0.))
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.requires_grad]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
        self.assertTrue(any(g.abs().sum() > 0 for g in gradients))

    def test_zero_variance_groups_and_singular_time(self):
        advantages, eligible = group_advantages(torch.tensor([[2.,2.,2.],[1.,2.,3.]]))
        self.assertEqual(eligible.tolist(), [False, True])
        self.assertEqual(advantages[0].count_nonzero(), 0)
        torch.testing.assert_close(advantages[1].mean(), torch.tensor(0.))
        with self.assertRaises(ValueError):
            sde_stats(torch.zeros(1,4,5),torch.zeros(1,4,5),0.,0.1)


if __name__ == '__main__':
    unittest.main()

