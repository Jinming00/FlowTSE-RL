"""Condition routing, ragged batching and directional information-flow checks."""
from copy import deepcopy
from pathlib import Path
import unittest

import torch

from flowtse_rl.config import BASELINE_CONFIG, load_config, validate_config
from flowtse_rl.backbone.rotary import RotaryEmbedding
from flowtse_rl.data import FrameBatchSampler
from flowtse_rl.flow import flow_matching_loss
from flowtse_rl.rl import inject_lora, sample_window_sde, grpo_loss
from test_baseline import tiny_model


VARIANTS = (("channel_concat", "bidirectional"), ("time_concat", "bidirectional"),
            ("flowtse", "enrollment_readonly"), ("flowtse", "bidirectional"))


class ConditioningTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_rotary_cache_does_not_depend_on_autocast_history(self):
        query = torch.randn(2, 4, 41, 16)
        mixed, full = RotaryEmbedding(16), RotaryEmbedding(16)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            mixed(query.bfloat16(), query.bfloat16())
        expected = full(query, query)
        actual = mixed(query, query)
        torch.testing.assert_close(mixed._cos_cached, full._cos_cached, atol=0, rtol=0)
        torch.testing.assert_close(mixed._sin_cached, full._sin_cached, atol=0, rtol=0)
        for first, second in zip(actual, expected):
            torch.testing.assert_close(first, second, atol=0, rtol=0)

    def test_layouts_preserve_ragged_outputs_and_checkpointed_backward(self):
        for mode, attention in VARIANTS:
            with self.subTest(mode=mode, attention=attention):
                model = tiny_model(mode, attention, use_checkpoint=True).eval()
                state, mixture, enrollment = torch.randn(2, 4, 9), torch.randn(2, 4, 9), torch.randn(2, 4, 5)
                lengths, elengths, time = torch.tensor([9, 6]), torch.tensor([5, 2]), torch.tensor([0.3, 0.7])
                actual = model(state, time, enrollment, mixture, lengths, elengths)
                for index, (ml, el) in enumerate(zip(lengths.tolist(), elengths.tolist())):
                    expected = model(state[index:index+1, :, :ml], time[index:index+1], enrollment[index:index+1, :, :el],
                                     mixture[index:index+1, :, :ml], lengths[index:index+1], elengths[index:index+1])
                    torch.testing.assert_close(actual[index:index+1, :, :ml], expected, atol=1e-6, rtol=1e-5)
                state[1, :, 6:] = mixture[1, :, 6:] = float('nan')
                enrollment[1, :, 2:] = float('nan')
                torch.testing.assert_close(model(state, time, enrollment, mixture, lengths, elengths), actual)
                self.assertEqual(actual[1, :, 6:].count_nonzero(), 0)
                model.train()
                loss = flow_matching_loss(model, torch.randn_like(state), mixture, enrollment, lengths, elengths)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_directional_attention_blocks_only_enrollment_reading_other_tokens(self):
        for attention in ("enrollment_readonly", "bidirectional"):
            with self.subTest(attention=attention):
                model = tiny_model("flowtse", attention).eval()
                state = torch.randn(1, 4, 5, requires_grad=True)
                mixture = torch.randn(1, 4, 5, requires_grad=True)
                enrollment = torch.randn(1, 4, 2, requires_grad=True)
                hidden = []
                handle = model.final_layer.register_forward_pre_hook(lambda module, args: hidden.append(args[0]))
                output = model(state, torch.tensor([0.4]), enrollment, mixture, torch.tensor([5]), torch.tensor([2]))
                handle.remove()
                gradients = torch.autograd.grad(hidden[0][:, :2].square().sum(), (state, mixture), retain_graph=True)
                for gradient in gradients:
                    if attention == "enrollment_readonly":
                        self.assertEqual(gradient.count_nonzero(), 0)
                    else:
                        self.assertGreater(float(gradient.abs().sum()), 0)
                # Check the allowed direction too, with nonzero initialized gates.
                for gradient in torch.autograd.grad(output.square().sum(), (state, mixture, enrollment)):
                    self.assertGreater(float(gradient.abs().sum()), 0)

    def test_new_interfaces_support_lora_sde_replay(self):
        for mode, attention in VARIANTS[1:3]:
            with self.subTest(mode=mode):
                model = tiny_model(mode, attention).eval()
                inject_lora(model, rank=4, alpha=8)
                mixture, enrollment = torch.randn(2, 4, 7), torch.randn(2, 4, 3)
                lengths, elengths = torch.tensor([7, 5]), torch.tensor([3, 2])
                _, records = sample_window_sde(model, mixture, enrollment, lengths, elengths, steps=5,
                                               generator=torch.Generator().manual_seed(29))
                loss, metrics = grpo_loss(model, records, torch.tensor([1., -1.]), mixture, enrollment, lengths, elengths)
                torch.testing.assert_close(metrics['ratio'], torch.tensor(1.))
                torch.testing.assert_close(metrics['reference_kl'], torch.tensor(0.))
                loss.backward()
                gradients = [p.grad for p in model.parameters() if p.requires_grad]
                self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
                self.assertTrue(any(g.abs().sum() > 0 for g in gradients))

    def test_retained_config_uses_selected_bidirectional_conditioning(self):
        root = BASELINE_CONFIG.parent
        paths = sorted(root.glob('*.yaml'))
        self.assertEqual(paths, [BASELINE_CONFIG])
        for path in paths:
            with self.subTest(config=path.name):
                config = load_config(path)
                validate_config(config)
                self.assertEqual(config['model']['conditioning_mode'], 'flowtse')
                self.assertEqual(config['model']['attention_mode'], 'bidirectional')

    def test_time_concat_counts_both_mixture_and_state_in_batch_budget(self):
        class Examples:
            def __len__(self):
                return 4

            def estimated_frames(self, index, hop_length, mixture_copies=1):
                return 5 + 10 * mixture_copies

        kwargs = dict(dataset=Examples(), max_frames=60, max_samples=4, hop_length=256, shuffle=False, seed=42)
        self.assertEqual(list(FrameBatchSampler(**kwargs)), [[0, 1, 2, 3]])
        self.assertEqual(list(FrameBatchSampler(**kwargs, mixture_copies=2)), [[0, 1], [2, 3]])


if __name__ == '__main__':
    unittest.main()
