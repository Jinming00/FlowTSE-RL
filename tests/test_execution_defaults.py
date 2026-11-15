"""Exercise implicit execution policy, inheritance, and full/tail dispatch."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import yaml

from flowtse_rl.config import BASELINE_CONFIG, load_config, validate_config
from flowtse_rl.model import MelUDiT


class ExecutionDefaultTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_clean_yaml_keeps_accelerated_full_batch_and_eager_tail(self):
        raw = yaml.safe_load(BASELINE_CONFIG.read_text())
        hidden = {'use_checkpoint', 'compile_blocks', 'skip_redundant_mask',
                  'vectorized_packing', 'compile_mode', 'compile_batch_size'}
        self.assertFalse(hidden & raw['model'].keys())
        self.assertNotIn('fused', raw['optimizer'])
        self.assertNotIn('num_workers', raw['data'])
        config = load_config(BASELINE_CONFIG)
        validate_config(config)
        self.assertEqual(config['data']['num_workers'], 8)
        options = dict(config['model'], hidden_size=32, depth=2, num_heads=4, mlp_ratio=2)
        candidate = MelUDiT(**options).train()
        for block in [*candidate.in_blocks, candidate.mid_block, *candidate.out_blocks]:
            self.assertFalse(block.use_checkpoint)
        self.assertTrue(candidate.vectorized_packing)
        self.assertTrue(candidate.skip_redundant_mask)
        self.assertEqual(candidate.compile_mode, 'max-autotune-no-cudagraphs')
        reference = deepcopy(candidate)
        reference.compile_blocks = False
        calls = []
        def compile_stub(fn, **kwargs):
            calls.append(kwargs)
            return fn
        with patch('torch.compile', side_effect=compile_stub):
            for batch in [32, 24, 32]:
                candidate.zero_grad(set_to_none=True)
                reference.zero_grad(set_to_none=True)
                state = torch.randn(batch, 100, 5)
                inputs = (state, torch.rand(batch), torch.randn(batch, 100, 3),
                          torch.randn_like(state), torch.full((batch,), 5), torch.full((batch,), 3))
                expected = reference(*inputs)
                actual = candidate(*inputs)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                actual.square().mean().backward()
                expected.square().mean().backward()
                for a, b in zip(candidate.parameters(), reference.parameters()):
                    torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
                self.assertEqual(len(calls), 3)

    def test_inheritance_resolves_batch_size_after_merging(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'parent.yaml').write_text(yaml.safe_dump(dict(data={'max_samples_per_batch': 32})))
            child = root/'child.yaml'
            child.write_text(yaml.safe_dump(dict(extends='parent.yaml', data={'max_samples_per_batch': 16})))
            self.assertEqual(load_config(child)['model']['compile_batch_size'], 16)
            child.write_text(yaml.safe_dump(dict(extends='parent.yaml',
                model={'compile_blocks': False, 'use_checkpoint': True, 'compile_batch_size': 8},
                optimizer={'fused': True})))
            config = load_config(child)
            self.assertFalse(config['model']['compile_blocks'])
            self.assertTrue(config['model']['use_checkpoint'])
            self.assertEqual(config['model']['compile_batch_size'], 8)
            self.assertTrue(config['optimizer']['fused'])
            child.write_text(yaml.safe_dump(dict(extends='parent.yaml', model={'use_checkpoint': True})))
            self.assertFalse(load_config(child)['model']['compile_blocks'])

    def test_loaded_defaults_are_not_shared_between_experiments(self):
        a = load_config(BASELINE_CONFIG)
        a['model']['compile_blocks'] = False
        a['optimizer']['fused'] = True
        b = load_config(BASELINE_CONFIG)
        self.assertTrue(b['model']['compile_blocks'])
        self.assertFalse(b['optimizer']['fused'])


if __name__ == '__main__':
    unittest.main()
