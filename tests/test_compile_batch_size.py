import copy
import unittest
from unittest.mock import patch

import torch

from flowtse_rl.model import MelUDiT


class CompileBatchSizeTests(unittest.TestCase):
    def model(self, mode='channel_concat', attention='bidirectional'):
        torch.manual_seed(42)
        result = MelUDiT(hidden_size=32, depth=2, num_heads=4, mlp_ratio=2,
            use_checkpoint=False, compile_blocks=True, compile_batch_size=2,
            conditioning_mode=mode, attention_mode=attention)
        with torch.no_grad():
            result.final_layer.linear.weight.normal_(std=.02)
            for block in [*result.in_blocks, result.mid_block, *result.out_blocks]:
                block.adaLN_modulation[-1].weight.normal_(std=.01)
        return result.train()

    def batch(self, count):
        return (torch.randn(count,100,8), torch.rand(count), torch.randn(count,100,6),
                torch.randn(count,100,8), torch.full((count,),8),torch.full((count,),6))

    def test_full_partial_full_routes_keep_compiled_cache_and_exact_partial_gradients(self):
        for mode,attention in [('channel_concat','bidirectional'),('time_concat','bidirectional'),
                               ('flowtse','enrollment_readonly'),('flowtse','bidirectional')]:
            with self.subTest(mode=mode,attention=attention):
                model=self.model(mode,attention)
                reference=copy.deepcopy(model);reference.compile_blocks=False
                calls=[]
                def compiler(function,**options):
                    def compiled(*args,**kwargs):
                        calls.append(args[0].shape[0])
                        return function(*args,**kwargs)
                    return compiled
                with patch('torch.compile',side_effect=compiler) as compile_call:
                    model(*self.batch(2)).square().mean().backward()
                    built=compile_call.call_count
                    self.assertEqual(built,3)
                    self.assertEqual(calls,[2,2,2])
                    model.zero_grad(set_to_none=True)
                    batch=self.batch(1)
                    left=model(*batch);right=reference(*batch)
                    torch.testing.assert_close(left,right,rtol=0,atol=0)
                    left.square().mean().backward();right.square().mean().backward()
                    for p,q in zip(model.parameters(),reference.parameters()):
                        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,rtol=0,atol=0)
                    self.assertEqual(calls,[2,2,2])
                    self.assertEqual(compile_call.call_count,built)
                    model(*self.batch(2)).square().mean().backward()
                    self.assertEqual(calls,[2]*6)
                    self.assertEqual(compile_call.call_count,built)
                self.assertEqual(set(model.state_dict()),set(reference.state_dict()))

    def test_partial_first_does_not_build_or_call_compiled_blocks(self):
        model=self.model()
        with patch('torch.compile',side_effect=AssertionError('Partial batch must stay eager')):
            model(*self.batch(1)).square().mean().backward()
        self.assertTrue(all(block._compiled_forward is None for block in
            [*model.in_blocks,model.mid_block,*model.out_blocks]))

    def test_unspecified_batch_size_preserves_previous_default(self):
        model=self.model();model.compile_batch_size=None
        with patch('torch.compile',side_effect=lambda fn,**kwargs:fn) as compiler:
            model(*self.batch(1)).square().mean().backward()
        self.assertEqual(compiler.call_count,3)

    def test_invalid_compile_batch_size_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'compile_batch_size'):
            MelUDiT(hidden_size=32,depth=2,num_heads=4,compile_batch_size=0)
