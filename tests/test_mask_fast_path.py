"""Mask visibility and numerical contracts for the no-padding fast path."""
import unittest

import torch

from test_baseline import tiny_model


class MaskFastPathTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    @staticmethod
    def masks(model, lengths, elengths):
        seen = []
        handle = model.in_blocks[0].attn.register_forward_pre_hook(
            lambda module,args,kwargs: seen.append((kwargs['context_mask'],kwargs['attention_mask'])),
            with_kwargs=True)
        output = model(torch.randn(2,4,9),torch.tensor([0.3,0.7]),torch.randn(2,4,5),
                       torch.randn(2,4,9),lengths,elengths)
        handle.remove()
        return seen[0], output

    def test_all_valid_bidirectional_omits_padding_mask(self):
        for mode in ['channel_concat','time_concat']:
            with self.subTest(mode=mode):
                model=tiny_model(mode).eval()
                (key,pairwise),_=self.masks(model,torch.tensor([9,9]),torch.tensor([5,5]))
                self.assertIsNone(key)
                self.assertIsNone(pairwise)
                model.skip_redundant_mask=False
                (key,pairwise),_=self.masks(model,torch.tensor([9,9]),torch.tensor([5,5]))
                self.assertTrue(key.all())
                self.assertIsNone(pairwise)

    def test_padding_and_directional_visibility_are_preserved(self):
        for mode in ['channel_concat','time_concat']:
            with self.subTest(mode=mode):
                (key,pairwise),output=self.masks(tiny_model(mode).eval(),torch.tensor([9,6]),torch.tensor([5,2]))
                self.assertIsNotNone(key)
                self.assertTrue((~key).any())
                self.assertIsNone(pairwise)
                self.assertEqual(output[1,:,6:].count_nonzero(),0)
        (key,pairwise),_=self.masks(tiny_model('flowtse','enrollment_readonly').eval(),
                                   torch.tensor([9,9]),torch.tensor([5,5]))
        self.assertIsNotNone(key)
        self.assertIsNotNone(pairwise)
        self.assertFalse(pairwise[0,:5,5:].any())
        self.assertTrue(pairwise[0,5:,:5].all())

    def test_forward_and_input_parameter_gradients_match_reference(self):
        for mode in ['channel_concat','time_concat']:
            # Equal packed lengths can also have different segment boundaries.
            patterns=[([9,9],[5,5]),([9,7],[3,5] if mode=='channel_concat' else [1,5])]
            for ml,el in patterns:
                with self.subTest(mode=mode,mixture_lengths=ml):
                    reference=tiny_model(mode,skip_redundant_mask=False).eval()
                    candidate=tiny_model(mode,skip_redundant_mask=True).eval()
                    values=[torch.randn(2,4,9),torch.randn(2,4,9),torch.randn(2,4,5)]
                    time=torch.tensor([0.3,0.7]);lengths=torch.tensor(ml);elengths=torch.tensor(el)
                    upstream=torch.randn(2,4,9)
                    outputs=[];gradients=[]
                    for model in [reference,candidate]:
                        inputs=[value.clone().requires_grad_(True) for value in values]
                        out=model(inputs[0],time,inputs[2],inputs[1],lengths,elengths)
                        grads=torch.autograd.grad((out*upstream).sum(),inputs+list(model.parameters()))
                        outputs.append(out.detach());gradients.append(grads)
                    torch.testing.assert_close(outputs[0],outputs[1],atol=1e-6,rtol=1e-5)
                    for before,after in zip(*gradients):
                        torch.testing.assert_close(before,after,atol=2e-6,rtol=2e-4)


if __name__=='__main__':
    unittest.main()
