"""Packing changes must preserve gradients, masks and checkpoint keys."""
import unittest
from unittest.mock import patch
import torch
from test_baseline import tiny_model


class VectorizedPackingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_uniform_masked_compile_preserves_visibility_and_all_gradients(self):
        eager=tiny_model('flowtse','enrollment_readonly').train()
        compiled=tiny_model('flowtse','enrollment_readonly').train()
        compiled.compile_blocks=True
        calls=[]
        def compile_stub(fn,**kwargs):
            def checked(tokens,condition,skip=None,token_mask=None,attention_mask=None):
                self.assertTrue(token_mask.all())
                self.assertFalse(attention_mask[:,:5,5:].any())
                self.assertTrue(attention_mask[:,5:,:5].all())
                calls.append(fn.__self__)
                return fn(tokens,condition,skip,token_mask,attention_mask)
            return checked
        values=[torch.randn(2,4,9),torch.randn(2,4,9),torch.randn(2,4,5)]
        upstream=torch.randn_like(values[0]);outputs=[];grads=[]
        with patch('torch.compile',side_effect=compile_stub):
            for model in (eager,compiled):
                inputs=[v.clone().requires_grad_(True) for v in values]
                hidden=[]
                handle=model.final_layer.register_forward_pre_hook(lambda module,args: hidden.append(args[0]))
                out=model(inputs[0],torch.tensor([0.3,0.7]),inputs[2],inputs[1],
                          torch.tensor([9,9]),torch.tensor([5,5]))
                handle.remove()
                # Enrollment must remain independent of mixture/state at every block.
                forbidden=torch.autograd.grad(hidden[0][:,:5].square().sum(),inputs[:2],retain_graph=True)
                self.assertTrue(all(g.count_nonzero()==0 for g in forbidden))
                outputs.append(out)
                grads.append(torch.autograd.grad((out*upstream).sum(),inputs+list(model.parameters())))
        self.assertEqual(len(calls),3)
        torch.testing.assert_close(outputs[0],outputs[1],rtol=0,atol=0)
        for before,after in zip(*grads):
            torch.testing.assert_close(before,after,rtol=0,atol=0)
        self.assertEqual(set(eager.state_dict()),set(compiled.state_dict()))
        eager.load_state_dict(compiled.state_dict(),strict=True)

    def test_masked_compile_falls_back_for_padding_eval_and_no_grad(self):
        model=tiny_model('flowtse','enrollment_readonly');model.compile_blocks=True
        phase=['uniform'];calls=[]
        def compile_stub(fn,**kwargs):
            def checked(tokens,*args,**kw):
                self.assertEqual(phase[0],'uniform')
                self.assertEqual(fn.__self__.attn.rotary._seq_len_cached,tokens.shape[1])
                calls.append(fn.__self__)
                return fn(tokens,*args,**kw)
            return checked
        def run(ml,el,lengths=None,elengths=None):
            return model(torch.randn(2,4,ml),torch.tensor([0.3,0.7]),torch.randn(2,4,el),
                         torch.randn(2,4,ml),torch.tensor(lengths or [ml,ml]),
                         torch.tensor(elengths or [el,el]))
        with patch('torch.compile',side_effect=compile_stub):
            model.train();run(9,5)
            phase[0]='evaluation';model.eval()
            with torch.no_grad():run(11,7)
            phase[0]='padding';model.train();out=run(9,5,[9,6],[5,2])
            self.assertEqual(out[1,:,6:].count_nonzero(),0)
            phase[0]='no_grad'
            with torch.no_grad():run(9,5)
            self.assertEqual(len(calls),3)
            phase[0]='uniform';run(9,5)
            self.assertEqual(len(calls),6)

    def test_training_restores_rotary_cache_after_variable_length_evaluation(self):
        model=tiny_model();model.compile_blocks=True
        def compile_stub(fn,**kwargs):
            def checked(tokens,*args,**kw):
                self.assertEqual(fn.__self__.attn.rotary._seq_len_cached,tokens.shape[1])
                return fn(tokens,*args,**kw)
            return checked
        def run(ml,el):
            model(torch.randn(2,4,ml),torch.tensor([0.3,0.7]),torch.randn(2,4,el),
                  torch.randn(2,4,ml),torch.tensor([ml,ml]),torch.tensor([el,el]))
        with patch('torch.compile',side_effect=compile_stub):
            model.train();run(9,5)
            model.eval()
            with torch.no_grad():run(11,7)
            self.assertEqual(model.mid_block.attn.rotary._seq_len_cached,18)
            model.train();run(9,5)
            self.assertEqual(model.mid_block.attn.rotary._seq_len_cached,14)

    def test_uniform_outputs_and_all_gradients_match(self):
        for mode,attention in [('channel_concat','bidirectional'),('time_concat','bidirectional'),
                               ('flowtse','enrollment_readonly')]:
            with self.subTest(mode=mode):
                models=[tiny_model(mode,attention,vectorized_packing=flag) for flag in [False,True]]
                values=[torch.randn(2,4,9),torch.randn(2,4,9),torch.randn(2,4,5)]
                upstream=torch.randn_like(values[0]);outputs=[];grads=[]
                for model in models:
                    inputs=[v.clone().requires_grad_(True) for v in values]
                    out=model(inputs[0],torch.tensor([0.3,0.7]),inputs[2],inputs[1],
                              torch.tensor([9,9]),torch.tensor([5,5]))
                    outputs.append(out)
                    grads.append(torch.autograd.grad((out*upstream).sum(),inputs+list(model.parameters())))
                self.assertEqual(models[1].last_packing_path,'vectorized')
                self.assertEqual(set(models[0].state_dict()),set(models[1].state_dict()))
                models[1].load_state_dict(models[0].state_dict(),strict=True)
                torch.testing.assert_close(outputs[0],outputs[1],rtol=1e-5,atol=1e-6)
                for before,after in zip(*grads):
                    torch.testing.assert_close(before,after,rtol=2e-4,atol=2e-6)

    def test_ragged_and_equal_total_different_segments_fall_back(self):
        for mode in ['channel_concat','time_concat']:
            for ml,el in [([9,6],[5,2]),([9,7],[3,5] if mode=='channel_concat' else [1,5])]:
                with self.subTest(mode=mode,lengths=ml):
                    ref=tiny_model(mode,vectorized_packing=False).eval()
                    candidate=tiny_model(mode,vectorized_packing=True).eval()
                    state=torch.randn(2,4,9);mix=torch.randn_like(state);enroll=torch.randn(2,4,5)
                    for i in range(2):
                        state[i,:,ml[i]:]=mix[i,:,ml[i]:]=float('nan')
                        enroll[i,:,el[i]:]=float('nan')
                    args=(state,torch.tensor([0.3,0.7]),enroll,mix,torch.tensor(ml),torch.tensor(el))
                    torch.testing.assert_close(ref(*args),candidate(*args),rtol=0,atol=0)
                    self.assertEqual(candidate.last_packing_path,'reference')


if __name__=='__main__':unittest.main()
