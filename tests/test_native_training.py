import copy
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader

from flowtse_rl.engine import FlowSystem, make_optimizer_scheduler, seed_worker
from flowtse_rl.training import run_training
from flowtse_rl.utils.native_checkpoint import CheckpointManager, NativeCheckpointIO, resume_position


def config(root):
    return dict(seed=42, audio=dict(sample_rate=24000,n_fft=1024,win_length=1024,hop_length=256,n_mels=100),
        model=dict(mel_dim=100,hidden_size=32,depth=2,num_heads=4,mlp_ratio=2,use_checkpoint=False,compile_blocks=False),
        vocoder={}, optimizer=dict(lr=.01,betas=[.9,.999],weight_decay=.01,warmup_epochs=0,minimum_lr_ratio=1.,fused=False),
        train=dict(num_epochs=2,max_steps=-1,precision='32-true',accumulation_steps=1,gradient_clip_val=1.,log_interval=2,wandb=False),
        ddp=dict(num_nodes=1,num_gpus=1,use_ddp=False), validation=dict(every_n_epochs=1),
        checkpoint=dict(dir=str(root),resume=None,monitor='val/pesq',mode='max',save_top_k=3,last_every_n_epochs=1))


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Linear(2,1)

    def forward(self,batch):
        assert 'metadata' not in batch
        return (self.model(batch['mixture'])-batch['target']).square().mean()


def loader():
    def collate(ids):
        wave=torch.tensor([[i/5.,(i+1)/7.] for i in ids])
        return dict(mixture=wave,target=wave.sum(1,keepdim=True)*.4,enrollment=wave,
                    mixture_lengths=torch.full((len(ids),),2),target_lengths=torch.full((len(ids),),2),
                    enrollment_lengths=torch.full((len(ids),),2),metadata=[dict(record_index=i) for i in ids])
    return DataLoader(list(range(6)),batch_size=2,collate_fn=collate,num_workers=0)


class NativeTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.accelerator=Accelerator(cpu=True,mixed_precision='no',gradient_accumulation_steps=1,
                                    step_scheduler_with_optimizer=False)


    def test_full_native_resume_matches_uninterrupted_optimizer_updates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            torch.manual_seed(17);full=Tiny()
            full_config=config(root/'full')
            result=run_training(full_config,system=full,loaders=(loader(),None),accelerator=self.accelerator)
            torch.manual_seed(17);first=Tiny()
            first_config=config(root/'resume');first_config['train']['num_epochs']=1
            run_training(first_config,system=first,loaders=(loader(),None),accelerator=self.accelerator)
            resumed=Tiny();resume_config=config(root/'resume')
            resume_config['checkpoint']['resume']=str(root/'resume/checkpoints/last.ckpt')
            resumed_result=run_training(resume_config,system=resumed,loaders=(loader(),None),accelerator=self.accelerator)
            self.assertEqual(result['global_step'],resumed_result['global_step'])
            for p,q in zip(full.parameters(),resumed.parameters()):
                torch.testing.assert_close(p,q,rtol=0,atol=0)
            io=NativeCheckpointIO();a=io.load_checkpoint(root/'full/checkpoints/last.ckpt');b=io.load_checkpoint(root/'resume/checkpoints/last.ckpt')
            self.assertEqual(a['lr_schedulers'],b['lr_schedulers'])
            for key,state in a['optimizer_states'][0]['state'].items():
                for name,value in state.items():torch.testing.assert_close(value,b['optimizer_states'][0]['state'][key][name],rtol=0,atol=0)

    def test_constant_lr_resume_preserves_optimizer_progress_without_cosine_restart(self):
        with tempfile.TemporaryDirectory() as root:
            settings = config(root)
            settings['train']['num_epochs'] = 1
            settings['optimizer']['minimum_lr_ratio'] = .1
            run_training(settings, system=Tiny(), loaders=(loader(), None), accelerator=self.accelerator)
            checkpoint = Path(root) / 'checkpoints/last.ckpt'
            before = NativeCheckpointIO().load_checkpoint(checkpoint)
            self.assertNotEqual(before['lr_schedulers'][0]['base_lrs'], [1e-5])
            settings['checkpoint']['resume'] = str(checkpoint)
            settings['optimizer'].update(schedule='constant', lr=1e-5, warmup_epochs=0)
            settings['train'].update(num_epochs=2, log_interval=1, wandb=True)
            logger = Mock()
            actual_rates = []
            def tracked_scheduler(*args):
                optimizer, scheduler = make_optimizer_scheduler(*args)
                optimizer.register_step_pre_hook(lambda opt, args, kwargs: actual_rates.append(opt.param_groups[0]['lr']))
                return optimizer, scheduler
            with patch('flowtse_rl.training.make_optimizer_scheduler', side_effect=tracked_scheduler):
                run_training(settings, system=Tiny(), loaders=(loader(), None), accelerator=self.accelerator,
                             logger_factory=lambda _: logger)
            self.assertEqual(actual_rates, [1e-5] * 3)
            rows = [call.args[0] for call in logger.log.call_args_list if 'lr-AdamW' in call.args[0]]
            self.assertTrue(rows)
            self.assertTrue(all(row['lr-AdamW'] == 1e-5 for row in rows))
            after = NativeCheckpointIO().load_checkpoint(checkpoint)
            self.assertEqual(after['global_step'], before['global_step'] + 3)
            self.assertEqual(after['lr_schedulers'][0]['last_epoch'], before['lr_schedulers'][0]['last_epoch'] + 3)
            self.assertEqual(after['lr_schedulers'][0]['base_lrs'], [1e-5])
            for state in after['optimizer_states'][0]['state'].values():
                self.assertEqual(int(state['step']), after['global_step'])
            settings['train']['num_epochs'] = 3
            run_training(settings, system=Tiny(), loaders=(loader(), None), accelerator=self.accelerator,
                         logger_factory=lambda _: logger)
            again = NativeCheckpointIO().load_checkpoint(checkpoint)
            self.assertEqual(again['global_step'], 9)
            self.assertEqual(again['optimizer_states'][0]['param_groups'][0]['lr'], 1e-5)

    def test_accumulation_partial_tail_matches_explicit_pytorch(self):
        with tempfile.TemporaryDirectory() as root:
            settings=config(root);settings['train'].update(num_epochs=1,accumulation_steps=2,gradient_clip_val=0)
            torch.manual_seed(13);native=Tiny();reference=copy.deepcopy(native)
            optimizer=torch.optim.AdamW(reference.model.parameters(),lr=.01,betas=(.9,.999),weight_decay=.01,fused=False)
            optimizer.zero_grad(set_to_none=True)
            for index,batch in enumerate(loader()):
                values={k:v for k,v in batch.items() if k!='metadata'}
                (reference(values)/2).backward()
                if index==1 or index==2:optimizer.step();optimizer.zero_grad(set_to_none=True)
            result=run_training(settings,system=native,loaders=(loader(),None),accelerator=self.accelerator)
            self.assertEqual(result['global_step'],2)
            for p,q in zip(native.parameters(),reference.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)

    def test_logged_training_and_learning_rate_have_epoch_axis_on_resume(self):
        with tempfile.TemporaryDirectory() as root:
            settings=config(root)
            settings['train']['wandb']=True
            logger=Mock()
            run_training(settings, system=Tiny(), loaders=(loader(),None), accelerator=self.accelerator,
                         logger_factory=lambda config: logger)
            rows=[call.args[0] for call in logger.log.call_args_list]
            training=[row for row in rows if 'train/fm_loss' in row]
            # Logging follows the global optimizer-step interval across epochs.
            self.assertEqual([row['epoch'] for row in training],[2/3,1.,1+1/3,2.])
            self.assertEqual([row['epoch'] for row in rows],sorted(row['epoch'] for row in rows))
            self.assertTrue(all('lr-AdamW' in row and 'trainer/global_step' in row for row in training))
            settings['train']['num_epochs']=3
            settings['checkpoint']['resume']=str(Path(root)/'checkpoints/last.ckpt')
            logger.reset_mock()
            run_training(settings, system=Tiny(), loaders=(loader(),None), accelerator=self.accelerator,
                         logger_factory=lambda config: logger)
            training=[call.args[0] for call in logger.log.call_args_list if 'train/fm_loss' in call.args[0]]
            self.assertEqual([row['epoch'] for row in training],[2+2/3,3.])

    def test_partial_validation_and_resume_log_actual_fractional_epoch(self):
        with tempfile.TemporaryDirectory() as root:
            settings=config(root)
            settings['train'].update(wandb=True,max_steps=2,val_check_interval=2,log_interval=1)
            settings['checkpoint']['monitor']='val/fm_loss'
            logger=Mock()
            score=({'val/fm_loss':.5},.01,0.)
            with patch('flowtse_rl.training.validate',return_value=score):
                run_training(settings,system=Tiny(),loaders=(loader(),object()),accelerator=self.accelerator,
                             logger_factory=lambda config:logger)
            rows=[call.args[0] for call in logger.log.call_args_list]
            self.assertEqual([row['epoch'] for row in rows if 'train/fm_loss' in row],[1/3,2/3])
            self.assertEqual([row['epoch'] for row in rows if 'val/fm_loss' in row],[2/3])
            self.assertEqual(rows[-1]['epoch'],2/3)
            saved=NativeCheckpointIO().load_checkpoint(Path(root)/'checkpoints/last.ckpt')
            self.assertEqual((saved['epoch'],saved['flowtse_trainer']['next_batch']),(0,2))
            settings['train']['max_steps']=3
            settings['checkpoint']['resume']=str(Path(root)/'checkpoints/last.ckpt')
            logger.reset_mock()
            with patch('flowtse_rl.training.validate',return_value=score):
                run_training(settings,system=Tiny(),loaders=(loader(),object()),accelerator=self.accelerator,
                             logger_factory=lambda config:logger)
            rows=[call.args[0] for call in logger.log.call_args_list]
            self.assertTrue(all(row['epoch']==1. for row in rows))

    def test_old_lightning_partial_epoch_is_not_skipped(self):
        legacy=dict(epoch=0,global_step=400,loops={'fit_loop':{'epoch_progress':{'current':{'processed':1}},
            'epoch_loop.batch_progress':{'current':{'processed':400,'completed':400}}}})
        self.assertEqual(resume_position(legacy,1011,1),(0,400))
        legacy['loops']['fit_loop']['epoch_loop.batch_progress']['current'].update(processed=1011,completed=1011)
        self.assertEqual(resume_position(legacy,1011,1),(1,0))
        legacy['loops']['fit_loop']['epoch_loop.batch_progress']['current'].update(processed=3,completed=3)
        with self.assertRaisesRegex(ValueError,'accumulation'):resume_position(legacy,1011,2)



    def test_top_three_aliases_last_and_atomic_recovery(self):
        with tempfile.TemporaryDirectory() as root:
            settings=config(root);manager=CheckpointManager(settings)
            for epoch,score in enumerate([1.,3.,2.,4.]):
                manager.save(lambda ranking:dict(callbacks={'NativeCheckpointManager':ranking},epoch=epoch,global_step=epoch,
                    hyper_parameters={'config':settings}),epoch,epoch,{'val/pesq':score},last=True)
            self.assertEqual(sorted(manager.ranked.values()),[2.,3.,4.])
            self.assertTrue((Path(root)/'checkpoints/best.ckpt').is_symlink())
            restored=CheckpointManager(settings);restored.restore(manager.io.load_checkpoint(Path(root)/'checkpoints/last.ckpt'))
            self.assertEqual(restored.ranked,manager.ranked)
            self.assertFalse(list((Path(root)/'checkpoints').glob('*.tmp')))


if __name__=='__main__':unittest.main()
