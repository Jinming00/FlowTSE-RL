"""Exercise LR boundaries, update-budget conversion, and resume continuity."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
import torch
from flowtse_rl.config import BASELINE_CONFIG,load_config,validate_config,resolve_warmup_steps
from flowtse_rl.engine import make_optimizer_scheduler

ROOT=Path(__file__).resolve().parents[1]
def configured(config,total=4000):
    config=deepcopy(config);config['model'].update(hidden_size=32,depth=2,num_heads=4,mlp_ratio=2)
    module=SimpleNamespace(model=torch.nn.Linear(2,1))
    optimizer,scheduler=make_optimizer_scheduler(module,config,total)
    return module,optimizer,scheduler

class EpochWarmupTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config=load_config(BASELINE_CONFIG)
        self.config['train']['num_epochs']=200  # Fixed diagnostic scheduler budget.
    def test_epoch_boundaries_and_cosine_floor_use_real_optimizer_updates(self):
        _,optimizer,scheduler=configured(self.config)
        self.assertEqual(optimizer.defaults['betas'],[.9,.999])
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'],1e-6)
        expected={99:1e-4,100:1e-4,2050:5.5e-5,4000:1e-5}
        for step in range(1,4001):
            optimizer.step();scheduler.step()
            if step in expected:self.assertAlmostEqual(optimizer.param_groups[0]['lr'],expected[step],places=12)
    def test_epoch_warmup_tracks_changed_update_budgets_and_resume(self):
        self.assertEqual(resolve_warmup_steps(self.config,4000),100)
        self.assertEqual(resolve_warmup_steps(self.config,2000),50)
        a,oa,sa=configured(self.config)
        for _ in range(47):oa.step();sa.step()
        b,ob,sb=configured(self.config);ob.load_state_dict(oa.state_dict());sb.load_state_dict(sa.state_dict())
        for _ in range(80):
            oa.step();sa.step();ob.step();sb.step()
            self.assertEqual(oa.param_groups[0]['lr'],ob.param_groups[0]['lr'])
    def test_legacy_step_warmup_and_invalid_ambiguous_configuration(self):
        config=deepcopy(self.config)
        config['optimizer'].pop('warmup_epochs')
        config['optimizer']['warmup_steps']=2000
        _,optimizer,scheduler=configured(config)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'],5e-8)
        optimizer.step();scheduler.step();self.assertAlmostEqual(optimizer.param_groups[0]['lr'],5e-8)
        optimizer.step();scheduler.step();self.assertAlmostEqual(optimizer.param_groups[0]['lr'],1e-7)
        for key,value in [('warmup_steps',2000),('warmup_epochs',-1),('warmup_epochs',201),('warmup_epochs','5')]:
            invalid=deepcopy(self.config);invalid['optimizer'][key]=value
            with self.assertRaises(ValueError):validate_config(invalid)
if __name__=='__main__':unittest.main()
