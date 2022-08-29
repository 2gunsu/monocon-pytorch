import os
import sys
import glob
import torch
import numpy as np
import pandas as pd

from yacs.config import CfgNode
from typing import Dict, Union, Any
from datetime import datetime, timedelta
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.decorators import decorator_timer
from utils.engine_utils import tprint, export_cfg, load_cfg, count_trainable_params


class BaseEngine:
    def __init__(self, 
                 cfg: Union[str, CfgNode], 
                 auto_resume: bool = True,
                 is_test: bool = False):
        
        # Configuration
        if isinstance(cfg, str):
            self.cfg = load_cfg(cfg_file=cfg)
        elif isinstance(cfg, CfgNode):
            self.cfg = cfg
        else:
            raise Exception("Argument 'cfg' must be either a string or a CfgNode.")
        
        self.version = (cfg.VERSION)
        self.description = (cfg.DESCRIPTION)
        
        # Counter Params (1-based Numbering)
        self.epochs = 1
        
        target_epochs = (cfg.SOLVER.OPTIM.NUM_EPOCHS)
        assert (self.epochs <= target_epochs), \
            f"Argument 'target_epochs'({target_epochs}) must be equal to or greater than 'epochs'({self.epochs})."
        self.target_epochs = target_epochs
        self.global_iters = 1
        
        # Period Params
        self.log_period = (cfg.PERIOD.LOG_PERIOD)
        self.val_period = (cfg.PERIOD.EVAL_PERIOD)
        
        # Dataset and Data-Loader
        self.train_dataset, self.train_loader = \
            self.build_loader(is_train=True) if not is_test else (None, None)
        self.test_dataset, self.test_loader = self.build_loader(is_train=False)
        
        # Model, Optimizer, Schduler
        self.model = self.build_model()
        self.optimizer, self.scheduler = \
            self.build_solver() if not is_test else (None, None)
        
        # Directory
        self.root = (cfg.OUTPUT_DIR)
        self.writer_dir = os.path.join(self.root, 'tf_logs')
        self.weight_dir = os.path.join(self.root, 'checkpoints')
        
        if not is_test:
            exist = False
            if os.path.isdir(self.weight_dir) and auto_resume:
                pth_files = sorted(glob.glob(os.path.join(self.weight_dir, r'*.pth')))
                if len(pth_files) > 0:
                    exist = True
                    latest_weight = pth_files[-1]
                    self.load_checkpoint(latest_weight)
                    tprint(f"Existing checkpoint '{latest_weight}' is found and loaded automatically.")
            
            if not exist:
                for dir_ in [self.writer_dir, self.weight_dir]:
                    os.makedirs(dir_, exist_ok=True)
        
            # Writer
            self.writer = SummaryWriter(self.writer_dir)
        
        # Storage
        self.epoch_times = []
        self.entire_losses = []
        
    def build_model(self):
        raise NotImplementedError
    
    def build_solver(self):
        raise NotImplementedError
    
    def build_loader(self, is_train: bool):
        raise NotImplementedError
        
    def train(self, resume_from: str = None) -> None:
        assert torch.cuda.is_available(), "CUDA is not available."
        assert (self.epochs < self.target_epochs), \
            "Argument 'target_epochs' must be equal to or greater than 'epochs'."
            
        # Print Info
        self._print_engine_info()
            
        # Export Current Configuration
        export_cfg(self.cfg, os.path.join(self.root, 'config.yaml'))
        
        # Resume Training if 'resume_from' is specified.
        if (resume_from is not None):
            self.load_checkpoint(resume_from)
            tprint(f"Training resumes from '{resume_from}'. (Start Epoch: {self.epochs})")
        
        # Start Training
        tprint(f"Training will be proceeded from epoch {self.epochs} to epoch {self.target_epochs}.")
        tprint(f"Result files will be saved to '{self.root}'.")
        for epoch in range(self.epochs, self.target_epochs + 1):
            print(f" Epoch {self.epochs:3d} / {self.target_epochs:3d} ".center(90, "="))
            
            avg_loss, elapsed_time = self.train_one_epoch()
            
            self.epoch_times.append(elapsed_time)
            time_info = self._get_time_info()
            
            print(f"\n- Average Loss: {avg_loss:.3f}")
            print(f"- Epoch Time: {time_info['epoch_time']}")
            print(f"- Remain Time: {time_info['remain_time']}")
            print(f"- Estimated End-Time: {time_info['end_time']}")
            
            # Validation
            if (self.val_period > 0) and (epoch % self.val_period == 0):
                self.model.eval()
                
                tprint(f"Evaluating on Epoch {epoch}...", indent=True)
                eval_dict = self.evaluate()

                # Write evaluation results to tensorboard.
                self._update_dict_to_writer(eval_dict, tag='eval')
                
                self.model.train()
                
                # Save Checkpoint (.pth)
                self.save_checkpoint(post_fix=None)
        
        # Save Final Checkpoint (.pth)
        self.save_checkpoint(post_fix='final')
    
    @decorator_timer
    def train_one_epoch(self):
        raise NotImplementedError
                
    @torch.no_grad()
    def evaluate(self):
        raise NotImplementedError
            
    @torch.no_grad()
    def test(self):
        raise NotImplementedError
    
    def save_checkpoint(self, 
                        post_fix: str = None,
                        save_after_update: bool = True,
                        verbose: bool = True) -> None:
        
        save_epoch = self.epochs
        if save_after_update:
            save_epoch -= 1
        
        if (post_fix is None):
            file_name = f'epoch_{save_epoch:03d}.pth'
        else:
            file_name = f'epoch_{save_epoch:03d}_{post_fix}.pth'
        file_path = os.path.join(self.weight_dir, file_name)
        
        # Hard-coded
        attr_except = ['cfg', 'writer', 'train_loader', 'test_loader', 'train_dataset', 'test_dataset'
                       'model', 'optimizer', 'scheduler',]
        attrs = {k: v for k, v in self.__dict__.items() \
            if not callable(getattr(self, k)) and (k not in attr_except)}
        engine_dict = {
            'engine_attrs': attrs,
            'state_dict': {
                'model': self.model.state_dict() \
                    if (self.model is not None) else None,
                'optimizer': self.optimizer.state_dict() \
                    if (self.optimizer is not None) else None,
                'scheduler': self.scheduler.state_dict() \
                    if (self.scheduler is not None) else None,
            }
        }
        
        torch.save(engine_dict, file_path)
        if verbose:
            tprint(f"Checkpoint is saved to '{file_path}'.")
    
    def load_checkpoint(self, 
                        ckpt_file: str, 
                        verbose: bool = False) -> None:
        
        engine_dict = torch.load(ckpt_file)
        
        
        # Load Engine Attributes
        attrs = engine_dict['engine_attrs']
        for attr_k, attr_v in attrs.items():
            setattr(self, attr_k, attr_v)
        
            
        state_dict = engine_dict['state_dict']
        
        # Load Model
        if (state_dict['model'] is not None) and (self.model is not None):
            self.model.load_state_dict(state_dict['model'])
        
        # Load Optimizer
        if (state_dict['optimizer'] is not None) and (self.optimizer is not None):
            self.optimizer.load_state_dict(state_dict['optimizer'])
        
        # Load Scheduler
        if (state_dict['scheduler'] is not None) and (self.scheduler is not None):
            self.scheduler.load_state_dict(state_dict['scheduler'])
        
        if verbose:
            tprint(f"Checkpoint is loaded from '{ckpt_file}'.")
            
    def _epoch_update(self):
        self.epochs += 1
    
    def _iter_update(self):
        self.global_iters += 1
        
    def _update_dict_to_writer(self, data: Dict[str, Union[torch.Tensor, float]], tag: str):
        for k, v in data.items():
            self.writer.add_scalar(f'{tag}/{k}',
                                   scalar_value=v if isinstance(v, float) else v.detach().item(),
                                   global_step=self.global_iters)
       
    def _get_time_info(self) -> Dict[str, str]:
        # Average time for single epoch
        avg_epoch_time = np.mean(self.epoch_times)
        last_epoch_time_str = str(timedelta(seconds=self.epoch_times[-1]))[:-7]
        
        # Remain Time
        remain_epochs = (self.target_epochs - (self.epochs - 1))
        remain_time = (remain_epochs * avg_epoch_time)
        remain_time_str = str(timedelta(seconds=remain_time))[:-7]
        
        # End Time
        current_time = datetime.now()
        end_time = current_time + timedelta(seconds=remain_time)
        end_time_str = str(end_time)[:-7]
        
        return {
            'epoch_time': last_epoch_time_str,
            'remain_time': remain_time_str,
            'end_time': end_time_str}
        
    def _print_engine_info(self):
        print(f"\n==================== Engine Info ====================")
        print(f"- Root: {self.root}")
        print(f"- Version: {self.version}")
        print(f"- Description: {self.description}")
        
        print(f"\n- Seed: {self.cfg.SEED}")
        print(f"- Device: GPU {self.cfg.GPU_ID} ({torch.cuda.get_device_name(self.cfg.GPU_ID)})")
        
        print(f"\n- Model: {self.model.__class__.__name__} (# Params: {count_trainable_params(self.model)})")
        print(f"- Optimizer: {self.optimizer.__class__.__name__}")
        print(f"- Scheduler: {self.scheduler.__class__.__name__}\n")
        
        print(f"- Epoch Progress: {self.epochs}/{self.target_epochs}")
        print(f"- # Train Samples: {len(self.train_dataset)}")
        print(f"- # Test Samples: {len(self.test_dataset)}")
        print(f"=====================================================\n")
        

    @property
    def current_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']
    
    @property
    def current_device(self) -> torch.device:
        return torch.device(f'cuda:{self.cfg.GPU_ID}')
