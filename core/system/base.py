import pdb

import pytorch_lightning as pl
from core.data.parameters import PData
import abc
import hydra
from omegaconf import OmegaConf
import torch.optim.lr_scheduler
import warnings
import yaml
from typing import Optional, Union, List, Dict, Any, Sequence
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
import types
from core.tasks import tasks


def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)


def merge_configs(cfg):
    includes = cfg.get('tasks', [])
    combined_config = {}

    for config_file in includes:
        config_path = f'./configs/task/{config_file}.yaml'
        config = load_yaml(config_path)
        combined_config[config_file] = OmegaConf.create(config)

    return combined_config


class BaseSystem(pl.LightningModule, abc.ABC):
    def __init__(self, cfg):
        super(BaseSystem, self).__init__()
        # when save  hyperparameters, the self.task will be ignored
        self.save_hyperparameters()
        self.main_cfg = cfg.task
        task_cfg = merge_configs(self.main_cfg)
        self.automatic_optimization = False
        self.config = cfg.system
        self.train_cfg = self.config.train
        self.model_cfg = self.config.model
        self.model = self.build_model()
        self.loss_func = self.build_loss_func()
        self.data_transform = self.build_data_transform()
        self.build_task(task_cfg)
        self.get_datasets()

    def build_task(self, task_cfg, **kwargs):
        task_dict = {}
        idx = 0
        for name, cfg in task_cfg.items():
            task_dict[idx] = tasks[cfg.name](cfg)
            idx += 1
        # self.task = tasks[task_cfg.name](task_cfg)
        self.task = task_dict

    def get_task(self, idx):
        return self.task[idx]

    def get_datasets(self):
        datasets = [task.cfg.data.dataset for idx, task in self.task.items()]
        self.dataset_list = datasets

    def get_param_data(self):
        # test_shapes = self.get_test_shape()
        param_data = PData(self.main_cfg.param, self.task)

        return param_data

    def build_data_transform(self):
        if 'data_transform' in self.config and self.config.data_transform is not None:
            return hydra.utils.instantiate(self.config.data_transform)
        else:
            return None

    def build_trainer(self):
        trainer = hydra.utils.instantiate(self.train_cfg.trainer)
        pdb.set_trace()
        return trainer

    def task_func(self, input, hidden_dim, idx):
        return self.task[idx].test_g_model(input, hidden_dim)

    def task_val_func(self, input, hidden_dim, idx):
        return self.task[idx].val_g_model(input, hidden_dim)

    def training_step(self, batch, batch_idx, **kwargs):
        optimizer = self.optimizers()
        loss = self.forward(batch, **kwargs)
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()

        if hasattr(self, 'lr_scheduler'):
            self.lr_scheduler.step()

        return {'loss': loss}

    def build_model(self, **kwargs):
        model = hydra.utils.instantiate(self.model_cfg.arch)
        return model

    def build_loss_func(self):
        if 'loss_func' in self.train_cfg:
            loss_func = hydra.utils.instantiate(self.train_cfg.loss_func)
            return loss_func
        else:
            warnings.warn("No loss function is specified, using default loss function")


    def configure_optimizers(self, **kwargs):
        params = self.model.parameters()
        self.optimizer = hydra.utils.instantiate(self.train_cfg.optimizer, params)

        if 'lr_scheduler' in self.train_cfg and self.train_cfg.lr_scheduler is not None:
            self.lr_scheduler = hydra.utils.instantiate(self.train_cfg.lr_scheduler)

        return self.optimizer

    def validation_step(self, batch, batch_idx, **kwargs):
        # TODO using task layer
        pass

    @abc.abstractmethod
    def forward(self, x, **kwargs):
        raise NotImplementedError