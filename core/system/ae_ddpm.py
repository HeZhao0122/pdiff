import pdb

import hydra.utils
import pytorch_lightning as pl
import torch
from typing import Any
import statistics
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseSystem
from core.utils.ddpm import *
from core.utils.utils import *
from core.module.prelayer.latent_transformer import Param2Latent
from .ddpm import DDPM


class AE_DDPM(DDPM):
    def __init__(self, config, **kwargs):
        super(AE_DDPM, self).__init__(config)
        ae_model = hydra.utils.instantiate(config.system.ae_model)
        self.in_dim = config.system.in_dim
        # input_dim = config.system.ae_model.in_dim
        # # input_dim = config.system.in_dim
        # input_noise = torch.randn((1, input_dim))
        # latent_dim = ae_model.encode(input_noise).shape
        # config.system.model.arch.model.in_dim = latent_dim[-1] * latent_dim[-2]
        super(AE_DDPM, self).__init__(config)
        # self.pos = self.task.get_pos().unsqueeze(0).float().cuda()
        # self.ae_model = torch.load('./param_data/AE_all.pt', map_location='cpu')
        self.ae_model = ae_model
        self.save_hyperparameters()
        self.split_epoch = self.train_cfg.split_epoch
        self.dataset_list = self.train_cfg.datasets
        self.save_epoch = self.train_cfg.generate_epoch
        self.fine_tune_epoch = self.train_cfg.fine_tune_epoch
        self.loss_func = nn.MSELoss()
        self.reg_func = nn.KLDivLoss(log_target=True)
        self.train_loader = self.task
        # self.use_pos = config.system.ae_model.pos
        total_param = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'Total param: {total_param}')

    def ae_forward(self, batch, **kwargs):

        loss = self.ae_model(batch)
        # output = self.ae_model(batch, train=True)
        # loss = output
        # loss = self.loss_func(batch, output, **kwargs)
        # kl_loss = latent.kl()
        # reg_loss = self.reg_func(F.softmax(output), F.softmax(batch))
        # self.log('epoch', self.current_epoch)
        self.log('ae_loss', loss.cpu().detach().mean().item(), on_epoch=True, prog_bar=True, logger=True)
        # print(loss)
        # self.log('kl_loss', kl_loss.cpu().detach().mean().item(), on_epoch=True, prog_bar=True, logger=True)
        return loss
        # return loss

    def dec_fine_tune(self, batch, **kwargs):
        loss = self.ae_model(batch, fine_tune=False)
        # loss = self.loss_func(batch, output, **kwargs)
        self.log('dec_loss', loss.cpu().detach().mean().item(), on_epoch=True, prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx, **kwargs):
        ddpm_optimizer, ae_optimizer = self.configure_optimizers()
        param, mask, enc, label, shapes = batch
        if self.current_epoch < self.split_epoch:
            loss = self.ae_forward((param, mask), **kwargs)
            ae_optimizer.zero_grad()
            self.manual_backward(loss)
            ae_optimizer.step()
        else:
            loss = self.forward((param, enc, shapes), **kwargs)
            ddpm_optimizer.zero_grad()
            self.manual_backward(loss)
            ddpm_optimizer.step()
        # else:
        #     self.ae_model.freeze_encoder()
        #     loss = self.dec_fine_tune(batch, **kwargs)
        #     dec_optimizer.zero_grad()
        #     self.manual_backward(loss)
        #     dec_optimizer.step()

        if hasattr(self, 'lr_scheduler'):
            self.lr_scheduler.step()
        return {'loss': loss}

    def pre_process(self, batch):
        latent = self.ae_model.encode(batch)
        self.latent_shape = latent.shape[-2:]
        return latent

    def post_process(self, outputs):
        # pdb.set_trace()
        outputs = outputs.reshape(-1, *self.latent_shape)
        return self.ae_model.decode(outputs)

    def validation_step(self, pbatch, batch_idx, **kwargs: Any):
        batch = pbatch[0]
        mask = pbatch[1]
        labels = pbatch[3]
        dims = pbatch[4][:, 0]
        # batch = batch.view(batch.shape[0], -1)[:, :self.in_dim]
        if self.current_epoch < self.split_epoch:
            # todo
            good_param = batch[:10]
            good_mask = mask[:10]
            good_dim = dims[:10]
            input_accs = []
            for i, param in enumerate(good_param):
                param = param.view(-1)[:self.in_dim]
                acc, test_loss, output_list = self.task_val_func(param, good_dim[i])
                input_accs.append(acc)

            print("input model accuracy:{}".format(input_accs))

            """
            AE reconstruction parameters
            """
            print('---------------------------------')
            print('Test the AE model')
            print(f'Hidden dim: {good_dim}')
            ae_rec_accs = []
            # latent, ae_params = self.ae_model.reconstruct(good_param)
            latent = self.ae_model.encode(good_param)
            shape_latent = self.ae_model.encode(good_mask.float().reshape(good_param.shape))
            if self.current_epoch == self.split_epoch-1:
                all_l = self.ae_model.encode(batch)
                torch.save(self.ae_model, './param_data/AE.pt')
                torch.save(all_l, './param_data/latent.pt')
                torch.save(labels, './param_data/labels.pt')
                print(f'Save Latent!')
            print("latent shape:{}".format(latent.shape))
            ae_params = self.ae_model.decode(latent + shape_latent)
            print("ae params shape:{}".format(ae_params.shape))
            ae_params = ae_params.cpu()
            for i, param in enumerate(ae_params):
                param = param.to(batch.device)
                param = param.view(-1)[:self.in_dim]
                acc, test_loss, output_list = self.task_val_func(param, good_dim[i])
                ae_rec_accs.append(acc)

            # lists = {dataset: [] for dataset in self.dataset_list}
            # for param, label in zip(ae_params, labels):
            #     label = int(label)
            #     lists[self.dataset_list[label]].append(param.view(1, -1))

            # for dataset, parameters in lists.items():
            #     path = f'./param_data/{dataset}/generate.pt'
            #     parameters = torch.cat(parameters, dim=0)
            #     torch.save(parameters, path)
            #     print(f'Save in {path}')

            best_ae = max(ae_rec_accs)
            print(f'AE reconstruction models accuracy:{ae_rec_accs}')
            print(f'AE reconstruction models best accuracy:{best_ae}')
            print('---------------------------------')
            self.log('ae_acc', best_ae)
            self.log('best_g_acc', 0)
        else:
            dict = super(AE_DDPM, self).validation_step(pbatch, batch_idx, **kwargs)
            self.log('ae_acc', 94.3)
            self.log('ae_loss', 0 )
            return dict
    # def test_step(self,batch, batch_idx, **kwargs: Any):
    #     ae_rec_accs = []
    #     num_samples = 100
    #     ae_params = self.ae_model.sample(num_samples)
    #     ae_params = ae_params.cpu()
    #     for i, param in enumerate(ae_params):
    #         param = param.to(batch.device)
    #         acc, test_loss, output_list = self.task_func(param)
    #         ae_rec_accs.append(acc)
    #
    #     best_ae = max(ae_rec_accs)
    #     print(f'Test AE reconstruction models accuracy:{ae_rec_accs}')
    #     print(f'Test AE reconstruction models best accuracy:{best_ae}')
    #     print(f'Test AE reconstruction models mean accuracy:{statistics.mean(ae_rec_accs)}')
    #     print(f'Test AE reconstruction models median accuracy:{statistics.median(ae_rec_accs)}')
    #     print('---------------------------------')
    #     dict = super(AE_DDPM, self).test_step(batch, batch_idx, **kwargs)
    #     return dict

    def configure_optimizers(self, **kwargs):
        ae_parmas = self.ae_model.parameters()
        ddpm_params = self.model.parameters()

        self.ddpm_optimizer = hydra.utils.instantiate(self.train_cfg.optimizer, ddpm_params)
        self.ae_optimizer = hydra.utils.instantiate(self.train_cfg.optimizer, ae_parmas)

        if 'lr_scheduler' in self.train_cfg and self.train_cfg.lr_scheduler is not None:
            self.lr_scheduler = hydra.utils.instantiate(self.train_cfg.lr_scheduler)

        return self.ddpm_optimizer, self.ae_optimizer


class P_DDPM(DDPM):
    def __init__(self, config, **kwargs):

        super(P_DDPM, self).__init__(config)
        self.save_hyperparameters()
        self.split_epoch = self.train_cfg.split_epoch
        total_param = sum(p.numel() for p in self.parameters() if p.requires_grad)
        config.system.model.arch.model.in_dim = config.system.ae_model.in_dim
        print(f'Total param: {total_param}')

    def ae_forward(self, batch, **kwargs):
        output = self.ae_model(batch)
        # loss = output
        loss = self.loss_func(batch, output, **kwargs)
        # reg_loss = self.reg_func(F.softmax(output), F.softmax(batch))
        # self.log('epoch', self.current_epoch)
        self.log('ae_loss', loss.cpu().detach().mean().item(), on_epoch=True, prog_bar=True, logger=True)
        # self.log('kl_loss_*1e+3', reg_loss.cpu().detach().mean().item(), on_epoch=True, prog_bar=True, logger=True)
        return loss
        # return loss

    def training_step(self, batch, batch_idx, **kwargs):
        ddpm_optimizer = self.optimizers()
        batch = batch.unsqueeze(1)
        loss = self.forward(batch, **kwargs)
        ddpm_optimizer.zero_grad()
        self.manual_backward(loss)
        ddpm_optimizer.step()

        if hasattr(self, 'lr_scheduler'):
            self.lr_scheduler.step()
        return {'loss': loss}

    def pre_process(self, batch):

        return batch

    def post_process(self, outputs):
        # pdb.set_trace()
        return outputs

    def validation_step(self, batch, batch_idx, **kwargs: Any):
        batch = batch.unsqueeze(1)
        dict = super(P_DDPM, self).validation_step(batch, batch_idx, **kwargs)
        self.log('ae_acc', 94.3)
        self.log('ae_loss', 0 )
        return dict

    def configure_optimizers(self, **kwargs):
        ddpm_params = self.model.parameters()

        self.ddpm_optimizer = hydra.utils.instantiate(self.train_cfg.optimizer, ddpm_params)

        if 'lr_scheduler' in self.train_cfg and self.train_cfg.lr_scheduler is not None:
            self.lr_scheduler = hydra.utils.instantiate(self.train_cfg.lr_scheduler)

        return self.ddpm_optimizer

