import os
import pdb

import hydra.utils
from statistics import median

from .base_task import BaseTask
from core.data.vision_dataset import VisionData
from core.data.Graph_dataset import GraphCFTData, NodeCFTData
from core.data.parameters import PData
import torch.distributions.normal as normal
from core.utils.utils import *
import torch.nn as nn
import datetime
import statistics
from core.utils import *
import glob
import omegaconf
import json
import copy


class GraphCFT(BaseTask):
    def __init__(self, config, **kwargs):
        super(GraphCFT, self).__init__(config, **kwargs)
        self.train_loader = self.task_data.train_dataloader()
        self.eval_loader = self.task_data.val_dataloader()
        self.test_loader = self.task_data.test_dataloader()

    def init_task_data(self):
        return GraphCFTData(self.cfg.data)

    def set_param_data(self):
        param_data = PData(self.cfg.param)
        self.model = param_data.get_model()
        self.train_layer = param_data.get_train_layer()
        return param_data

    def test_g_model(self, input):
        net = self.model
        train_layer = self.train_layer
        param = input
        target_num = 0
        for name, module in net.named_parameters():
            if name in train_layer:
                target_num += torch.numel(module)

        params_num = torch.squeeze(param).shape[0]  # + 30720
        try:
            assert (target_num == params_num)
        except:
            print(f'real param num: {target_num}')
        param = torch.squeeze(param)
        model = partial_reverse_tomodel(param, net, train_layer).to(param.device)

        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        output_list = []

        with torch.no_grad():
            for data in self.test_loader:
                data = data.cuda()
                # inputs, edge_index, targets, batch, = data.x, data.edge_index, data.y, data.batch
                outputs = net(data.x, data.edge_index, data.batch)
                loss = F.cross_entropy(outputs, data.y)

                test_loss += loss.item()
                predicted = outputs.argmax(dim=1)
                total += data.y.size(0)
                correct += int((predicted == data.y).sum())

        test_loss /= total
        acc = 100. * correct / total
        del model
        return acc, test_loss, output_list

    def val_g_model(self, input):
        net = self.model
        train_layer = self.train_layer
        param = input
        target_num = 0
        for name, module in net.named_parameters():
            if name in train_layer:
                target_num += torch.numel(module)
        params_num = torch.squeeze(param).shape[0]  # + 30720
        assert (target_num == params_num)
        param = torch.squeeze(param)
        model = partial_reverse_tomodel(param, net, train_layer).to(param.device)

        model.eval()
        test_loss = 0
        correct = 0
        total = 0

        output_list = []

        with torch.no_grad():
            # ToDO 原文里面用的是train loader！！！
            for data in self.train_loader:
                data = data.cuda()
                # inputs, edge_index, targets, batch, = data.x, data.edge_index, data.y, data.batch
                outputs = net(data.x, data.edge_index, data.batch)
                loss = F.cross_entropy(outputs, data.y)

                test_loss += loss.item()
                predicted = outputs.argmax(dim=1)
                total += data.y.size(0)
                correct += int((predicted == data.y).sum())

        test_loss /= total
        acc = 100. * correct / total
        del model
        return acc, test_loss, output_list

    def train_for_data(self):
        net = self.build_model()
        optimizer = self.build_optimizer(net)
        criterion = nn.CrossEntropyLoss()
        scheduler = hydra.utils.instantiate(self.cfg.lr_scheduler, optimizer)
        epoch = self.cfg.epoch
        save_num = self.cfg.save_num_model
        all_epoch = epoch + save_num

        best_acc = 0
        train_loader = self.train_loader
        eval_loader = self.eval_loader
        test_loader = self.test_loader
        train_layer = self.cfg.train_layer

        if train_layer == 'all':
            train_layer = [name for name, module in net.named_parameters()]

        data_path = getattr(self.cfg, 'save_root', 'param_data')

        tmp_path = os.path.join(data_path, 'tmp_{}'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
        # tmp_path = os.path.join(data_path, 'tmp')
        final_path = os.path.join(data_path, self.cfg.data.dataset)

        os.makedirs(tmp_path, exist_ok=True)
        os.makedirs(final_path, exist_ok=True)


        save_model_accs = []
        parameters = []

        net = net.cuda()
        for i in range(0, all_epoch):
            self.train(net, criterion, optimizer, train_loader, i)
            acc = self.test(net, criterion, eval_loader)
            best_acc = max(acc, best_acc)

            if i == (epoch - 1):
                print("saving the model")
                torch.save(net, os.path.join(tmp_path, "whole_model.pth"))
                fix_partial_model(train_layer, net)
                parameters = []
            if i >= epoch:
                parameters.append(state_part(train_layer, net))
                save_model_accs.append(acc)
                if len(parameters) == 10 or i == all_epoch - 1:
                    torch.save(parameters, os.path.join(tmp_path, "p_data_{}.pt".format(i)))
                    parameters = []

            scheduler.step()
        print("training over")
        test_acc = self.test(net, criterion, test_loader)
        print(f"Test acc: {test_acc}")

        pdata = []
        for file in glob.glob(os.path.join(tmp_path, "p_data_*.pt")):
            buffers = torch.load(file)
            for buffer in buffers:
                param = []
                for key in buffer.keys():
                    if key in train_layer:
                        param.append(buffer[key].data.reshape(-1))
                param = torch.cat(param, 0)
                pdata.append(param)
        batch = torch.stack(pdata)
        mean = torch.mean(batch, dim=0)
        std = torch.std(batch, dim=0)

        # check the memory of p_data
        useage_gb = get_storage_usage(tmp_path)
        print(f"path {tmp_path} storage usage: {useage_gb:.2f} GB")

        state_dic = {
            'pdata': batch.cpu().detach(),
            'mean': mean.cpu(),
            'std': std.cpu(),
            'model': torch.load(os.path.join(tmp_path, "whole_model.pth")),
            'train_layer': train_layer,
            'performance': save_model_accs,
            'cfg': config_to_dict(self.cfg)
        }

        torch.save(state_dic, os.path.join(final_path, "data.pt"))
        json_state = {
            'cfg': config_to_dict(self.cfg),
            'performance': save_model_accs

        }
        json.dump(json_state, open(os.path.join(final_path, "config.json"), 'w'))

        # copy the code file(the file) in state_save_dir
        shutil.copy(os.path.abspath(__file__), os.path.join(final_path,
                                                            os.path.basename(__file__)))

        # delete the tmp_path
        shutil.rmtree(tmp_path)
        print("data process over")
        return {'save_path': final_path}

    def train(self, net, criterion, optimizer, trainloader, epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, data in enumerate(trainloader):
            data = data.cuda()
            # inputs, edge_index, targets, batch, = data.x, data.edge_index, data.y, data.batch
            optimizer.zero_grad()
            outputs = net(data.x, data.edge_index, data.batch)
            loss = criterion(outputs, data.y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            total += data.y.size(0)
            correct += int((predicted == data.y).sum())

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    def test(self, net, criterion, testloader):
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(testloader):
                data = data.cuda()
                # inputs, edge_index, targets, batch, = data.x, data.edge_index, data.y, data.batch
                outputs = net(data.x, data.edge_index, data.batch)
                loss = criterion(outputs, data.y)

                test_loss += loss.item()
                predicted = outputs.argmax(dim=1)
                total += data.y.size(0)
                correct += int((predicted == data.y).sum())
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            return 100. * correct / total


class NodeCFT(BaseTask):
    def __init__(self, config, **kwargs):
        super(NodeCFT, self).__init__(config, **kwargs)
        self.graph = self.task_data.get_graph
        self.train_mask = self.task_data.train_dataloader()
        self.val_mask = self.task_data.val_dataloader()
        self.test_mask = self.task_data.test_dataloader()

    def init_task_data(self):
        return NodeCFTData(self.cfg.data)

    def set_param_data(self):
        param_data = PData(self.cfg.param)
        self.model = param_data.get_model()
        self.train_layer = param_data.get_train_layer()
        return param_data

    def get_param_dims(self):
        param_data = PData(self.cfg.param)
        return param_data.get_param_dim()

    def test_g_model(self, input, hidden_dim=None):
        if hidden_dim:
            self.cfg.model.hidden_dim = hidden_dim
            net = self.build_model()
            # import pdb;pdb.set_trace()
        else:
            net = self.model
        train_layer = self.train_layer
        # test_layer = self.test_layer
        param = input
        target_num = 0
        for name, module in net.named_parameters():
            # print(name)
            if name in train_layer:
                target_num += torch.numel(module)

        params_num = torch.squeeze(param).shape[0]  # + 30720
        # try:
        #     assert (target_num == params_num)
        # except:
        #     print(f'real_num: {target_num}')
        param = torch.squeeze(param)
        model = partial_reverse_tomodel(param, net, train_layer).to(param.device)

        model.eval()
        # test_loss = 0
        correct = 0
        total = 0

        output_list = []

        with torch.no_grad():
            data = self.graph.cuda()
            # inputs, edge_index, targets, batch, = data.x, data.edge_index, data.y, data.batch
            outputs = model(data.x, data.edge_index)
            loss = F.cross_entropy(outputs[self.test_mask], data.y[self.test_mask])

            predicted = outputs.argmax(dim=1)
            total += data.y[self.test_mask].size(0)
            correct += int((predicted[self.test_mask] == data.y[self.test_mask]).sum())

        test_loss = loss.item()
        acc = 100. * correct / total
        del model
        return acc, test_loss, output_list

    def val_g_model(self, input, hidden_dim=None):
        if hidden_dim:
            self.cfg.model.hidden_dim = hidden_dim
            net = self.build_model()
        # import pdb;pdb.set_trace()
        else:
            net = self.model
        train_layer = self.train_layer
        # test_layer = self.test_layer
        param = input
        target_num = 0
        for name, module in net.named_parameters():
            if name in train_layer:
                target_num += torch.numel(module)
        params_num = torch.squeeze(param).shape[0]  # + 30720
        # assert (target_num == params_num)
        param = torch.squeeze(param)
        model = partial_reverse_tomodel(param, net, train_layer).to(param.device)

        model.eval()
        # test_loss = 0
        correct = 0
        total = 0

        output_list = []

        with torch.no_grad():
            data = self.graph.cuda()
            # inputs, edge_index, targets, batch, = data.x, data.edge_index, data.y, data.batch
            outputs = model(data.x, data.edge_index)
            loss = F.cross_entropy(outputs[self.train_mask], data.y[self.train_mask])

            predicted = outputs.argmax(dim=1)
            total += data.y[self.train_mask].size(0)
            correct += int((predicted[self.train_mask] == data.y[self.train_mask]).sum())

        test_loss = loss.item()
        acc = 100. * correct / total
        del model
        return acc, test_loss, output_list

    def noise_test(self, params, onet):
        '''
        params: one set of params
        '''
        net = copy.deepcopy(onet)
        if self.cfg.train_layer != 'all':
            train_layer = self.cfg.train_layer
        else:
            train_layer = [name for name, module in net.named_parameters()]
        # 不能直接加正态分布噪声
        p_mean = torch.mean(params)
        p_var = torch.var(params)
        noise_data = normal.Normal(p_mean, p_var)
        noise = noise_data.sample(params.shape)
        noise_params = params + noise
        model = partial_reverse_tomodel(noise_params, net, train_layer).cuda()
        model.eval()
        # test_loss = 0
        correct = 0
        total = 0

        output_list = []

        with torch.no_grad():
            data = self.graph.cuda()
            # inputs, edge_index, targets, batch, = data.x, data.edge_index, data.y, data.batch
            outputs = model(data.x, data.edge_index)
            loss = F.cross_entropy(outputs[self.test_mask], data.y[self.test_mask])

            predicted = outputs.argmax(dim=1)
            total += data.y[self.test_mask].size(0)
            correct += int((predicted[self.test_mask] == data.y[self.test_mask]).sum())

        test_loss = loss.item()
        acc = 100. * correct / total
        del model
        return acc, test_loss


    def train_for_data(self, seed=1):
        net = self.build_model()
        optimizer = self.build_optimizer(net)
        criterion = nn.CrossEntropyLoss()
        scheduler = hydra.utils.instantiate(self.cfg.lr_scheduler, optimizer)
        epoch = self.cfg.epoch
        save_num = self.cfg.save_num_model
        all_epoch = epoch + save_num

        best_acc = 0
        train_layer = self.cfg.train_layer
        # test_layer = self.cfg.test_layer

        if train_layer == 'all':
            train_layer = [name for name, module in net.named_parameters()]
            # print(train_layer)

        shape_list = [pa.view(-1).shape[0] for name, pa in net.named_parameters() if name in train_layer]
        print(sum(shape_list))
        param_num = len(shape_list)
        # pos_enc = []
        # for idx, length in enumerate(shape_list):
        #     pos_enc.append(torch.ones(length)*idx)
        # pos_enc = torch.cat(pos_enc).long()
        # pos_enc_one_hot = nn.functional.one_hot(pos_enc, param_num)

        data_path = getattr(self.cfg, 'save_root', 'param_data')

        tmp_path = os.path.join(data_path, self.cfg.data.dataset)
        tmp_path = os.path.join(tmp_path, 'tmp_{}'.format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
        # tmp_path = os.path.join(data_path, 'tmp')
        final_path = os.path.join(data_path, self.cfg.data.dataset)
        final_path = os.path.join(final_path, f'all_seed_all/{seed}/{self.cfg.model.hidden_dim}_{self.cfg.optimizer.lr}_{self.cfg.optimizer.momentum}')

        os.makedirs(tmp_path, exist_ok=True)
        os.makedirs(final_path, exist_ok=True)


        save_model_accs = []
        parameters = []
        first_epoch_param = None
        net = net.cuda()
        # path = './param_data/Cora/generate.pt'
        # params = torch.load(path)
        # accs = []
        # for param in params:
        #     # net = self.build_model()
        #     # net = net.cuda()
        #     net = partial_reverse_tomodel(param.cuda(), net, train_layer)
        for i in range(0, all_epoch):
            self.train(net, criterion, optimizer, self.train_mask, i)
            acc = self.test(net, criterion, self.val_mask)
            best_acc = max(acc, best_acc)
            # if acc == best_acc:
            #     test_acc = self.test(net, criterion, self.test_mask)
            if i == 0:
                first_epoch_param = state_part(train_layer, net)
            if i == (epoch - 1):
                # print("saving the model")
                torch.save(net, os.path.join(tmp_path, "whole_model.pth"))
                fix_partial_model(train_layer, net)
                parameters = []
            if i >= epoch:
                parameters.append(state_part(train_layer, net))
                save_model_accs.append(acc)
                if len(parameters) == 5 or i == all_epoch - 1:
                    torch.save(parameters, os.path.join(tmp_path, "p_data_{}.pt".format(i)))
                    parameters = []

            scheduler.step()

        print(best_acc)
        # test_accs.append(self.test(net, criterion, self.val_mask))

        # print(f'Transfer ACC: {test_accs}')
        # print("training over")
        # transfer_acc = self.transfer_test(net, criterion, self.train_mask, train_layer)
        # print(f"Transfer acc: {transfer_acc}")

        # first_param = []
        #
        # # 顺序是固定的，现在改成按照config里面的顺序
        # for key in train_layer:
        #     first_param.append(first_epoch_param[key].reshape(-1))
        # # for key in buffer.keys():
        # #     if key in train_layer:
        # #         param.append(buffer[key].reshape(-1))
        # first_param = torch.cat(first_param, 0).cpu().detach()
        # torch.save(first_param, f'./data/{self.cfg.data.dataset}/first_param.pt')

        pdata = []
        param_list = get_param_dims(train_layer, net)
        # print(f'param_dims: {param_list}')
        for file in glob.glob(os.path.join(tmp_path, "p_data_*.pt")):
            buffers = torch.load(file)
            for buffer in buffers:
                param = []

                # 顺序是固定的，现在改成按照config里面的顺序
                for key in train_layer:
                    param.append(buffer[key].reshape(-1))
                # for key in buffer.keys():
                #     if key in train_layer:
                #         # ToDO 按照列展开来应该比较适合1D conv
                #         param.append(buffer[key].reshape(-1))
                param = torch.cat(param, 0)
                pdata.append(param)
        batch = torch.stack(pdata)
        mean = torch.mean(batch, dim=0)
        std = torch.std(batch, dim=0)
        # print(f'Param shape: {batch.shape}')

        # n_acc, n_test = [], []
        # for param in batch:
        #     acc, test_l = self.noise_test(param, net)
        #     n_acc.append(acc)
        #     n_test.append(test_l)
        # print(f'noised acc: {n_acc}')
        # print(f'noised mean acc: {sum(n_acc)/len(n_acc)}')
        # print(f'noised test loss: {n_test}')
        # print(f'performance decrease: {test_acc-sum(n_acc)/len(n_acc)}')

        t_acc = self.transfer_test(net, criterion, self.test_mask, train_layer)
        print(f'best transfer acc: {max(t_acc)}')
        print(f'transfer mean acc: {statistics.mean(t_acc)}')
        print(f'transfer test loss: {statistics.median(t_acc)}')

        # check the memory of p_data
        useage_gb = get_storage_usage(tmp_path)
        # print(f"path {tmp_path} storage usage: {useage_gb:.2f} GB")

        state_dic = {
            'pdata': batch.cpu().detach(),
            'mean': mean.cpu(),
            'std': std.cpu(),
            'model': torch.load(os.path.join(tmp_path, "whole_model.pth")),
            'train_layer': train_layer,
            'param_list': param_list,
            # 'initial': first_param,
            'performance': save_model_accs,
            # 'pos': pos_enc,
            'cfg': config_to_dict(self.cfg)
        }

        torch.save(state_dic, os.path.join(final_path, "data.pt"))
        json_state = {
            'cfg': config_to_dict(self.cfg),
            'performance': save_model_accs

        }
        json.dump(json_state, open(os.path.join(final_path, "config.json"), 'w'))

        # copy the code file(the file) in state_save_dir
        shutil.copy(os.path.abspath(__file__), os.path.join(final_path,
                                                            os.path.basename(__file__)))

        # delete the tmp_path
        shutil.rmtree(tmp_path)
        print(f'best model performance: {max(save_model_accs)}')
        print(f'mean model performance: {sum(save_model_accs)/len(save_model_accs)}')
        print(f'median model performance: {median(save_model_accs)}')
        print("data process over")
        return {'save_path': final_path}

    def train(self, net, criterion, optimizer, mask, epoch):
        # print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0

        data = self.graph.cuda()
        # inputs, edge_index, targets, batch, = data.x, data.edge_index, data.y, data.batch
        optimizer.zero_grad()
        outputs = net(data.x, data.edge_index)
        loss = criterion(outputs[mask], data.y[mask])
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predicted = outputs.argmax(dim=1)
        total += data.y[mask].size(0)
        correct += int((predicted[mask] == data.y[mask]).sum())

        # print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss, 100. * correct / total, correct, total))

    def test(self, net, criterion, mask):
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            data = self.graph.cuda()
            # inputs, edge_index, targets, batch, = data.x, data.edge_index, data.y, data.batch
            outputs = net(data.x, data.edge_index)
            loss = criterion(outputs[mask], data.y[mask])

            test_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            total += data.y[mask].size(0)
            correct += int((predicted[mask] == data.y[mask]).sum())
            # print('Loss: %.3f | Acc: %.3f%% (%d/%d)' % (test_loss, 100. * correct / total, correct, total))
            return 100. * correct / total

    def transfer_test(self, onet, criterion, mask, train_layer):
        net = copy.deepcopy(onet)
        path = './param_data/Cora/generate.pt'
        params = torch.load(path)
        accs = []
        for param in params:
            model = partial_reverse_tomodel(param.cuda(), net, train_layer)
            acc = self.test(model, criterion, mask)
            accs.append(acc)
        return accs

    def load_param(self):
        path = './param_data/Cora/generate.pt'
        params = torch.load(path)
        return params


