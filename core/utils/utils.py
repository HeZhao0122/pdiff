
import torch
import os
import torch.nn.functional as F
import random
import numpy as np



def state_part(train_list, net):
    part_param = {}
    for name, weights in net.named_parameters():
        if name in train_list:
            part_param[name] = weights.detach().cpu()
    return part_param

def get_param_dims(train_list, net):
    param_list = []
    for name, weights in net.named_parameters():
        if name in train_list:
            size = weights.view(-1).detach().cpu().shape
            param_list.append(int(size[0]))
    return param_list


def fix_partial_model(train_list, net):
    # print(train_list)
    for name, weights in net.named_parameters():
        if name not in train_list:
            weights.requires_grad = False

def partial_reverse_tomodel(flattened, model, train_layer, test_layer=None):
    layer_idx = 0
    # for name, pa in model.named_parameters():
    #     if name in train_layer:
    #         pa_shape = pa.shape
    #         pa_length = pa.view(-1).shape[0]
    #         pa.data = flattened[layer_idx:layer_idx + pa_length].reshape(pa_shape)
    #         pa.data.to(flattened.device)
    #         layer_idx += pa_length
    pa_list = {name: pa for name, pa in model.named_parameters() if name in train_layer}
    for name in train_layer:
        pa = pa_list[name]
        pa_shape = pa.shape
        pa_length = pa.view(-1).shape[0]
        if test_layer is not None and name in test_layer:
            pa.data = flattened[layer_idx:layer_idx + pa_length].reshape(pa_shape)
            pa.data.to(flattened.device)
        if test_layer is None:
            pa.data = flattened[layer_idx:layer_idx + pa_length].reshape(pa_shape)
            pa.data.to(flattened.device)
        layer_idx += pa_length
    del pa_list
    return model

import pdb
def test_generated_partial(net, param, train_layer, dataloader, fea_path=None):
    target_num = 0
    for name, module in net.named_parameters():
        if name in train_layer:
            target_num += torch.numel(module)
    params_num = torch.squeeze(param).shape[0]  # + 30720
    assert (target_num == params_num)
    param = torch.squeeze(param)
    net = partial_reverse_tomodel(param, net, train_layer).to(param.device)
    acc, loss, output_list = test(net, dataloader, fea_path=fea_path)
    del net
    return acc, output_list


def test(model, test_loader,fea_path=None):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    output_list = []

    with torch.no_grad():
        if fea_path is not None:
            expert_files = os.listdir(fea_path)
            for m in expert_files:
                models = fea_path + m
                fea_targets = torch.load(models)
                targets = fea_targets[1].to('cuda')
                inputs = fea_targets[0].to('cuda')
                outputs = model.forward_norm(inputs)
                _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc = 100.*correct/total
            return acc, _
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            target=target.to(torch.int64)
            test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss

            total += data.shape[0]
            pred = torch.max(output, 1)[1]
            output_list += pred.cpu().numpy().tolist()
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= total
    acc = 100. * correct / total
    del model
    return acc, test_loss, output_list


def top_acc_params(self, accs, params, topk):
    sorted_list = sorted(accs, reverse=True)[:topk]
    max_indices = [accs.index(element) for element in sorted_list]
    best_params = params[max_indices, :]
    del params
    return best_params

def _warmup_beta(start, end, n_timestep, warmup_frac):

    betas               = end * torch.ones(n_timestep, dtype=torch.float64)
    warmup_time         = int(n_timestep * warmup_frac)
    betas[:warmup_time] = torch.linspace(start, end, warmup_time, dtype=torch.float64)

    return betas

def make_beta_schedule(schedule, start, end, n_timestep):
    if schedule == "quad":
        betas = torch.linspace(start ** 0.5, end ** 0.5, n_timestep, dtype=torch.float64) ** 2
    elif schedule == 'linear':
        betas = torch.linspace(start, end, n_timestep, dtype=torch.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(start, end, n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(start, end, n_timestep, 0.5)
    elif schedule == 'const':
        betas = end * torch.ones(n_timestep, dtype=torch.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / (torch.linspace(n_timestep, 1, n_timestep, dtype=torch.float64))
    else:
        raise NotImplementedError(schedule)

    return betas

def accumulate(model1, model2, decay=0.9999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


import os
import shutil

def get_storage_usage(path):
    list1 = []
    fileList = os.listdir(path)
    for filename in fileList:
        pathTmp = os.path.join(path,filename)  
        if os.path.isdir(pathTmp):   
            get_storage_usage(pathTmp)
        elif os.path.isfile(pathTmp):  
            filesize = os.path.getsize(pathTmp)  
            list1.append(filesize) 
    usage_gb = sum(list1)/1024/1024/1024
    return usage_gb

def reverse_tomodel(flattened, model):
    example_parameters = [p for p in model.parameters()]
    length = 0
    reversed_params = []

    for p in example_parameters:
        flattened_params = flattened[length: length+p.numel()]
        reversed_params.append(flattened_params.reshape(p.shape))
        length += p.numel()

    layer_idx = 0
    for p in model.parameters():
        p.data = reversed_params[layer_idx]
        p.data.to(flattened.device)
        p.data.requires_grad_(True)
        layer_idx += 1
    return model

def test_ensem(self, best_params, net):
    stacked = torch.stack(list(torch.squeeze(best_params)))
    mean = torch.mean(stacked, dim = 0)
    ensemble_model = reverse_tomodel(mean, net)
    acc,_= test(ensemble_model.cuda(), self.testloader)
    del best_params
    return acc

def test_ensem_partial(self, best_params,dataloader,fea_path=None):
    stacked = torch.stack(list(torch.squeeze(best_params)))
    mean = torch.mean(stacked, dim = 0)
    acc = test_generated_partial(self, mean,dataloader,fea_path=fea_path)
    del best_params
    return acc

class TemporaryRandomSeed:
    def __init__(self, seed):
        self.seed = seed
        self.state = None
        self.numpy_state = None
        self.torch_state = None
        self.cuda_state = None

    def __enter__(self):
        # 保存当前的随机状态
        self.state = random.getstate()
        self.numpy_state = np.random.get_state()
        self.torch_state = torch.get_rng_state()
        if torch.cuda.is_available():
            self.cuda_state = torch.cuda.get_rng_state_all()

        # 设置新的随机种子
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def __exit__(self, exc_type, exc_value, traceback):
        # 恢复之前的随机状态
        random.setstate(self.state)
        np.random.set_state(self.numpy_state)
        torch.set_rng_state(self.torch_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(self.cuda_state)