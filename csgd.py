import torch
import numpy as np
from collections import OrderedDict
from torch.optim.optimizer import Optimizer, required

class CSGD(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, keep_ratio=0, epi=0):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov, keep_ratio=keep_ratio, epi=epi)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(CSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def assign_name(self, name_list):
        for group in self.param_groups:
            for iter, p in enumerate(group['params']):
                param_state = self.state[p]
                param_state['name'] = name_list[iter]

    def load_mask(self, mask):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if state['name'] in mask.keys():
                    state['mask'] = mask[state['name']]

    def apply_mask(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if 'mask' in state:
                    p.data = p.data * state['mask']

    def generate_cluster_matrix(self, return_matrix=False):
        print('generate_cluster_matrix')
        cluster_matrix = OrderedDict()
        H_index = OrderedDict()
        cardinality = OrderedDict()
        for group in self.param_groups:
            keep_ratio = group['keep_ratio']
            for p in group['params']:
                param_state = self.state[p]
                weights_shape = p.data.size()
                if 'fc' in param_state['name'] or 'linear' in param_state['name']:
                    pass
                elif 'conv' in param_state['name'] or 'bn' in param_state['name']:
                    channel =  p.data.size()[0]
                    n_cluster = int(channel * keep_ratio)
                    param_state['n_cluster'] = n_cluster
                    param_state['cluster_matrix'] = torch.zeros((channel, channel)).cuda()
                    param_state['H_index'] = torch.zeros((channel,))
                    param_state['cardinality'] =  torch.zeros((n_cluster, ))
                    for i in range(channel):
                        #if i < n_cluster:
                        param_state['H_index'][i] = i % n_cluster
                        param_state['cardinality'][int(param_state['H_index'][i].item())] += 1
                        #else:
                        #    random_index = np.random.randint(0, n_cluster)
                        #    if param_state['cardinality'][random_index] >= (int(1/keep_ratio) + 1):
                         #       random_index = np.random.randint(0, n_cluster)
                         #   param_state['H_index'][i] = random_index
                         #   param_state['cardinality'][int(param_state['H_index'][i].item())] += 1
                    for i in range(channel):
                        for j in range(channel):
                            if param_state['H_index'][i] == param_state['H_index'][j]:
                                param_state['cluster_matrix'][i, j] = 1 / float(param_state['cardinality'][int(param_state['H_index'][i].item())])
                            else:
                                param_state['cluster_matrix'][i, j] = 0

                    cluster_matrix[param_state['name']] = param_state['cluster_matrix']
                    H_index[param_state['name']] = param_state['H_index']
                    cardinality[param_state['name']] = param_state['cardinality']
                else:
                    pass

        if return_matrix:
            return cluster_matrix,  H_index, cardinality

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            epi = group['epi']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                if 'cluster_matrix' in param_state:
                    if len(d_p.size()) == 4:
                        shape = d_p.size()
                        d_p_r = d_p.view(shape[0], -1)
                        d_p_r = torch.matmul(param_state['cluster_matrix'], d_p_r)
                        d_p = d_p_r.view(shape)
                    else:
                        d_p = torch.matmul(param_state['cluster_matrix'], d_p)
                if weight_decay != 0:
                    if 'cluster_matrix' in param_state:
                        if len(d_p.size()) == 4:
                            d_p.add_(weight_decay, p.data)
                            shape = d_p.size()
                            p_r = p.data.view(shape[0], -1)
                            p_r = torch.matmul(param_state['cluster_matrix'], p_r)
                            p_r = p_r.view(shape)
                            d_p.add_(epi, p.data - p_r)
                        else:
                            d_p.add_(weight_decay, p.data)
                            d_p.add_(epi, p.data - torch.matmul(param_state['cluster_matrix'], p.data))
                    else:
                        d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                p.data.add_(-group['lr'], d_p)

        return loss


    def step_with_mask(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                if 'mask' in param_state:
                    d_p = d_p * param_state['mask']
                p.data.add_(-group['lr'], d_p)

        return loss
