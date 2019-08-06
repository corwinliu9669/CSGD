'''
This file includes functions used for evaluation and other usages
'''
import os
import torch
import logging
from collections import OrderedDict
import numpy as np
############################ Miscellance #####################################
def check_path(path):
	if path == 'None':
		pass
	else:
		if os.path.exists(path):
			pass
		else:
			os.mkdir(path)

def str2bool(value):
    return value.lower() == 'true'


def mgpu_dict_to_sgpu_dict(weights_dict):
	w_d = OrderedDict()
	for k, v in weights_dict.items():
    		new_k = k.replace('module.', '')
    		print(new_k)
    		w_d[new_k] = v
	return w_d


def generate_mask(weight_dict, ratio):
	print('Generate Mask')
	mask = OrderedDict() 
	for  i, name in enumerate(weight_dict):
		p = weight_dict[name]
		if len(p.size()) == 1 and 'bias' not in name:
			length =  p.size()[0]
			thre_index = int(ratio * length)
			p_numpy = torch.abs(p.data).cpu().numpy()
			np.sort(p_numpy)
			thre = p_numpy[thre_index]
			mask[name] = torch.gt(torch.abs(p.data), thre).float()
			print('Sparsity of ' + name)
			print(mask[name].sum().item() / p.size()[0])
		elif len(p.size()) == 2:
			size =  p.size()
			length =  p.size()[0] * p.size()[1]
			thre_index = int(ratio * length)
			print(thre_index)
			p_numpy = torch.flatten(torch.abs(p.data)).cpu().numpy()
			p_numpy = np.sort(p_numpy)
			thre = p_numpy[thre_index]
			print(thre)
			mask[name] = torch.gt(torch.abs(p.data), thre).float()
			mask[name] = mask[name].view(size[0], size[1])
			print('Sparsity of ' + name)
			print(mask[name].sum().item() / (p.size()[0] *p.size()[1] ))
		elif len(p.size()) == 4:
			size =  p.size()
			length =  p.size()[0] * p.size()[1] * p.size()[2] * p.size()[3]
			thre_index = int(ratio * length)
			p_numpy = torch.flatten(torch.abs(p.data)).cpu().numpy()
			np.sort(p_numpy)
			thre = p_numpy[thre_index]
			mask[name] = torch.gt(torch.abs(p.data), thre).float()
			mask[name] = mask[name].view(size[0], size[1], size[2], size[3])
			print('Sparsity of ' + name)
			print(mask[name].sum().item() / (p.size()[0] *p.size()[1] * p.size()[2] * p.size()[3]))
	return mask

############################ Descent lr ########################################
def descent_lr(lr, epoch, optimizer, interval):
        lr = lr * (0.1 ** (epoch //interval))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('***********************************')
        print('learning rate:', lr)
        print('***********************************')


def assign_lr(lr_new, optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_new

############################ Eavuation ########################################

def evaluate(model, data_loader, device):
	model.eval()
	correct = num = 0
	for iter, pack in enumerate(data_loader):
		data, target = pack[0].to(device), pack[1].to(device)
		logits = model(data)
		_, pred = logits.max(1)
		correct += pred.eq(target).sum().item()
		num += data.shape[0]
	print('Correct : ', correct)
	print('Num : ', num)
	print('Test ACC : ', correct / num)
	torch.cuda.empty_cache()
	model.train()
	return correct/num

############################ Save #############################################

def save_masks(mask_dict, path):
	print('Save Mask')
	torch.save(mask_dict, path)

def save_initailization(weight_dict, path):
	print('Save Initialization')
	torch.save(weight_dict, path)

def save_model(model, path):
	print('Save Model Weights')
	torch.save(model.state_dict(), path)

def save_checkpoints(model,optimizer, path):
	print('Save Check Point')
	save_dict = {'model': model.state_dict(), 'optimizer' : optimizer.state_dict()}
	torch.save(model.state_dict(), path)

############################ Load #############################################

def load_masks(optimizer, path):
	print('Load Mask')
	mask_dic = torch.load(path)
	optimizer.load_mask(mask_dict)
	print('Load Done')


def load_initailization(path):
	print('Load Initialization')
	model.load_state_dict(torch.load(path))
	print('Load Done')


def load_weights(model, path):
	print('Load Model Weights')
	model.load_state_dict(torch.load(path))
	print('Load Done')


def load_checkpoints(model,optimizer, path):
	print('Load Check Point')
	check_point = torch.load(path)
	model.load_state_dict(check_point['model'])
	optimizer.load_state_dict(check_point['optimizer'])
	print('Load Done')


############################ Record into log or txt ############################################
def record_grad():
	pass
	
def record_norm():
	pass

def record_spars():
	##### only used for slbi
	pass

def write_into_log():
	pass
