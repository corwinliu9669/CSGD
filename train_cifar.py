#coding=utf-8
'''
This is the file for training and pruning networks for training model on mnist
'''

import torch
import os
import argparse
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.utils import *
import logging
import argparse
from data_loader import load_data
from model.resnet_cifar import resnet20
from csgd import CSGD

#########################################################################################################
parser = argparse.ArgumentParser()
####################################### optimizer settings #####################################
parser.add_argument('--lr_initial', default=3e-2, type=float)
parser.add_argument('--weight_decay', default=5e-4, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--nesterov", default=True, type=str2bool)
parser.add_argument("--keep_ratio", default=0.625, type=float)
parser.add_argument("--epi", default=3e-3, type=float)
####################################### training settings #####################################
parser.add_argument("--train_batch_size", default=64, type=int)
parser.add_argument("--val_batch_size", default=64, type=int)
parser.add_argument("--model_name", default='resnet20', type=str)

parser.add_argument("--learning_rate_interval", default=200, type=int)
parser.add_argument("--print_interval", default=100, type=int)
parser.add_argument("--checkpoint_interval", default=1, type=int)
parser.add_argument("--record_interval", default=100, type=int)
parser.add_argument("--weights_interval", default=1, type=int)
parser.add_argument("--train_from_scratch", default=True, type=str2bool) # whether train from scratch

parser.add_argument("--load_from_checkpoint", default=False, type=str2bool) # whether continue training
parser.add_argument("--previous_checkpoint_path", type=str)
parser.add_argument("--use_cuda", default=True, type=str2bool)
parser.add_argument("--parallel", default=True, type=str2bool)
parser.add_argument("--epoch_start", default=1, type=int)
parser.add_argument("--epoch_end", default=600, type=int)
parser.add_argument("--gpu_num", default='0, 1, 2, 3', type=str)


parser.add_argument("--test_pretrained_model", default=False, type=str2bool)
parser.add_argument("--pretrained_model_path", type=str)
parser.add_argument("--load_pretrained_model", default=False, type=str2bool)
parser.add_argument("--save_initialization", default=True, type=str2bool)
parser.add_argument("--init_name", type=str)
parser.add_argument("--training_model", default=True, type=str2bool)

####################################### pruning settings #####################################


####################################### path settings #####################################
parser.add_argument("--check_path", default=False, type=str2bool)
parser.add_argument("--save_check_point", default=False, type=str2bool)
parser.add_argument("--check_point_name", type=str)
parser.add_argument("--check_point_path", type=str)
parser.add_argument("--save_weights", default=False, type=str2bool)
parser.add_argument("--weights_name", type=str)
parser.add_argument("--weights_name_path", type=str)



####################################### record settings #####################################
#To be complete

#########################################################################################################
args = parser.parse_args()


###### Data Loader ############
train_loader = load_data(dataset="Cifar10", train=True, batch_size=args.train_batch_size, shuffle=True)
test_loader = load_data(dataset="Cifar10", train=False, batch_size=args.val_batch_size, shuffle=False)
all_num = args.epoch_end * len(train_loader)

print('num of all step:', all_num)
print('num of step per epoch:', len(train_loader))

###### check whether path exists #####
if args.check_path :
	if args.check_point_path != None:
		check_path(args.check_point_path)
	if args.weights_name_path != None:
		check_path(args.weights_name_path)
	if args.retrain_weight_path != None:
		check_path(args.retrain_weight_path)
	checkpoint_path = args.check_point_path + args.check_point_name + '.pth'


######## pick model #########
device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")
torch.cuda.empty_cache()
if args.use_cuda:
	os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num
if args.model_name == 'resnet20':
	model = resnet20().to(device)
else:
	raise NameError('Wrong Model Name')

if args.parallel and  args.use_cuda:
	model = nn.DataParallel(model)

######  check whether to load pretrained model #####

if args.load_pretrained_model:
	load_weights(model, args.pretrained_model_path)
	if args.test_pretrained_model:
		print('Test Pretrained Model')
		evaluate(model, test_loader, device)

###### Get Name List ##########
name_list = []
slbi_list = []
for name, p in model.named_parameters():
    name_list.append(name)
    print(name)
    print(p.size())

############ optimizer ##################################
optimizer = CSGD(model.parameters(), lr=args.lr_initial, momentum=args.momentum, nesterov=args.nesterov, \
                weight_decay=args.weight_decay, keep_ratio=args.keep_ratio, epi=args.epi)

optimizer.assign_name(name_list)
cluster_matrix,  H_index, cardinality = optimizer.generate_cluster_matrix(return_matrix=True)
#print(cardinality)
#print(cluster_matrix['module.conv1.weight'])



######  check whether to load checkpoint#####
if args.load_from_checkpoint:
	load_checkpoints(model,optimizer, path)
########## training model #######
if args.training_model:
	max_val_acc = 0
	print('Training Model : ' + args.model_name)
	for ep in range(args.epoch_start, args.epoch_end + 1):
#		weight_path = args.weights_name_path + args.weights_name + '_' + str(ep) + '.pth'
		model.train()
		descent_lr(args.lr_initial, ep, optimizer, args.learning_rate_interval)
		loss_val = 0
		correct = num = 0
		for iter, pack in enumerate(train_loader):
			data, target = pack[0].to(device), pack[1].to(device)
			logits = model(data)
			loss = F.nll_loss(logits, target)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			_, pred = logits.max(1)
			loss_val += loss.item()
			correct += pred.eq(target).sum().item()
			num += data.shape[0]
			if (iter + 1) % args.print_interval == 0:
				print('*******************************')
				print('epoch : ', ep)
				print('iteration : ', iter + 1)
				print('loss : ', loss_val/100)
				print('Correct : ', correct)
				print('Num : ', num)
				print('Train ACC : ', correct/num)
				correct = num = 0
				loss_val = 0
		print('Test Model')
		tmp_val_acc = evaluate(model, test_loader, device)
		if max_val_acc < tmp_val_acc:
			max_val_acc = tmp_val_acc
		if (ep) % args.weights_interval == 0 and args.save_weights:
			print('Save Weights')
			save_model(model, weight_path)
		if (ep) % args.checkpoint_interval == 0 and args.save_check_point:
			save_checkpoints(model, optimizer, checkpoint_path)
	print('Training Done')
	print('Best Val Acc : ', max_val_acc)

########### Extract Model #############

