from __future__ import division, print_function
import torch
import torchvision
from torchvision import datasets, transforms
import numpy as np
'''
This file is used to load data
Data used in this project includes MNIST, Cifar10 and ImageNet
'''


def load_data(dataset='Cifar10', train=False, download=True, transform=None, batch_size=1, shuffle=True):
	if dataset == 'MNIST':
		data_loader = torch.utils.data.DataLoader(datasets.MNIST('/home/wwx/data/MNIST', train=train, download=download, transform=transforms.Compose([transforms.ToTensor(),])), batch_size=batch_size, shuffle=shuffle)
	elif dataset == 'Cifar10':
		if train:
			trans = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
		else:
			trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
		data_loader = torch.utils.data.DataLoader(datasets.CIFAR10('/home/wwx/data/Cifar10', train=train, download=download, transform=trans), batch_size=batch_size, shuffle=shuffle)

	elif dataset == 'ImageNet':
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
		if train:
			dataset = torchvision.datasets.ImageFolder('/home/wwx/data/ILSVRC2012/train', transform=transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
]))
			data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=32)
		else:
			dataset = torchvision.datasets.ImageFolder('/home/lc/data/ILSVRC2012/val/', transform=transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
]))
			data_loader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=shuffle, num_workers=32)
	else:
		print('No such dataset')
	return data_loader

if __name__ == '__main__':
	data_loader = load_data()
	for data, target in data_loader:
		print(data)
		print(target) 
		stop

