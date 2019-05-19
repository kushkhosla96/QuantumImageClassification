import os
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms

class MNISTData:
	def __init__(self, mean=.5, std=1.0, batch_size=128, root="./data"):
		self.mean = mean
		self.std = std
		self.root = root
		self.transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize((self.mean,), (self.std,))])
		
		if not os.path.exists(root):
			os.mkdir(root)
		
		self.train_set = dset.MNIST(root=root, train=True, transform=self.transformation, download=True)
		self.test_set = dset.MNIST(root=root, train=False, transform=self.transformation, download=True)

		self.batch_size = batch_size

		self.train_loader = torch.utils.data.DataLoader(dataset=self.train_set, batch_size=self.batch_size, shuffle=True)
		self.test_loader = torch.utils.data.DataLoader(dataset=self.test_set, batch_size=self.batch_size, shuffle=False)

	def get_train_loader(self):
		return self.train_loader

	def get_test_loader(self):
		return self.test_loader
