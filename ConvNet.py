import numpy as np
import torch
import torch.nn as nn


class ConvNet(nn.Module):
	def __init__(self, gpu=False):
		super(ConvNet, self).__init__()
		self.gpu = gpu
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4)
		self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
		self.conv2 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3)
		self.fc1 = nn.Linear(8 * 22 * 22, 1000)
		self.dropout = nn.Dropout()
		self.fc2 = nn.Linear(1000, 10)
		
	def forward(self, x):
		x = self.conv1(x)
		x = self.pool(x)
		x = self.conv2(x)
		x = x.view(-1, 8 * 22 * 22)
		x = self.fc1(x)
		x = self.dropout(x)
		x = self.fc2(x)
		return x
