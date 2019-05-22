import numpy as np
import torch
import torch.nn as nn


class AutoEncoder(nn.Module):

	def __init__(self):
		super(AutoEncoder, self).__init__()

		
		self.encoder = nn.Sequential(
			nn.Linear(28 * 28, 256),
			nn.ReLU(True),
			nn.Linear(256, 64),
			nn.ReLU(True),
			nn.Linear(64, 16),
			nn.ReLU(True))
			

		self.decoder = nn.Sequential(
			nn.Linear(16, 64),
			nn.ReLU(True),
			nn.Linear(64, 256),
			nn.ReLU(True),
			nn.Linear(256, 28 * 28),
			nn.Sigmoid())
			

	def forward(self, x):
		x = x.view(x.size(0), -1)
		x = self.encoder(x)
		x = self.decoder(x)
		x = x.view(x.size(0), 1, 28, 28)
		return x

