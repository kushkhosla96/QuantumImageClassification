import numpy as np
import torch
import torch.nn as nn


class AutoEncoder(nn.Module):

	def __init__(self, layer_sizes):
		super(AutoEncoder, self).__init__()

		self.layer_sizes = [28 * 28] + layer_sizes
		self.encoding_size = self.layer_sizes[-1]

		encoder_modules = []
		for i in range(len(self.layer_sizes) - 1):
			encoder_modules.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i+1]))
			encoder_modules.append(nn.ReLU(True))			
		self.encoder = nn.Sequential(*encoder_modules)	

		decoder_modules = []
		for i in range(len(self.layer_sizes) - 1, 0, -1):
			decoder_modules.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i-1]))
			decoder_modules.append(nn.ReLU(True))
		decoder_modules.pop()
		decoder_modules.append(nn.Sigmoid())

		self.decoder = nn.Sequential(*decoder_modules)	

	def forward(self, x):
		x = x.view(x.size(0), -1)
		x = self.encoder(x)
		x = self.decoder(x)
		x = x.view(x.size(0), 1, 28, 28)
		return x

	def get_encoding_size(self):
		return self.encoding_size

