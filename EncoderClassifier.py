import torch
import torch.nn as nn


class EncoderClassifier(nn.Module):
	
	def __init__(self, encoder, encoding_size):
		super(EncoderClassifier, self).__init__()

		self.encoder = encoder
		self.encoding_size = encoding_size

		for child in self.encoder.children():
			for param in child.parameters():
				param.requires_grad = False

		self.classifier = nn.Sequential(
			nn.Linear(self.encoding_size, 1000),
			nn.Dropout(),
			nn.Linear(1000, 10))


	# input is of size (b, 1, 28, 28)
	def forward(self, x):
		x = x.view(x.size(0), -1)
		x = self.encoder(x)
		return self.classifier(x)

