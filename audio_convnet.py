import torch, os
from torch.optim import *
from torch.autograd import *
from torch import nn
from torch.nn import functional as F
from sklearn.feature_extraction.image import extract_patches_2d
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

class AudioConvNet(nn.Module):

	def __init__(self):
		super(AudioConvNet, self).__init__()
		self.pool = nn.MaxPool2d(2,stride=2,padding=1)
		
		self.cnn1 = nn.Conv2d(1, 64, 3, stride=2, padding=1)
		self.cnn2 = nn.Conv2d(64, 64, 3, padding=1)
		self.bat1 = nn.BatchNorm2d(64)

		self.cnn3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
		self.cnn4 = nn.Conv2d(128, 128, 3, padding=1)
		self.bat2 = nn.BatchNorm2d(128)

		self.cnn5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
		self.cnn6 = nn.Conv2d(256, 256, 3, padding=1)
		self.bat3 = nn.BatchNorm2d(256)

		self.cnn7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
		self.cnn8 = nn.Conv2d(512, 512, 3, padding=1)
		self.bat4 = nn.BatchNorm2d(512)

		self.fc1  = nn.Linear(100352, 1)

	def forward(self, inp):
		c = F.relu(self.bat1(self.cnn1(inp)))
		c = F.relu(self.bat1(self.cnn2(c)))
		c = self.pool(c)
		
		c = F.relu(self.bat2(self.cnn3(c)))
		c = F.relu(self.bat2(self.cnn4(c)))
		c = self.pool(c)
		
		c = F.relu(self.bat3(self.cnn5(c)))
		c = F.relu(self.bat3(self.cnn6(c)))
		c = self.pool(c)
		
		c = F.relu(self.bat4(self.cnn7(c)))
		c = F.relu(self.bat4(self.cnn8(c)))
		
		c = F.sigmoid(self.fc1(c.view(inp.shape[0], -1)))

		return c

	def loss(self, output):
		return (output.mean())**2

if __name__ == "__main__":
	model = AudioConvNet().cuda()
	print("Model loaded.")
	image = Variable(torch.rand(64, 1, 257, 200)).cuda()
	print("Image loaded.")

	optim = SGD(model.parameters(), lr=1e-4)
	for i in range(100):
		optim.zero_grad()
		c = model(image)
		loss = model.loss(c)
		loss.backward()
		optim.step()
		print(loss.data[0])
