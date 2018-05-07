from image_convnet import *
from audio_convnet import *
from utils.dataloader import *

## Main NN starts here
class AVENet(nn.Module):

	def __init__(self):
		super(AVENet, self).__init__()

		self.relu   = F.relu
		self.imgnet = ImageConvNet()
		self.audnet = AudioConvNet()

		# Vision subnetwork
		self.vpool4  = nn.MaxPool2d(14, stride=14)
		self.vfc1    = nn.Linear(512, 128)
		self.vfc2    = nn.Linear(128, 128)
		self.vl2norm = nn.BatchNorm1d(128)

		# Audio subnetwork
		self.apool4  = nn.MaxPool2d((16, 12), stride=(16, 12))
		self.afc1    = nn.Linear(512, 128)
		self.afc2    = nn.Linear(128, 128)
		self.al2norm = nn.BatchNorm1d(128)

		# Combining layers
		self.mse     = F.mse_loss
		self.fc3     = nn.Linear(1, 2)
		self.softmax = F.softmax


	def forward(self, image, audio):
		# Image
		img = self.imgnet(image)
		img = self.vpool4(img).squeeze(2).squeeze(2)
		img = self.relu(self.vfc1(img))
		img = self.vfc2(img)
		img = self.vl2norm(img)

		# Audio
		aud = self.audnet(audio)
		aud = self.apool4(aud).squeeze(2).squeeze(2)
		aud = self.relu(self.afc1(aud))
		aud = self.afc2(aud)
		aud = self.al2norm(aud)

		# Join them 
		mse = self.mse(img, aud, reduce=False).mean(1).unsqueeze(1)
		out = self.fc3(mse)
		out = self.softmax(out, 1)

		return out, img, aud


	def get_image_embeddings(self, image):
		# Just get the image embeddings
		img = self.imgnet(image)
		img = self.vpool4(img).squeeze(2).squeeze(2)
		img = self.relu(self.vfc1(img))
		img = self.vfc2(img)
		img = self.vl2norm(img)
		return img
