from image_convnet import *
from audio_convnet import *

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
		# self.vl2norm = 

		# Audio subnetwork
		self.apool4  = nn.MaxPool2d((16, 12), stride=(16, 12))
		self.afc1    = nn.Linear(512, 128)
		self.afc2    = nn.Linear(128, 128)

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

		# Audio
		aud = self.audnet(audio)
		aud = self.apool4(aud).squeeze(2).squeeze(2)
		aud = self.relu(self.afc1(aud))
		aud = self.afc2(aud)

		# Join them 
		mse = self.mse(img, aud)
		out = self.fc3(mse)
		out = self.softmax(out)

		return out, img, aud


if __name__ == "__main__":
	model = AVENet().cuda()
	image = Variable(torch.rand(2, 3, 224, 224)).cuda()
	audio = Variable(torch.rand(2, 1, 257, 200)).cuda()

	out, v, a = model(image, audio)
	print(image.shape, audio.shape)
	print(v.shape, a.shape, out.shape)

