from image_convnet import *
from audio_convnet import *
from dataloader import *
from torch.optim import *

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

# Demo to check if things are working
def demo():
	model = AVENet()
	image = Variable(torch.rand(2, 3, 224, 224))
	audio = Variable(torch.rand(2, 1, 257, 200))

	out, v, a = model(image, audio)
	print(image.shape, audio.shape)
	print(v.shape, a.shape, out.shape)

# Main function here
def main(use_cuda=True, EPOCHS=100, save_checkpoint=20, batch_size=64, model_name="avenet.pt"):
	
	model = getAVENet(use_cuda)
	dataset = GetAudioVideoDataset()
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
	crossEntropy = nn.CrossEntropyLoss()

	optim = Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

	for epoch in range(EPOCHS):
		# Run algo
		for (img, aud, out) in dataloader:
			optim.zero_grad()

			# Filter the bad ones first
			out = out.squeeze(1)
			idx = out != 2
			if idx.sum() == 0:
				continue

			print(img.shape, aud.shape, out.shape)
			img = Variable(img[idx, :, :, :])
			aud = Variable(aud[idx, :, :, :])
			out = Variable(out[idx])
			print(img.shape, aud.shape, out.shape)

			if use_cuda:
				img = img.cuda()
				aud = aud.cuda()
				out = out.cuda()

			o, _, _ = model(img, aud)
			# print(o)
			# print(o.shape, out.shape)
			loss = crossEntropy(o, out)
			loss.backward()
			optim.step()
			print("Loss: %f"%(loss.data[0]))

		if epoch>0 and epoch%save_checkpoint==0:
			torch.save(model.state_dict(), model_name)
			print("Checkpoint saved.")


def getAVENet(use_cuda):
	model = AVENet()
	model.fc3.weight.data[0] = -0.1
	model.fc3.weight.data[1] =  0.1
	model.fc3.bias.data[0] =  1.0
	model.fc3.bias.data[1] = -1.0
	if use_cuda:
		model = model.cuda()

	return model


if __name__ == "__main__":
	cuda = True
	main(use_cuda=cuda, batch_size=64)
	# demo()


