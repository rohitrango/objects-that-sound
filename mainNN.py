from image_convnet import *
from audio_convnet import *
from dataloader import *
from torch.optim import *
from torchvision.transforms import *
import warnings
import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
## Handle warnings here
# CITE: https://stackoverflow.com/questions/858916/how-to-redirect-python-warnings-to-a-custom-stream
warnings_file = open("warning_logs.txt", "w+")
def customwarn(message, category, filename, lineno, file=None, line=None):
    warnings_file.write(warnings.formatwarning(message, category, filename, lineno))

warnings.showwarning = customwarn

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



# Demo to check if things are working
def demo():
	model = AVENet()
	image = Variable(torch.rand(2, 3, 224, 224))
	audio = Variable(torch.rand(2, 1, 257, 200))

	out, v, a = model(image, audio)
	print(image.shape, audio.shape)
	print(v.shape, a.shape, out.shape)

# Main function here
def main(use_cuda=True, lr=0.25e-4, EPOCHS=100, save_checkpoint=500, batch_size=64, model_name="avenet.pt"):
	
	lossfile = open("losses.txt", "a+")
	print("Using batch size: %d"%batch_size)
	print("Using lr: %f"%lr)

	model = getAVENet(use_cuda)

	# Load from before
	if os.path.exists(model_name):
		model.load_state_dict(torch.load(model_name))
		print("Loading from previous checkpoint.")


	dataset = GetAudioVideoDataset()
	valdataset = GetAudioVideoDataset(video_path="Video_val/", audio_path="Audio_val/", validation=True)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
	valdataloader = DataLoader(valdataset, batch_size=batch_size, shuffle=True)

	crossEntropy = nn.CrossEntropyLoss()
	print("Loaded dataloader and loss function.")

	# optim = Adam(model.parameters(), lr=lr, weight_decay=1e-7)
	optim = SGD(model.parameters(), lr=lr)
	print("Optimizer loaded.")
	model.train()

	try:
		for epoch in range(EPOCHS):
			# Run algo
			for subepoch, (img, aud, out) in enumerate(dataloader):
				optim.zero_grad()

				# Filter the bad ones first
				out = out.squeeze(1)
				idx = (out != 2).numpy().astype(bool)
				if idx.sum() == 0:
					continue

				# Find the new variables
				img = torch.Tensor(img.numpy()[idx, :, :, :])
				aud = torch.Tensor(aud.numpy()[idx, :, :, :])
				out = torch.LongTensor(out.numpy()[idx])

				# Print shapes
				img = Variable(img)
				aud = Variable(aud)
				out = Variable(out)

				# print(img.shape, aud.shape, out.shape)

				M = img.shape[0]
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

				# Calculate accuracy
				_, ind = o.max(1)
				accuracy = (ind.data == out.data).sum()*1.0/M

				# Periodically print subepoch values
				if subepoch%10 == 0:
					model.eval()
					for (img, aud, out) in valdataloader:
						break
					# Filter the bad ones first
					out = out.squeeze(1)
					idx = (out != 2).numpy().astype(bool)
					if idx.sum() == 0:
						continue
					# Find the new variables
					img = torch.Tensor(img.numpy()[idx, :, :, :])
					aud = torch.Tensor(aud.numpy()[idx, :, :, :])
					out = torch.LongTensor(out.numpy()[idx])

					# Print shapes
					img = Variable(img, volatile=True)
					aud = Variable(aud, volatile=True)
					out = Variable(out, volatile=True)

					# print(img.shape, aud.shape, out.shape)

					M = img.shape[0]
					if use_cuda:
						img = img.cuda()
						aud = aud.cuda()
						out = out.cuda()

					o, _, _ = model(img, aud)
					valloss = crossEntropy(o, out)
					# Calculate valaccuracy
					_, ind = o.max(1)
					valaccuracy = (ind.data == out.data).sum()*1.0/M

					print("Epoch: %d, Subepoch: %d, Loss: %f, Valloss: %f, batch_size: %d, acc: %f, valacc: %f"%(epoch, subepoch, loss.data[0], valloss.data[0], M, accuracy, valaccuracy))
					lossfile.write("Epoch: %d, Subepoch: %d, Loss: %f, Valloss: %f, batch_size: %d, acc: %f, valacc: %f\n"%(epoch, subepoch, loss.data[0], valloss.data[0], M, accuracy, valaccuracy))
					model.train()
				
				# Save model
				if subepoch%save_checkpoint == 0 and subepoch > 0:
					torch.save(model.state_dict(), model_name)
					print("Checkpoint saved.")

	except Exception as e:
		print(e)
		torch.save(model.state_dict(), "backup"+model_name)
		print("Checkpoint saved and backup.")

	lossfile.close()



def getAVENet(use_cuda):
	model = AVENet()
	# model.fc3.weight.data[0] = -0.1
	# model.fc3.weight.data[1] =  0.1
	# model.fc3.bias.data[0] =   1.0
	# model.fc3.bias.data[1] = - 1.0
	model.fc3.weight.data[0] = -0.7090
	model.fc3.weight.data[1] =  0.7090
	model.fc3.bias.data[0] =   1.2186
	model.fc3.bias.data[1] = - 1.2186
	if use_cuda:
		model = model.cuda()

	return model


def checkValidation(use_cuda=True, batch_size=64, model_name="avenet.pt", validation=True):
	
	print("Using batch size: %d"%batch_size)
	model = getAVENet(use_cuda)

	# Load from before
	if os.path.exists(model_name):
		model.load_state_dict(torch.load(model_name))
		print("Loading from previous checkpoint.")


	print("Model name: {0}".format(model_name))
	if validation == True or validation == "validation":
		print("Using validation")
		dataset = GetAudioVideoDataset(video_path="Video_val/", audio_path="Audio_val/", validation=True)
	elif validation == "test":
		print("Using test.")
		dataset = GetAudioVideoDataset(video_path="Video_test/", audio_path="Audio_test/", validation="test")
	else:
		print("Using training")
		dataset = GetAudioVideoDataset()


	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
	crossEntropy = nn.CrossEntropyLoss()
	print("Loaded dataloader and loss function.")
	model.eval()
	correct = []
	count = []
	losses= []
	print(len(dataset))

	try:
		for subepoch, (img, aud, out) in enumerate(dataloader):
			# Filter the bad ones first
			out = out.squeeze(1)
			idx = (out != 2).numpy().astype(bool)
			if idx.sum() == 0:
				continue

			# Find the new variables
			img = torch.Tensor(img.numpy()[idx, :, :, :])
			aud = torch.Tensor(aud.numpy()[idx, :, :, :])
			out = torch.LongTensor(out.numpy()[idx])

			# Print shapes
			img = Variable(img, volatile=True)
			aud = Variable(aud, volatile=True)
			out = Variable(out, volatile=True)

			# print(img.shape, aud.shape, out.shape)

			M = img.shape[0]
			if use_cuda:
				img = img.cuda()
				aud = aud.cuda()
				out = out.cuda()

			o, _, _ = model(img, aud)

			loss = crossEntropy(o, out).data[0]
			# Calculate accuracy
			_, ind = o.max(1)
			acc = (ind.data == out.data).sum()*1.0
			correct.append(acc)
			losses.append(loss)
			count.append(M)
			print(subepoch, acc, M, acc/M, loss)
			if subepoch == 100:
				break

	except Exception as e:
		print(e)

	M = sum(count)
	corr = sum(correct)
	print("Total frames: {0}:".format(M))
	print("Total correct: {0}".format(corr))
	print("Total accuracy: {0}".format(corr/M))
	print("Total loss: {0}".format(np.mean(losses)))
	

def getVideoEmbeddings(model_name="avenet.pt"):
	# Get video embeddings on the test set
	dataset = GetAudioVideoDataset(video_path="Video_test/", audio_path="Audio_test/", validation="test", return_tags=True)
	dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
	print("Loading data.")
	for img, aud, res, vidTags, audTags in dataloader:
		break

	model = getAVENet(True)

	# Load from before
	if os.path.exists(model_name):
		model.load_state_dict(torch.load(model_name))
		print("Loading from previous checkpoint.")

	model.eval()
	embed = model.get_image_embeddings(Variable(img, volatile=True).cuda())
	print(embed.data)

	embed = embed.data.cpu().numpy()
	vidTags = vidTags.numpy()

	data = TSNE(n_iter=5000).fit_transform(embed)
	X = data[:,0]
	Y = data[:,1]
	print(X.shape, Y.shape, vidTags.shape)
	for i in range(X.shape[0]):
		plt.scatter(X[i], Y[i])

	for i in range(X.shape[0]):
		plt.annotate(vidTags[i,0], (X[i], Y[i]))

	print(vidTags)
	plt.show()



def generateEmbeddingsForVideoAudio(model_name="avenet.pt", use_cuda=True):
	# Get video embeddings on the test set
	dataset = GetAudioVideoDataset(video_path="Video_test/", audio_path="Audio_test/", validation="test", return_tags=True)
	dataloader = DataLoader(dataset, batch_size=128, shuffle=True)
	print("Loading data.")
	for img, aud, res, vidTags, audTags in dataloader:
		break

	model = getAVENet(True)

	# Load from before
	if os.path.exists(model_name):
		model.load_state_dict(torch.load(model_name))
		print("Loading from previous checkpoint.")

	imgList, audList, resList, vidTagList, audTagList = [], [], [], [], []

	model.eval()
	for i, (img, aud, res, vidTag, audTag) in enumerate(dataloader):
		# Filter the bad ones first
		res = res.squeeze(1)
		idx = (res != 2).numpy().astype(bool)
		if idx.sum() == 0:
			continue

		# Find the new variables
		img = torch.Tensor(img.numpy()[idx, :, :, :])
		aud = torch.Tensor(aud.numpy()[idx, :, :, :])
		res = torch.LongTensor(res.numpy()[idx])
		vidTag = vidTag.numpy()[idx]
		audTag = audTag.numpy()[idx]

		# Print shapes
		img = Variable(img, volatile=True)
		aud = Variable(aud, volatile=True)
		res = Variable(res, volatile=True)

		# print(img.shape, aud.shape, res.shape)

		M = img.shape[0]
		if use_cuda:
			img = img.cuda()
			aud = aud.cuda()
			res = res.cuda()

		o, _, _ = model(img, aud)
		_, ind = o.max(1)

		
		# Grab the correct indices
		idx = (ind == res).data.cpu().numpy().astype(bool)
		print(type(idx))
		imgList.append(img.data.cpu().numpy()[idx, :, :, :])
		audList.append(aud.data.cpu().numpy()[idx, :, :, :])
		resList.append(res.data.cpu().numpy()[idx])
		vidTagList.append(vidTag[idx])
		audTagList.append(audTag[idx])

		if i == 200:
			break

	torch.save([imgList, audList, resList, vidTagList, audTagList], "savedEmbeddings.pt")





if __name__ == "__main__":
	cuda = True
	generateEmbeddingsForVideoAudio()
	# main(use_cuda=cuda, batch_size=64)
	# getVideoEmbeddings()
	# checkValidation(use_cuda=cuda, validation="test", batch_size=128, model_name="models/avenet.pt")
	# demo()
	warnings_file.close()

