
import torchvision
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from math import sqrt

def getDevice():
	if torch.cuda.is_available():
		print("Using GPU 0")
		device = torch.device("cuda:0")
	else:
		print("No GPU, using CPU")
		device = torch.device("cpu")
	return device

def numpyImages(imgs,padding=2):
	numPerRow = int(sqrt(imgs.size()[0]))
	images = torchvision.utils.make_grid(imgs,numPerRow,padding=2)
	images_min = images.min()
	images_max = images.max()
	images = images / 2 + 0.5     # unnormalize
	npimages = images.cpu().detach().numpy()
	images = np.transpose(npimages, (1, 2, 0))
	return np.clip(images,0,1)

def conditionalPad(success,images,padding=2):
	shape = images.size()
	# Makes a background of green for success, red for failure
	background = torch.zeros((
		success.size()[0],
		3,
		shape[-2]+2*padding,
		shape[-1]+2*padding,
		))

	for i,s in enumerate(success):
		if s == 1:
			background[i,1] = 1
		else:
			background[i,0] = 1

	background_mask = torch.zeros((success.size()[0],3,shape[-2],shape[-1]))
	background_mask = F.pad(background_mask,(padding,padding,padding,padding), value=1)
	background = torch.mul(background,background_mask)

	# Make the images RGB if not
	if images.shape[1] == 1:
		images = images.expand(-1,3,-1,-1)

	paddedImages = F.pad(images,(padding,padding,padding,padding))
	background = background.type_as(paddedImages)

	return paddedImages + background

def imshow(imgs, num = 25,filename=None):
	images = numpyImages(imgs[:num,])
	plt.figure()
	# show images
	plt.imshow(images)
	if filename is None:
		plt.show()
	else:
		plt.savefig(filename,dpi=fig.dpi*4,bbox_inches='tight')

def trainModel(model,loader,optimizer,criterion,epochs,device=None,update_rate=20):
	if device is None:
		device = getDevice()

	print(f"Training device: {device}")
	model.to(device)
	epoch_size = len(loader)
	for epoch in range(epochs):  # loop over the dataset multiple times
		# For tracking the loss and making the progress bar
		epoch_loss = torch.zeros((1,))
		epoch_loss = epoch_loss.to(device)
		update_loss = torch.zeros((1,))
		update_loss = update_loss.to(device)
		dataIterator = tqdm(enumerate(loader, 0),total = epoch_size)
		dataIterator.set_description("update loss: %.3f, epoch loss: %.3f" % (0,0))

		for i, data in dataIterator:
			# get the inputs
			inputs, labels = data
			inputs, labels = inputs.to(device), labels.to(device)

			# zero the parameter gradients
			optimizer.zero_grad()

			# forward + backward + optimize
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()

			# Update progress bar
			update_loss += loss
			if i % update_rate == update_rate - 1:    # print every 500 mini-batches
				epoch_loss += update_loss
				epoch_loss, update_loss = epoch_loss.cpu(), update_loss.cpu()
				dataIterator.set_description(
					"update loss: %.3f, epoch loss: %.3f" % (
					update_loss[0] / update_rate,
					epoch_loss[0]/(i + 1),
					))
				update_loss.zero_()
				epoch_loss, update_loss = epoch_loss.to(device), update_loss.to(device)

		epoch_loss = epoch_loss.cpu()
		print("Epoch %d/%d loss: %.4f" % (epoch+1,epochs,epoch_loss[0]/epoch_size))
	model.cpu()

def testAccuracy(model,test_set,device = None, kl_tracker=None):
	if device is None:
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	cpu = torch.device("cpu")
	model.to(device)

	correct = 0.0
	total = 0.0
	with torch.no_grad():
		for data in test_set:
			images, labels = data
			if kl_tracker is not None:
				for image in images.numpy():
					kl_tracker.push(image.flatten())
			images = images.to(device)
			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			_, predicted = _.to(cpu), predicted.to(cpu)

			total += labels.size(0)
			correct += (predicted == labels).sum().item()
	model.to(cpu)
	return correct/total
