from targets.mnistMLP import trainMNISTMLP
from targets.mnistConvNet import trainMNISTConvNet
import torch
from targets.datasets import MNIST
from targets.utils import *
import pygoko
import numpy as np

def testAttackAccuracy(model,test_set,device = None, kl_tracker=None):
	if device is None:
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	cpu = torch.device("cpu")
	model.to(device)

	incorrect = None

	correct = 0.0
	total = 0.0
	with torch.no_grad():
		for data in test_set:
			images, labels = data
			if incorrect is not None:
				images,labels = incorrect
			if kl_tracker is not None:
				for image in images.numpy():
					kl_tracker.push(image.flatten())
			images = images.to(device)
			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			_, predicted = _.to(cpu), predicted.to(cpu)
			if incorrect is None:
				incorrect_indexes = np.arange(predicted.shape[0])[predicted != labels]
				if len(incorrect_indexes) > 0:
					images = images.to(cpu)
					incorrect_image = images[incorrect_indexes[0]].repeat(images.shape[0],1,1,1)
					incorrect_label = labels[incorrect_indexes[0]].repeat(images.shape[0],1)
					incorrect = incorrect_image,incorrect_label


			total += labels.size(0)
			correct += (predicted == labels).sum().item()
	model.to(cpu)
	return correct/total

def main():
	model1 = torch.load("models/mnistMLP.pkl")
	model2 = torch.load("models/mnistConvNet.pkl")
	mnist = MNIST()
	trainloader = mnist.training(50000)
	for arr,label in trainloader:
		train_set_array = arr.numpy().reshape([-1,784]).astype(np.float32)
	tree = pygoko.CoverTree()
	tree.set_cutoff(50)
	tree.set_scale_base(1.3)
	tree.set_resolution(-30)
	tree.fit(train_set_array)

	print("============= KL Divergence Normal =============")
	kl_tracker = tree.kl_div_sgd(0.005,0.9)
	testloader = mnist.testing()
	accuracy = testAccuracy(model1,testloader,kl_tracker=kl_tracker)
	print(accuracy)
	kl_arr = []
	for kl,address in kl_tracker.all_kl():
		kl_arr.append(kl)
		print(kl,address)
	kl_arr = np.array(kl_arr)
	print(kl_arr.mean(),kl_arr.var(),kl_arr.max(),len(kl_arr))
	print("============= KL Divergence Attack =============")

	kl_attack_tracker = tree.kl_div_sgd(0.005,0.9)
	accuracy = testAttackAccuracy(model1,testloader,kl_tracker=kl_attack_tracker)
	print(accuracy)
	kl_attack_arr = []
	for kl,address in kl_attack_tracker.all_kl():
		kl_attack_arr.append(kl)
		print(kl,address)
	kl_attack_arr = np.array(kl_attack_arr)
	print(kl_attack_arr.mean(),kl_attack_arr.var(),kl_attack_arr.max(),len(kl_attack_arr))
if __name__ == "__main__":
	main()