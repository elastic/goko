from .targets.mnistMLP import trainMNISTMLP
from .targets.cifar10ResNet import trainCIFAR10ResNet
from .targets.cifar10Deep import trainCIFAR10Deep7,trainCIFAR10Deep10
from .targets.mnistConvNet import trainMNISTConvNet
import torch
import os

import argparse

def main():
	prog = "build_target_models"
	descr = "Build target models for attack testing purposes"
	parser = argparse.ArgumentParser(prog=prog, description=descr)
	parser.add_argument("-m", "--model", type=str, default=None, required=False, help="Target Model")
	parser.add_argument("-d", "--directory", type=str, default='./', required=False, help="Base location of models")
	args = parser.parse_args()

	targetModels = {
		"mnistMLP":trainMNISTMLP,
		"mnistConvNet":trainMNISTConvNet,
		"cifar10ResNet":trainCIFAR10ResNet,
		"cifar10Deep7":trainCIFAR10Deep7,
		"cifar10Deep10":trainCIFAR10Deep10}
    
	if not (args.model in targetModels.keys() or args.model is None):
		model_parse_error = "{} is not a supported model, try:\n".format(args.model)
		for m in targetModels.keys():
			model_parse_error += "\t" + m + "\n"
		parser.error(model_parse_error)

	if torch.cuda.is_available():
		print("Using GPU 0")
		device = torch.device("cuda:0")
	else:
		print("No GPU, using CPU")
		device = torch.device("cpu")
	cpu = torch.device("cpu")

	if torch.cuda.is_available():
		if args.model is None:
			print("Training all of them")
			for model in targetModels.values():
				model(num_components=3,device=device,directory=args.directory)
		else:
			targetModels[args.model](num_components=3,device=device,directory=args.directory)

	else:
		if args.model is None:
			print("Training MNIST MLP due to lack of GPU")
			trainMNISTMLP(num_components=3,device=device,directory=args.directory)
		else:
			targetModels[args.model](num_components=3,device=device,directory=args.directory)


if __name__ == "__main__":
	main()