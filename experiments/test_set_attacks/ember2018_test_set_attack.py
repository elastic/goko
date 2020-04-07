import lightgbm as lgb
import pygrandma
import numpy as np
from collections import defaultdict
import argparse,os
import json
from ember import read_vectorized_features


def unpack_stats(dataframe,stats):
	dataframe["max"].append(stats.max)
	dataframe["min"].append(stats.min)
	dataframe["nz_count"].append(stats.nz_count)
	dataframe["moment1_nz"].append(stats.moment1_nz)
	dataframe["moment2_nz"].append(stats.moment2_nz)
	dataframe["sequence_len"].append(stats.sequence_len)

def classify(y,pred,ember_threshold):
	return (ember_threshold < y and pred == 0) or (y < ember_threshold and pred == 1)

def uniform_generator_factory():
	def generator():
		np.random.randint(0,len(X_test))
	return generator

def normal_generator_factory():
	def generator():
		np.random.randint(0,len(X_test))
	return generator
ember_threshold = 0.8336
base_parameters = {
	"prior_weight" : 1.0,
	"observation_weight" : 1.3,
	"sequence_cap" : 50,
	"sequence_len" : 40,
	"sequence_count" : 20,
	"file_folder_prefix" : "uniform",
	"index_factory": uniform_generator_factory,
}

non_uniform_parameters = {
	"prior_weight" : 1.0,
	"observation_weight" : 1.3,
	"sequence_cap" : 50,
	"sequence_len" : 400,
	"sequence_count" : 200,
	"file_folder_prefix" : "normal",
	"index_factory": normal_generator_factory,
}

def baseline(parameters,tree):
	training_runs = []
	normal_stats = tree.kl_div_dirichlet_basestats(
		parameters["prior_weight"],
		parameters["observation_weight"],
		parameters["sequence_len"],
		parameters["sequence_count"],
		parameters["sequence_cap"])
	for i,vstats in enumerate(normal_stats):
		training_run = defaultdict(list)
		for stats in vstats:
		    unpack_stats(training_run,stats)

		training_runs.append(training_run)

	with open(f"{parameters['file_folder']}/expected_ember_stats.json","w") as file:
		for training_run in training_runs:
			file.write(json.dumps(training_run)+"\n")

def run_normal(parameters,tree):
	normal_runs = []
	for normal in range(sequence_count):
		normal_run = []
		normal_test_tracker = tree.kl_div_dirichlet(
			parameters["prior_weight"],
			parameters["observation_weight"],
			parameters["sequence_cap"])
		normal_test_stats = defaultdict(list)
		index_generator = parameters["index_factory"]()
		for i in range(sequence_len):
			i = index_generator()
			x = X_test[i]
			y = y_test[i]
			pred = bst.predict(x.reshape(1,-1))[0]

			normal_test_tracker.push(x)
			normal_test_stats["index"].append(int(i))
			normal_test_stats["y"].append(int(y))
			normal_test_stats["pred"].append(float(pred))
			unpack_stats(normal_test_stats,normal_test_tracker.stats())
		normal_runs.append(normal_test_stats)
	with open(f"{parameters['file_folder']}/normal_runs_ember_stats.json","w") as file:
		for normal_run in normal_runs:
			file.write(json.dumps(normal_run)+"\n")

def run_attacks(parameters,tree):
	attack_runs = []
	for attack in range(sequence_count):
		attack_index = None
		attack_run = []
		attack_test_tracker = tree.kl_div_dirichlet(
			parameters["prior_weight"],
			parameters["observation_weight"],
			parameters["sequence_cap"])
		attack_test_stats = defaultdict(list)
		index_generator = parameters["index_factory"]()
		for i in range(sequence_len):
			if attack_index is None:
				i = index_generator()
			else:
				i = attack_index
			x = X_test[i]
			y = y_test[i]
			pred = bst.predict(x.reshape(1,-1))[0]
			attack_test_stats["index"].append(int(i))
			attack_test_stats["y"].append(int(y))
			attack_test_stats["pred"].append(float(pred))
			if (pred < ember_threshold and y == 1) and attack_index is None:
				attack_index = i
			attack_test_tracker.push(x)
			unpack_stats(attack_test_stats,attack_test_tracker.stats())
		attack_runs.append(attack_test_stats)
	with open(f"{parameters['file_folder']}/attack_runs_ember_stats.json","w") as file:
		for attack_run in attack_runs:
			file.write(json.dumps(attack_run)+"\n")

def main():
	prog = "test set attack of ember"
	parser = argparse.ArgumentParser()
	parser.add_argument("-t","--treeyaml", type=str, required=True, help="gridsearch to find best parameters")
	parser.add_argument("datadir", metavar="DATADIR", type=str, help="Directory with raw features")
	args = parser.parse_args()

	model_dir = os.path.join(args.datadir, 'model.txt')
	bst = lgb.Booster(model_file=model_dir)  # init model

	X_test, y_test = read_vectorized_features(args.datadir, subset="test")
	X_test = X_test.copy()
	y_test = y_test.copy()

	tree = pygrandma.PyGrandma()
	tree.fit_yaml(args.treeyaml)

	parameters_list = [
		base_parameters,
		non_uniform_parameters,
	]
	for parameters in parameters_list:
		for prior_weight in [0.8,0.9,1.0]:
			parameters["prior_weight"] = prior_weight
			for observation_weight in [1.0,1.3,1.6]:
				parameters["observation_weight"] = observation_weight
				for sequence_cap in [25,50,75]:
					parameters["sequence_cap"] = sequence_cap
					parameters["sequence_len"] = 400
					parameters["sequence_count"] = 200
					prior_weight_str = str(parameters["prior_weight"]).replace(".","-")
					observation_weight_str = str(parameters["observation_weight"]).replace(".","-")
					sequence_cap_str = str(parameters["sequence_cap"])
					sequence_len_str = str(parameters["sequence_len"])
					sequence_count_str = str(parameters["sequence_count"])
					name = parameters["file_folder_prefix"] + "_".join([
						prior_weight_str,
						observation_weight_str,
						sequence_cap_str,
						sequence_len_str,
						sequence_count_str,
					])
					parameters["file_folder"] = name
					baseline(parameters,tree)
					run_normal(parameters,tree)
					run_attacks(parameters,tree)
	


if __name__ == "__main__":
	main()