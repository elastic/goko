import lightgbm as lgb
import pygrandma
import numpy as np
from collections import defaultdict
import argparse,os
import json
from ember import read_vectorized_features


def unpack_stats(dataframe,stats):
	dataframe["mean_max"].append(stats.mean_max)
	dataframe["var_max"].append(stats.var_max)
	dataframe["mean_min"].append(stats.mean_min)
	dataframe["var_min"].append(stats.var_min)
	dataframe["mean_nz_count"].append(stats.mean_nz_count)
	dataframe["var_nz_count"].append(stats.var_nz_count)
	dataframe["mean_mean"].append(stats.mean_mean)
	dataframe["var_mean"].append(stats.var_mean)
	dataframe["mean_nz"].append(stats.mean_nz)
	dataframe["var_nz"].append(stats.var_nz)
	dataframe["nz_total_count"].append(stats.nz_total_count)
	dataframe["sequence_count"].append(stats.sequence_count)
	dataframe["sequence_len"].append(stats.sequence_len)
def classify(y,pred,ember_threshold):
	return (ember_threshold < y and pred == 0) or (y < ember_threshold and pred == 1)
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

	training_expected_stats = defaultdict(list)

	normal_stats = tree.kl_div_sgd_basestats(0.005,0.8)
	for i,stats in enumerate(normal_stats):
	    unpack_stats(training_expected_stats,stats)

	with open("expected_ember_stats.json","w") as file:
		file.write(json.dumps(training_expected_stats))

	ember_threshold = 0.8336 # 1% fpr

	normal_runs = []
	for normal in range(200):
		normal_run = []
		normal_test_tracker = tree.kl_div_sgd(0.005,0.8)
		normal_test_stats = defaultdict(list)
		for i in range(1000):
			i = np.random.randint(0,len(X_test))
			x = X_test[i]
			y = y_test[i]
			pred = bst.predict(x.reshape(1,-1))[0]

			normal_test_tracker.push(x)
			normal_test_stats["index"].append(int(i))
			normal_test_stats["y"].append(int(y))
			normal_test_stats["pred"].append(float(pred))
			unpack_stats(normal_test_stats,normal_test_tracker.stats())
		normal_runs.append(normal_test_stats)
	with open("normal_runs_ember_stats.json","w") as file:
		for normal_run in normal_runs:
			file.write(json.dumps(normal_run)+"\n")
	attack_runs = []
	for attack in range(200):
		attack_index = None
		attack_run = []
		attack_test_tracker = tree.kl_div_sgd(0.005,0.8)
		attack_test_stats = defaultdict(list)
		for i in range(1000):
			if attack_index is None:
				i = np.random.randint(0,len(X_test))
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
	with open("attack_runs_ember_stats.json","w") as file:
		for attack_run in attack_runs:
			file.write(json.dumps(attack_run)+"\n")


if __name__ == "__main__":
	main()