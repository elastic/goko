import lightgbm as lgb
import pygoko
import numpy as np
from collections import defaultdict
import argparse,os
import json
from ember import read_vectorized_features
import pandas as pd
import matplotlib.pyplot as plt

def unpack_stats(dataframe,stats):
	dataframe["max"].append(stats.max)
	dataframe["min"].append(stats.min)
	dataframe["nz_count"].append(stats.nz_count)
	dataframe["moment1_nz"].append(stats.moment1_nz)
	dataframe["moment2_nz"].append(stats.moment2_nz)
	dataframe["sequence_len"].append(stats.sequence_len)

def classify(y,pred,ember_threshold):
	return (ember_threshold < y and pred == 0) or (y < ember_threshold and pred == 1)

ember_threshold = 0.8336
skip = 0

def baseline(parameters,tree):
	print(f"Creating the baseline for {parameters['file_folder']}")
	training_runs = []
	normal_stats = tree.kl_div_dirichlet_basestats(
		parameters["prior_weight"],
		parameters["observation_weight"],
		parameters["sequence_len"],
		parameters["sequence_count"],
		parameters["window_size"])
	for i,vstats in enumerate(normal_stats):
		training_run = defaultdict(list)
		for stats in vstats:
			unpack_stats(training_run,stats)

		training_runs.append(training_run)

	with open(f"{parameters['file_folder']}/expected_ember_stats.json","w") as file:
		for training_run in training_runs:
			file.write(json.dumps(training_run)+"\n")

def run_normal(parameters,tree,bst,X_test,y_test):
	print(f"Creating the normal runs for {parameters['file_folder']}")
	normal_runs = []
	for normal in range(parameters["sequence_count"]):
		normal_run = []
		normal_test_tracker = tree.kl_div_dirichlet(
			parameters["prior_weight"],
			parameters["observation_weight"],
			parameters["window_size"])
		normal_test_stats = defaultdict(list)
		index_generator = parameters["index_factory"]()
		for i in range(parameters["sequence_len"]):
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

def run_attacks(parameters,tree,bst,X_test,y_test):
	print(f"Creating the attack runs for {parameters['file_folder']}")
	attack_runs = []
	for attack in range(parameters["sequence_count"]):
		attack_index = None
		attack_run = []
		attack_test_tracker = tree.kl_div_dirichlet(
			parameters["prior_weight"],
			parameters["observation_weight"],
			parameters["window_size"])
		attack_test_stats = defaultdict(list)
		index_generator = parameters["index_factory"]()
		for i in range(parameters["sequence_len"]):
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

def open_experiment(folder_name):
	with open(f"{folder_name}expected_ember_stats.json","r") as file:
		training_runs = [json.loads(line) for line in file]
		training_keys = list(training_runs[0].keys())
		training_arrays = {k:np.array([training_run[k][skip:] for training_run in training_runs])  for k in training_keys}

		expected_runs = {}
		for k,v in training_arrays.items():
			expected_runs[f"mean_{k}"] = np.mean(v,axis=0)
			expected_runs[f"stddev_{k}"] = np.sqrt(np.var(v,axis=0))

		expected_runs["moment1_nz"] = np.sum(training_arrays["moment1_nz"],axis=0)
		expected_runs["moment2_nz"] = np.sum(training_arrays["moment2_nz"],axis=0)
		expected_runs["nz_count"] = np.sum(training_arrays["nz_count"],axis=0)
		expected_runs["mean_nz_mean"] = expected_runs["moment1_nz"]/expected_runs["nz_count"]
		expected_runs["stddev_nz_mean"] = np.sqrt(
			expected_runs["moment2_nz"]/expected_runs["nz_count"] - 
			np.square(expected_runs["mean_nz_mean"]))


	with open(f"{folder_name}normal_runs_ember_stats.json","r") as file:
		normal_runs = [{k: np.array(v[skip:]) for k,v in json.loads(line).items()} for line in file]
		for run in normal_runs:
			run["nz_mean"] = run["moment1_nz"]/run["nz_count"]
			run["nz_stddev"] = np.sqrt(run["moment2_nz"]/run["nz_count"] - np.square(run["nz_mean"]))

	with open(f"{folder_name}attack_runs_ember_stats.json","r") as file:
		attack_runs = [{k: np.array(v[skip:]) for k,v in json.loads(line).items()} for line in file]
		for run in attack_runs:
			run["nz_mean"] = run["moment1_nz"]/run["nz_count"]
			run["nz_stddev"] = np.sqrt(run["moment2_nz"]/run["nz_count"] - np.square(run["nz_mean"]))
	return expected_runs, normal_runs, attack_runs

def generate_traces(runs,col_name, yaxis_type,title, filename):
	runs["normal"]["expected_runs"][f'mean_p_stddev_{col_name}'] = runs["normal"]["expected_runs"][f"mean_{col_name}"]+runs["normal"]["expected_runs"][f'stddev_{col_name}']
	runs["normal"]["expected_runs"][f'mean_n_stddev_{col_name}'] = runs["normal"]["expected_runs"][f"mean_{col_name}"]-runs["normal"]["expected_runs"][f'stddev_{col_name}']
	runs["uniform"]["expected_runs"][f'mean_p_stddev_{col_name}'] = runs["uniform"]["expected_runs"][f"mean_{col_name}"]+runs["uniform"]["expected_runs"][f'stddev_{col_name}']
	runs["uniform"]["expected_runs"][f'mean_n_stddev_{col_name}'] = runs["uniform"]["expected_runs"][f"mean_{col_name}"]-runs["uniform"]["expected_runs"][f'stddev_{col_name}']
	x = np.arange(skip,len(runs["normal"]["expected_runs"][f"mean_{col_name}"]))


	fig,(ax2,ax1) = plt.subplots(1,2,sharey=True)
	ax1.plot(x[10:],runs["normal"]["expected_runs"][f"mean_{col_name}"][10:], color="green",
		label='Uniform Baseline')
	ax1.fill_between(
		x[10:], 
		runs["normal"]["expected_runs"][f'mean_p_stddev_{col_name}'][10:], 
		runs["normal"]["expected_runs"][f'mean_n_stddev_{col_name}'][10:], 
		color="green",
		alpha=0.5
	)
	attack = np.stack([run[col_name] for run in runs["normal"]["attack_runs"][:10]]).T
	ax1.plot(x[10:], attack[10:], 'r', color="red", label='Attack Sequence')

	normal = np.stack([run[col_name] for run in runs["normal"]["normal_runs"][:10]]).T
	ax1.plot(x[10:], normal[10:], 'r', color="blue", label='Normal Sequence')
	ax1.set_title("Binomial Interest")
	ax1.set_xlim(10,400)

	ax2.plot(x[10:],runs["uniform"]["expected_runs"][f"mean_{col_name}"][10:], color="green",
		label='Uniform Baseline')
	ax2.fill_between(
		x[10:], 
		runs["uniform"]["expected_runs"][f'mean_p_stddev_{col_name}'][10:], 
		runs["uniform"]["expected_runs"][f'mean_n_stddev_{col_name}'][10:], 
		color="green",
		alpha=0.5
	)
	attack = np.stack([run[col_name] for run in runs["uniform"]["attack_runs"][:10]]).T
	ax2.plot(x[10:], attack[10:], 'r', color="red", label='Attack Sequence')

	normal = np.stack([run[col_name] for run in runs["uniform"]["normal_runs"][:10]]).T
	ax2.plot(x[10:], normal[10:], 'r', color="blue", label='Normal Sequence')
	ax2.set_title("Uniform Interest")
	ax2.set_xlim(10,400)
	plt.yscale("log")
	fig.set_size_inches(12.00, 5.00)
	fig.set_dpi(2000)
	fig.tight_layout()
	handles, labels = plt.gca().get_legend_handles_labels()
	labels, ids = np.unique(labels, return_index=True)
	handles = [handles[i] for i in ids]
	plt.legend(handles, labels, loc='best')
	fig.savefig(filename, bbox_inches='tight')
	plt.close()

def sequence_analysis(expected_runs,sequence_run,col_name):
	stat_column = sequence_run[col_name]
	index = sequence_run["index"]
	std_dev_up = expected_runs[f"mean_{col_name}"]+expected_runs[f'stddev_{col_name}']
	std_dev_down = expected_runs[f"mean_{col_name}"]-expected_runs[f'stddev_{col_name}']

	true_explotation_time = None
	i = 0
	while true_explotation_time is None and i + 2 < len(index):
		if index[i] == index[i+1] and index[i+1] == index[i+2]:
			true_explotation_time = i
		i += 1
	    
	detection_time = None
	i = 0
	while detection_time is None and i < len(index):
		if stat_column[i] > std_dev_up[i] or stat_column[i] < std_dev_down[i]:
			detection_time = i
		i += 1
	return true_explotation_time, detection_time

def generate_results(pair):
	runs = {"normal":dict(),"uniform":{}}
	for n in pair:
		print(f"Opening {n}")
		if "normal" in n:
			fileprefix = n[len("normal"):]
			runs["normal"]["expected_runs"], runs["normal"]["normal_runs"], runs["normal"]["attack_runs"] = open_experiment(n+"/")
		else:
			runs["uniform"]["expected_runs"], runs["uniform"]["normal_runs"], runs["uniform"]["attack_runs"] = open_experiment(n+"/")

	generate_traces(runs,"max",
		title='Maximum of the KL Divergence of Query Sequences',
	    yaxis_type="log",
	    filename=f"{fileprefix}_max.png")

	generate_traces(runs,"min",
		title='Minimum of the KL Divergence of Query Sequences',
		yaxis_type="log",
		filename=f"{fileprefix}_min.png")

	generate_traces(runs,"nz_count",
		title="Count of non-zero KL Divergence",
		yaxis_type="log",
		filename=f"{fileprefix}_nz_count.png")

	generate_traces(runs,"moment1_nz",
		title='KL Divergence',
		yaxis_type="log",
		filename=f"{fileprefix}_moment1_nz.png")

	generate_traces(runs,"moment2_nz",
		title='Second Moment of KL Divergence',
		yaxis_type="log",
		filename=f"{fileprefix}_moment2_nz.png")

	data = generate_traces(runs,"nz_mean",
		title='Mean of the non-zero KL divergences',
		yaxis_type="log",
		filename=f"{fileprefix}_nz_mean.png")

	"""
	attack_results = []
	for run in attack_runs:
		attack_results.append(sequence_analysis(expected_runs,run,"nz_mean"))
	normal_results = []
	for run in normal_runs:
		normal_results.append(sequence_analysis(expected_runs,run,"nz_mean"))
	positives = 0
	negatives = 0
	results = attack_results + normal_results
	true_positive = 0.0
	false_positive = 0.0
	true_negative = 0.0
	false_negative = 0.0
	detection_mean_lag = 0.0
	for attack_start,detection in results:
		if attack_start is None:
			negatives += 1.0
			if detection is None:
				true_negative += 1.0
			else:
				false_negative += 1.0
		else:
			positives += 1.0
			if detection is None:
				false_positive += 1.0
			else:
				true_positive += 1.0
				detection_mean_lag += detection - attack_start
	detection_mean_lag /= true_positive
	true_positive /= positives
	false_positive /= positives
	true_negative /= negatives
	false_negative /= negatives
	results_summary = {
		"detection_mean_lag":detection_mean_lag,
		"true_positive":true_positive,
		"false_positive":false_positive,
		"true_negative":true_negative,
		"false_negative":false_negative,
		"positives": positives,
		"negatives": negatives,
	}
	with open(f"{folder_name}/detection_results.json","w") as file:
		file.write(json.dumps(results_summary)+"\n")
	"""
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

	tree = pygoko.CoverTree()
	tree.load_yaml_config(args.treeyaml)
	tree.fit()
	def uniform_generator_factory():
		def generator():
			return np.random.randint(0,len(X_test))
		return generator

	def normal_generator_factory():
		count = np.random.randint(0,1000)
		shift = np.random.randint(0,len(X_test) - count)
		def generator():
			return np.random.binomial(n=count, p=0.5, size=1)[0] + shift
		return generator

	parameters_list = [
		{
			"prior_weight" : 1.0,
			"observation_weight" : 1.3,
			"window_size" : 50,
			"sequence_len" : 40,
			"sequence_count" : 20,
			"file_folder_prefix" : "uniform",
			"index_factory": uniform_generator_factory,
		},
		{
			"prior_weight" : 1.0,
			"observation_weight" : 1.3,
			"window_size" : 50,
			"sequence_len" : 40,
			"sequence_count" : 20,
			"file_folder_prefix" : "normal",
			"index_factory": normal_generator_factory,
		},
	]
	file_pairs = list()
	for prior_weight in [0.8,1.0]:
		for observation_weight in [1.0,1.3,1.6]:
			for window_size in [25,50,75]:
				pair = []
				for parameters in parameters_list:
					parameters["observation_weight"] = observation_weight
					parameters["prior_weight"] = prior_weight
					parameters["window_size"] = window_size
					parameters["sequence_len"] = 400
					parameters["sequence_count"] = 100
					prior_weight_str = str(parameters["prior_weight"]).replace(".","-")
					observation_weight_str = str(parameters["observation_weight"]).replace(".","-")
					window_size_str = str(parameters["window_size"])
					sequence_len_str = str(parameters["sequence_len"])
					sequence_count_str = str(parameters["sequence_count"])
					name = parameters["file_folder_prefix"] +"_"+ "_".join([
						prior_weight_str,
						observation_weight_str,
						window_size_str,
						sequence_len_str,
						sequence_count_str,
					])
					parameters["file_folder"] = name
					pair.append(name)
					if not os.path.exists(name):
						os.mkdir(name)
						baseline(parameters,tree)
						run_normal(parameters,tree,bst,X_test,y_test)
						run_attacks(parameters,tree,bst,X_test,y_test)

				file_pairs.append(pair)

	
	for pair in file_pairs:
		generate_results(pair)

if __name__ == "__main__":
	main()