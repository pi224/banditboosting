from onlineAdaptive import AdaBoostOLM
from banditAdaptive import AdaBanditBoost
from online_binary import BinaryBanditBooster

from tqdm import tqdm
import numpy as np
import random
import argparse

import os
import config
import utils
import pickle as pkl


duplications = {
				'balance-scale.csv': 10,
				'car.data.csv': 6,
				'nursery.csv': 4,
				'mice_protein.csv': 8,
				'isolet50.csv': 15,
				'movement.csv': 1
			}


'''
Final script for running experiments. Tree hyperparameters and dataset duplication counts
are set here (because these things shouldn't change very often)
parameters rho, num_wls, gamma, random seed, and algorithm as sent in.

Upon finishing, this outputs last 20% accuracy and total accuracy to a .txt file
and the log of correct-incorrect predictions is saved in a pickle file of the same name

when running, try this:
python2 run.py --dataset='balance-scale.csv' --num_wls=1 --gamma=0.1 --rho=0.001 --algorithm=adabandit --seed=42
'''
def write(dataset, num_wls, gamma, rho, algorithm, seed, correctness):
	'''
	all the parameters are needed because this calculates the name
	'''
	num_instances = len(correctness)
	total_acc = sum(correctness) / float(num_instances)

	test_instances = int(num_instances * 0.2)
	test_acc = sum(correctness[-test_instances::]) / float(test_instances)


	name = ''
	name += '_dataset=' + dataset
	name += '_num_wls' + str(num_wls)
	name += '_gamma=' + str(gamma)
	name += '_rho=' + str(rho)
	name += '_algorithm=' + algorithm
	name += '_seed=' + str(seed)

	textfile = os.path.join(config.RESULTSDIR, name + '.txt')
	pklfile = os.path.join(config.RESULTSDIR, name + '.pkl')

	text = 'total_acc: ' + str(total_acc) + '\n'
	text += 'test 20 perc. acc:' + str(test_acc)
	print text

	if os.path.exists(textfile) or os.path.exists(pklfile):
		print 'a file already exists. exiting without saving ...'
		return

	with open(textfile, 'w') as file:
		file.write(text)
	with open(pklfile, 'wb') as file:
		pkl.dump(correctness, file)


def run(dataset, model):
	datafilepath = os.path.join(config.DATADIR, dataset)
	multiplier = duplications[dataset]
	rows = utils.get_rows(datafilepath) * multiplier
	rows = utils.shuffle(rows, seed = random.randint(1, 2000000))
	num_rows = len(rows)

	full_info_feedback = isinstance(model, AdaBoostOLM)
	correctness = [False] * num_rows
	for i in tqdm(range(num_rows)):
		row = rows[i]
		X = row[1:]
		Y = row[0]
		pred = model.predict(X)
		if full_info_feedback:
			model.update(Y)
		else:
			model.update(Y == pred)
		correctness[i] = pred == Y

	return correctness


def get_model(dataset, num_wls, gamma, rho, algorithm):
	'''
	Function that encodes static hyperparameters
	'''
	datafilepath = os.path.join(config.DATADIR, dataset)
	class_index = 0
	N = utils.get_num_instances(datafilepath)

	if algorithm == 'bin':
		model = BinaryBanditBooster(gamma=gamma, rho=rho)
		model.initialize_dataset(datafilepath, class_index, N)
		model.gen_weaklearners(num_wls=num_wls,
						min_grace=50, max_grace=200,
						min_tie=0.01, max_tie=0.9,
						min_conf=0.00000000001, max_conf=0.9,
						min_weight=5, max_weight=20,
						seed=random.randint(1, 2000000000))
		return model

	if algorithm == 'optbandit' or algorithm == 'adabandit':

		loss = ''
		if algorithm == 'optbandit':
			loss = 'zero_one'
		else:
			loss = 'logistic'

		print 'loss:', loss

		divide = dataset == 'isolet50.csv'
		model = AdaBanditBoost(loss=loss, gamma=gamma, rho=rho, divide=divide)
		model.initialize_dataset(datafilepath, class_index, N)
		model.M = 100

		if dataset == 'isolet50.csv' or dataset == 'mice_protein.csv':
			model.gen_weaklearners(num_wls=num_wls,
						min_grace=300, max_grace=301,
						min_tie=0.7, max_tie=0.8,
						min_conf=0.00000000001, max_conf=0.0000000001,
						min_weight=5, max_weight=20)
		else:
			model.gen_weaklearners(num_wls=num_wls,
						min_grace=5, max_grace=20,
						min_tie=0.01, max_tie=0.9,
						min_conf=0.01, max_conf=0.9,
						min_weight=5, max_weight=200,
						seed=random.randint(1, 2000000000))
		return model

	if algorithm == 'adafull' or algorithm == 'optfull':
		loss = ''
		if algorithm == 'optfull':
			loss = 'zero_one'
		else:
			loss = 'logistic'

		print 'loss:', loss

		model = AdaBoostOLM(loss=loss, gamma=gamma)
		model.initialize_dataset(datafilepath, class_index, N)
		model.M = 100
		model.gen_weaklearners(num_wls=num_wls,
						min_grace=50, max_grace=200,
						min_tie=0.01, max_tie=0.9,
						min_conf=0.00000000001, max_conf=0.9,
						min_weight=5, max_weight=20,
						seed=random.randint(1, 2000000000))
		return model

	assert False, 'model option not found'
	return


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--num_wls', type=int, required=True)
	parser.add_argument('--dataset', type=str, required=True)
	parser.add_argument('--algorithm', type=str, required=True)
	parser.add_argument('--gamma', type=float, required=True)
	parser.add_argument('--rho', type=float, required=True)
	parser.add_argument('--seed', type=int, required=True)
	args, unknown = parser.parse_known_args()

	# now everything is deterministic!
	random.seed(args.seed)
	np.random.seed(args.seed + random.randint(1, 100000000))

	model = get_model(args.dataset, args.num_wls, args.gamma, args.rho, args.algorithm)

	correctness = run(args.dataset, model)

	write(args.dataset, args.num_wls, args.gamma, args.rho, args.algorithm, args.seed, correctness)