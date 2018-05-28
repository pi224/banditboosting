from onlineAdaptive import AdaBoostOLM
from banditAdaptive import AdaBanditBoost
import utils
import os
import random
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from full_test import run as full_run
import numpy as np
sns.set_style('darkgrid')

def getModel(rho, datasize):
	model = AdaBanditBoost(loss='logistic', gamma=0.1, rho=rho)
	model.M = 100
	model.initialize_dataset(filename, class_index,
								probe_instances=datasize)
	model.gen_weaklearners(num_wls=1,
						   min_grace=5, max_grace=20,
						   min_tie=0.01, max_tie=0.9,
						   min_conf=0.01, max_conf=0.9,
						   min_weight=5, max_weight=200,
						   seed=random.randint(1, 2000000000))
	return model

if __name__ == '__main__':
	dataset = 'balance-scale.csv'
	multiplier = 6

	filename = os.path.join(DATADIR, dataset)
	class_index = 0
	training_ratio = 0.8
	N = utils.get_num_instances(filename) * multiplier
	train_N = int(N*training_ratio)
	rows = utils.get_rows(filename) * multiplier

	rhos = np.arange(0, 1, .05)
	accuracy = range(len(rhos))
	for i, rho in enumerate(rhos):
		average = 0
		for _ in range(10):
			model = getModel(rho, N)
			average += run(rows, model, train_N)
		average /= 10.0
		accuracy[i] = average