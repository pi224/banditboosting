from onlineAdaptive import AdaBoostOLM
from banditAdaptive import AdaBanditBoost
from loop_test import run as loop_run
import utils
import os
import random
import numpy as np
from tqdm import tqdm
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
import argparse
import config

# variables defined by arguments
LOSS = ''
NUM_WLS = 1
GAMMA = 0.0
DATAFILE = ''
MULTIPLIER = 0
RHORANGESPEC = ''
# default rho range
RHORANGE = np.arange(0.01, .21, .02)

# returns the last accuracy from loop_run
def run(rows, metric_window):
	accuracies = []
	for rho in RHORANGE:
		# create model
		print 'rho='+str(rho)
		BISmodel = AdaBanditBoost(loss=LOSS, gamma=GAMMA, rho=rho)
		BISmodel.M = 100
		BISmodel.initialize_dataset(filename, class_index,
									probe_instances=N*MULTIPLIER)
		BISmodel.gen_weaklearners(num_wls=NUM_WLS,
							   min_grace=5, max_grace=20,
							   min_tie=0.01, max_tie=0.9,
							   min_conf=0.01, max_conf=0.9,
							   min_weight=5, max_weight=200,
							   seed=random.randint(1, 2000000000))
		# use loop_run to get the accuracies
		accuracies += [loop_run(rows, BISmodel, metric_window)[1][-1]]
	return RHORANGE, accuracies

# rhos is an array. accuracies is many (a matrix)
def plotRun(rhos, accuracies):
	specs = '_loss='+str(LOSS)+'_num_wls='+str(NUM_WLS)+\
				'_gamma='+str(GAMMA)+'\n_DATAFILE='+DATAFILE+\
				'_DATAMULTIPLIER='+str(MULTIPLIER)+'_arrayspec='+RHORANGESPEC
	sns.tsplot(accuracies, time=rhos)
	plt.xlabel('rho')
	plt.ylabel('accuracy')
	plt.title('rho vs performance\n'+specs)
	plt.tight_layout()
	plt.subplots_adjust(top=.85)

	filename_identifier = specs.replace('\n', '')
	savefile = os.path.join(RESULTSDIR, OUTFILENAMEBASE+\
				filename_identifier+OUTFILETYPE)
	print 'saving graph to', savefile
	if os.path.exists(savefile):
		print 'error:', savefile, 'already exists'
		exit(1)
	plt.savefig(savefile)

OUTFILENAMEBASE = 'rho_test_out'
OUTFILETYPE = config.OUTFILETYPE
DATADIR = config.DATADIR
RESULTSDIR = config.RESULTSDIR
'''
to run, call:
python2 rho_test.py --loss=zero_one --num_wls=20 --gamma=0.1 --datafile=balance-scale.csv --multiplier=1 --array_spec=0.1_1_.001
'''

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--loss', type=str, required=True)
	parser.add_argument('--num_wls', type=int, required=True)
	parser.add_argument('--gamma', type=float, required=True)
	parser.add_argument('--datafile', type=str, required=True)
	parser.add_argument('--multiplier', type=int, required=True)
	'''
	the way to use this argument is like 0.1,0.2,0.3 (list of values)
	or 0.1_0.2_20 (use numpy.arange)
	the presence of an underscore signals the second interpretation
	'''
	parser.add_argument('--array_spec', type=str, required=True)
	args, unknown = parser.parse_known_args()
	LOSS = args.loss
	NUM_WLS = args.num_wls
	GAMMA = args.gamma
	DATAFILE = args.datafile
	MULTIPLIER = args.multiplier
	if '_' in args.array_spec:
		RHORANGESPEC = [float(arg) for arg in args.array_spec.split('_')]
		RHORANGE = np.arange(*arange_args)
	else:
		RHORANGESPEC = [float(arg) for arg in args.array_spec.split(',')]
		RHORANGE = np.asarray(arange_args)

	filename = os.path.join(DATADIR, DATAFILE)
	class_index = 0
	training_ratio = 0.8
	N = utils.get_num_instances(filename)
	test_N = int(N - N*training_ratio)
	assert(test_N > 0)
	rows = utils.get_rows(filename) * MULTIPLIER
	rows = utils.shuffle(rows, seed = random.randint(1, 2000000))

	print 'running ...'
	rho_runs = [run(utils.shuffle(rows, seed=random.randint(1, 2000000)),
					test_N)[-1] for _ in range(5)]
	plotRun(RHORANGE, rho_runs)