from onlineAdaptive import AdaBoostOLM
from banditAdaptive import AdaBanditBoost
import utils
import os
import random
from tqdm import tqdm
import matplotlib
matplotlib.use('agg')
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
sns.set_style('darkgrid')
import argparse
import config

# variables defined in running the script
LOSS = ''
NUM_WLS = 1
RHO = 0.0
GAMMA = 0.0
DATAFILE = ''
MULTIPLIER = 0

# this no longer shuffles the data
def run(rows, model, metric_window):
	window_accuracy = []
	starting_rows = rows[0:metric_window]
	testing_rows = rows[metric_window::]

	print 'initializing'
	for i in tqdm(range(len(starting_rows))):
		row = starting_rows[i]
		X = row[1:]
		Y = row[0]
		pred = model.predict(X)
		model.update(Y)
		window_accuracy.append(pred == Y)

	# we keep accuracy as a count because it's easier to update this way
	rolling_accuracy = sum(window_accuracy)
	avg_accuracies = []
	for i in tqdm(range(len(testing_rows))):
		avg_accuracies.append(rolling_accuracy)
		row = testing_rows[i]
		X = row[1:]
		Y = row[0]
		pred = model.predict(X)
		model.update(Y)
		window_accuracy.append(pred == Y)
		rolling_accuracy -= window_accuracy[0]
		rolling_accuracy += window_accuracy[-1]
		del window_accuracy[0]
		assert(len(window_accuracy) == metric_window)

	avg_accuracies = [x/float(metric_window) for x in avg_accuracies]
	sample_time = range(1, len(rows)+1)[metric_window:]
	assert(len(avg_accuracies) == len(sample_time))

	return sample_time, avg_accuracies

def plotRun(BISnum_examples, BISaccuracies, FISnum_examples, FISaccuracies):
	specs = '_loss='+str(LOSS)+'_num_wls='+str(NUM_WLS)+\
				'_rho='+str(RHO)+'_gamma='+str(GAMMA)+'\n_DATAFILE='+DATAFILE+\
				'_DATAMULTIPLIER='+str(MULTIPLIER)
	
	plt.plot(BISnum_examples, BISaccuracies, color='red')
	plt.plot(FISnum_examples, FISaccuracies, color='blue')
	plt.title('Number of examples vs performance\n'+specs)
	plt.xlabel('Instances count')
	plt.ylabel('Window accuracy')
	plt.tight_layout()
	plt.subplots_adjust(top=0.85)
	BISpatch = mpatches.Patch(color='red', label='BIS')
	FISpatch = mpatches.Patch(color='blue', label='FIS')

	filename_identifier = specs.replace('\n', '')
	print filename_identifier
	savefile = os.path.join(RESULTSDIR, OUTFILENAMEBASE+\
				filename_identifier+OUTFILETYPE)
	print 'saving graph to', savefile
	if os.path.exists(savefile):
		print 'error:', savefile, 'already exists'
		exit(1)
	plt.savefig(savefile)

OUTFILENAMEBASE = config.OUTFILENAMEBASE
OUTFILETYPE = config.OUTFILETYPE
DATADIR = config.DATADIR
RESULTSDIR = config.RESULTSDIR
'''
to run, call:
python2 loop_test.py --loss=zero_one --num_wls=20 --rho=0.1 --gamma=0.1 --datafile=balance-scale.csv --multiplier=1
'''
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--loss', type=str, required=True)
	parser.add_argument('--num_wls', type=int, required=True)
	parser.add_argument('--rho', type=float, required=True)
	parser.add_argument('--gamma', type=float, required=True)
	parser.add_argument('--datafile', type=str, required=True)
	parser.add_argument('--multiplier', type=int, required=True)
	args = parser.parse_args()
	LOSS = args.loss
	NUM_WLS = args.num_wls
	RHO = args.rho
	GAMMA = args.gamma
	DATAFILE = args.datafile
	MULTIPLIER = args.multiplier

	filename = os.path.join(DATADIR, DATAFILE)
	class_index = 0
	training_ratio = 0.8
	N = utils.get_num_instances(filename)
	test_N = int(N - N*training_ratio)
	assert(test_N > 0)
	rows = utils.get_rows(filename) * MULTIPLIER
	rows = utils.shuffle(rows, seed = random.randint(1, 2000000))

	FISmodel = AdaBoostOLM(loss=LOSS, gamma=GAMMA, rho=RHO)
	FISmodel.M = 100
	FISmodel.initialize_dataset(filename, class_index,
								probe_instances=N*MULTIPLIER)
	FISmodel.gen_weaklearners(num_wls=NUM_WLS,
						   min_grace=5, max_grace=20,
						   min_tie=0.01, max_tie=0.9,
						   min_conf=0.01, max_conf=0.9,
						   min_weight=5, max_weight=200,
						   seed=random.randint(1, 2000000000))

	BISmodel = AdaBanditBoost(loss=LOSS, gamma=GAMMA, rho=RHO)
	BISmodel.M = 100
	BISmodel.initialize_dataset(filename, class_index,
								probe_instances=N*MULTIPLIER)
	BISmodel.gen_weaklearners(num_wls=NUM_WLS,
						   min_grace=5, max_grace=20,
						   min_tie=0.01, max_tie=0.9,
						   min_conf=0.01, max_conf=0.9,
						   min_weight=5, max_weight=200,
						   seed=random.randint(1, 2000000000))

	print 'running BIS model ...'
	BISnum_examples, BISaccuracies = run(rows, BISmodel, test_N)
	print 'running FIS model ...'
	FISnum_examples, FISaccuracies = run(rows, FISmodel, test_N)

	plotRun(BISnum_examples, BISaccuracies,
				FISnum_examples, FISaccuracies)
	print 'done!'