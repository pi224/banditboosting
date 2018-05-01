from onlineAdaptive import AdaBoostOLM
from banditAdaptive import AdaBanditBoost
import utils
import os
import random
from tqdm import tqdm

DATADIR = '/mnt/c/Users/zhang/Documents/bash_home/Daniel_Tuning/data'
M = 100
gamma = 0.1
rounds = 20
num_weaklearners = 1

def run(rows, train_N, model):
	rows = utils.shuffle(rows, seed = random.randint(1, 2000000))
	train_rows = rows[:train_N]
	test_rows = rows[train_N:]

	print 'training'
	for i in tqdm(range(len(train_rows))):
		row = train_rows[i]
		X = row[1:]
		Y = row[0]
		pred = model.predict(X)
		model.update(Y)
	cnt = 0
	print 'testing'
	for i in tqdm(range(len(test_rows))):
		row = test_rows[i]
		X = row[1:]
		Y = row[0]
		pred = model.predict(X)
		model.update(Y)
		cnt += (pred == Y)*1
	return float(cnt)/len(test_rows)

'''
one weak learner, banditboost accuracy on balance-scale doubled: 62%
20 weak learners, banditboost accuracy on balance-scale doubled: 69%

one weak learner, banditboost accuracy on car.data doubled: 61%
20 weak learners, banditboost accuracy on car.data doubled: 69%%
'''
if __name__ == '__main__':
	dataset = 'car.data.csv'
	double = False

	filename = os.path.join(DATADIR, dataset)
	class_index = 0
	training_ratio = 0.8
	N = utils.get_num_instances(filename)
	if double:
		N *= 2
	train_N = int(N*training_ratio)
	rows = utils.get_rows(filename)
	if double:
		rows += rows

	print 'running model ...'

	avg = 0
	counter = 0
	try:
		for r in range(rounds):
			print 'starting round:', r
			model = AdaBanditBoost(loss='zero_one', gamma=gamma)
			model.M = M
			model.initialize_dataset(filename, class_index, N)
			model.gen_weaklearners(num_weaklearners,
								   min_grace=5, max_grace=20,
								   min_tie=0.01, max_tie=0.9,
								   min_conf=0.01, max_conf=0.9,
								   min_weight=5, max_weight=200,
								   seed=random.randint(1, 2000000000))
			avg += run(rows, train_N, model)
			counter += 1
	except:
		print 'recovered accuracy from', counter, 'rounds:', avg/float(counter)
		exit(1)

	print 'average accuracy on last 20%:', avg/float(rounds)