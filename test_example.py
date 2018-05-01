from hoeffdingtree import *    
from onlineAdaptive import AdaBoostOLM
from banditAdaptive import AdaBanditBoost
import utils
import os
import random

'''
BUG!
Num weak learners: 5
OnlineMBBM: 83.2
BanditBoost: 61.6
avg weight difference: 0.163848421053
avg booster weight: 0.69496
avg bandit wegith: 0.562115789474
'''

'''
with 20 wl on balance-scale, maybe the average is around 60?
with 1, idk, but the variance is crazy high
'''
DATADIR = '/mnt/c/Users/zhang/Documents/bash_home/Daniel_Tuning/data'

def main():
	# Load data
	filename = os.path.join(DATADIR, 'car.data.csv')
	class_index = 0
	training_ratio = 0.8

	N = utils.get_num_instances(filename)
	train_N = int(N*training_ratio)
	rows = utils.get_rows(filename)
	rows = utils.shuffle(rows, seed = random.randint(1, 2000))

	train_rows = rows[:train_N]
	test_rows = rows[train_N:]

	# Set parameters
	num_weaklearners = 20
	gamma = 0.1
	M = 100

	print 'Num weak learners:', num_weaklearners

	# Test Adaboost.OLM
	# model = AdaBoostOLM(loss='logistic')
	# model.initialize_dataset(filename, class_index, N)
	# dataset = model.get_dataset()
	# model.gen_weaklearners(num_weaklearners,
	#                        min_grace=5, max_grace=20,
	#                        min_tie=0.01, max_tie=0.9,
	#                        min_conf=0.01, max_conf=0.9,
	#                        min_weight=5, max_weight=200) 

	# for i, row in enumerate(train_rows):
	#     X = row[1:]
	#     Y = row[0]
	#     pred = model.predict(X)
	#     model.update(Y)

	# cnt = 0

	# for i, row in enumerate(test_rows):
	#     X = row[1:]
	#     Y = row[0]
	#     pred = model.predict(X)
	#     model.update(Y)
	#     cnt += (pred == Y)*1

	# result = round(100 * cnt / float(len(test_rows)), 2)
	# print 'Adaboost.OLM:', result

	# Test OnlineMBBM
	model = AdaBoostOLM(loss='zero_one', gamma=gamma)
	model.M = M
	model.initialize_dataset(filename, class_index, N)
	model.gen_weaklearners(num_weaklearners,
	                       min_grace=5, max_grace=20,
	                       min_tie=0.01, max_tie=0.9,
	                       min_conf=0.01, max_conf=0.9,
	                       min_weight=5, max_weight=200)

	for i, row in enumerate(train_rows):
	    X = row[1:]
	    Y = row[0]
	    pred = model.predict(X)
	    model.update(Y)

	cnt = 0

	for i, row in enumerate(test_rows):
		# for debugging

	    X = row[1:]
	    Y = row[0]
	    pred = model.predict(X)
	    model.update(Y)
	    cnt += (pred == Y)*1

	result = round(100 * cnt / float(len(test_rows)), 2)
	print 'OnlineMBBM:', result
	true_avg = model.sum/float(model.count)


	# Test BanditBoost
	model = AdaBanditBoost(loss='zero_one', gamma=gamma)
	model.M = M
	model.initialize_dataset(filename, class_index, N)
	model.gen_weaklearners(num_weaklearners,
	                       min_grace=5, max_grace=20,
	                       min_tie=0.01, max_tie=0.9,
	                       min_conf=0.01, max_conf=0.9,
	                       min_weight=5, max_weight=200)

	for i, row in enumerate(train_rows):
	    X = row[1:]
	    Y = row[0]
	    pred = model.predict(X)
	    model.update(Y)

	cnt = 0

	for i, row in enumerate(test_rows):
	    X = row[1:]
	    Y = row[0]
	    pred = model.predict(X)
	    model.update(Y)
	    cnt += (pred == Y)*1

	result = round(100 * cnt / float(len(test_rows)), 2)
	print 'BanditBoost:', result
	print 'avg weight difference:', model.sum/float(model.count) - true_avg

	return

	# debugging part
	booster_weight = 0
	bandit_weight = 0
	Z= 1000
	for _ in range(Z):
		booster = AdaBoostOLM(loss='zero_one', gamma=gamma)
		booster.M = M
		booster.initialize_dataset(filename, class_index, N)
		booster.gen_weaklearners(num_weaklearners,
		                       min_grace=5, max_grace=20,
		                       min_tie=0.01, max_tie=0.9,
		                       min_conf=0.01, max_conf=0.9,
		                       min_weight=5, max_weight=200,
		                       seed=random.randint(0, 100000000)) 
		bandit = AdaBanditBoost(loss='zero_one', gamma=gamma)
		bandit.M = M
		bandit.initialize_dataset(filename, class_index, N)
		bandit.gen_weaklearners(num_weaklearners,
		                       min_grace=5, max_grace=20,
		                       min_tie=0.01, max_tie=0.9,
		                       min_conf=0.01, max_conf=0.9,
		                       min_weight=5, max_weight=200,
		                       seed=random.randint(0, 100000000))

		first_row = train_rows[0]
		X = first_row[1:]
		Y = row[0]

		booster.predict(X)
		bandit.predict(X)
		booster.update(Y)
		bandit.update(Y)

		booster_weight += booster.sum
		bandit_weight += bandit.sum

	print 'avg booster weight:', booster_weight / float(Z)
	print 'avg bandit wegith:', bandit_weight / float(Z)

if __name__ == '__main__':
	main()