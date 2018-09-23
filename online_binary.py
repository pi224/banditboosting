import numpy as np
import csv
import copy

from hoeffdingtree import *


class BinaryBanditBooster:
	def __init__(self, rho, gamma):
		# takes in rho, and converts it to delta
		self.rho = rho
		assert gamma < 0.5
		self.gamma = gamma

		self.theta = gamma / (2 + gamma)

	def initialize_dataset(self, filename, class_index, probe_instances=10000):
		""" CODE HERE TAKEN FROM main.py OF HOEFFDINGTREE 
		Open and initialize a dataset in CSV format.
		The CSV file needs to have a header row, from where the attribute 
		names will be read, and a set of instances containing at least one 
		example of each value of all nominal attributes.

		Args:
			filename (str): The name of the dataset file (including filepath).
			class_index (int): The index of the attribute to be set as class.
			probe_instances (int): The number of instances to be used to 
				initialize the nominal attributes. (default 100)

		Returns:
			It does not return anything. Internal dataset will be updated. 
		"""
		self.class_index = class_index
		if not filename.endswith('.csv'):
			message = 'Unable to open \'{0}\'. Only datasets in \
				CSV format are supported.'
			raise TypeError(message.format(filename))
		with open(filename) as f:
			fr = csv.reader(f)
			headers = next(fr)

			att_values = [[] for i in range(len(headers))]
			instances = []
			try:
				for i in range(probe_instances):
					inst = next(fr)
					instances.append(inst)
					for j in range(len(headers)):
						try:
							inst[j] = float(inst[j])
							att_values[j] = None
						except ValueError:
							inst[j] = str(inst[j])
						if isinstance(inst[j], str):
							if att_values[j] is not None:
								if inst[j] not in att_values[j]:
									att_values[j].append(inst[j])
							else:
								raise ValueError(
									'Attribute {0} has both Numeric and Nominal values.'
									.format(headers[j]))
			# Tried to probe more instances than there are in the dataset file
			except StopIteration:
				pass

		attributes = []
		for i in range(len(headers)):
			if att_values[i] is None:
				attributes.append(Attribute(str(headers[i]), att_type='Numeric'))
			else:
				attributes.append(Attribute(str(headers[i]), att_values[i], 'Nominal'))

		dataset = Dataset(attributes, class_index)
		self.num_classes = dataset.num_classes()
		# here, set delta from rho
		self.delta = self.rho * self.num_classes / (self.num_classes - 1)

		# this dataset is for the booster to use
		self.dataset = dataset

		tree_dataset = copy.deepcopy(dataset)
		# here, modify tree_dataset so it always has two classes
		tree_dataset.attribute(class_index).set_values([str(-1), str(1)])
		self.tree_dataset = tree_dataset

	def gen_weaklearners(self, num_wls, min_conf = 0.00001, max_conf = 0.9, 
		min_grace = 1, max_grace = 10,
		min_tie = 0.001, max_tie = 1,
		min_weight = 10, max_weight = 200, seed=None):
		''' Generate weak learners.
		Args:
			num_wls (int): Number of weak learners PER CLASS!!!!
			Other args (float): Range to randomly generate parameters
			seed (int): Random seed
		Returns:
			It does not return anything. Generated weak learners are stored in 
			internal variables. 
		'''
		if seed is not None:
			nump.random.seed(seed)
		self.num_wls = num_wls
		self.weaklearners = [[HoeffdingTree() for i in range(num_wls)] 
			for _ in range(self.num_classes)]

		min_conf = np.log10(min_conf)
		max_conf = np.log10(max_conf)
		min_tie = np.log10(min_tie)
		max_tie = np.log10(max_tie)

		for wl_row in self.weaklearners:
			for wl in wl_row:
				wl._header = self.dataset
				conf = 10 ** np.random.uniform(low=min_conf, high=max_conf)
				wl.set_split_confidence(conf)
				grace = np.random.uniform(low=min_grace, high=max_grace)
				wl.set_grace_period(grace)
				tie = 10**np.random.uniform(low=min_tie, high=max_tie)
				wl.set_hoeffding_tie_threshold(tie)

		self.alphas = np.asarray([[1.0/self.num_wls] * self.num_wls] * self.num_classes)
		self.nu = self.delta**3 / self.num_classes
		return

	def make_cov_instance(self, X):
		'''Turns a list of covariates into an Instance set to self.tree_datset 
		with None in the location of the class of interest. This is required to 
		pass to a HoeffdingTree so it can make predictions.

		Args:
			X (list): A list of the covariates of the current data point. 
					  Float for numerical, string for categorical. Categorical 
					  data must have been in initial dataset

		Returns:
			pred_instance (Instance): An Instance with the covariates X and 
					  None in the correct locations

		'''
		inst_values = list(copy.deepcopy(X))
		inst_values.insert(self.class_index, None)

		indices = range(len(inst_values))
		del indices[self.class_index]
		for i in indices:
			if self.tree_dataset.attribute(index=i).type() == 'Nominal':
				inst_values[i] = int(self.tree_dataset.attribute(index=i)
					.index_of_value(str(inst_values[i])))
			else:
				inst_values[i] = float(inst_values[i])

		pred_instance = Instance(att_values = inst_values)
		pred_instance.set_dataset(self.tree_dataset)
		return pred_instance

	def make_full_instance(self, X, Y):
		'''Makes a complete Instance set to self.tree_dataset with 
		class of interest in correct place

		Args:
			X (list): A list of the covariates of the current data point. 
					  Float for numerical, string for categorical. Categorical 
					  data must have been in initial dataset
			Y (string): the class of interest corresponding to these covariates.
		
		Returns:
			full_instance (Instance): An Instance with the covariates X and Y 
							in the correct locations

		'''

		inst_values = list(copy.deepcopy(X))
		inst_values.insert(self.class_index, Y)
		for i in range(len(inst_values)):
			if self.tree_dataset.attribute(index=i).type() == 'Nominal':
				inst_values[i] = int(self.tree_dataset.attribute(index=i)
					.index_of_value(str(inst_values[i])))
			else:
				inst_values[i] = float(inst_values[i])

		
		full_instance = Instance(att_values=inst_values)
		full_instance.set_dataset(self.tree_dataset)
		return full_instance

	def find_Y(self, Y_index):
		'''Get class string from its index
		Args:
			Y_index (int): The index of Y
		Returns:
			Y (string): The class of Y
		'''

		Y = self.dataset.attribute(index=self.class_index).value(Y_index)
		return Y

	def find_Y_index(self, Y):
		'''Get class index from its string
		Args:
			Y (string): The class of Y
		Returns:
			Y_index (int): The index of Y
		'''

		Y_index = int(self.dataset.attribute(index=self.class_index)
					.index_of_value(Y))
		return Y_index



	def rand_max(self, array):
		'''
		returns one of the randomly best indices
		'''
		array = np.asarray(array)
		max_indices = np.where(array == np.max(array))[0]
		return np.random.choice(max_indices)

	def Ind(self, statement):
		return 1 if statement else 0


	def predict(self, X):
		self.X = X
		pred_inst = self.make_cov_instance(self.X)

		# gather wl predictions and stretch and scale to between -1 and 1
		self.wl_scores = np.asarray([[wl.distribution_for_instance(pred_inst)[1] for wl in wl_row] 
			for wl_row in self.weaklearners])
		self.wl_scores = self.wl_scores*2 - 1


		# from here on out, wl_scores, is inside [-1, 1]
		class_scores = np.asarray([np.dot(weights, scores) 
			for weights, scores in zip(self.alphas, self.wl_scores)])
		pred_index = self.rand_max(class_scores)

		self.Yhat_index = pred_index
		self.Yhat = self.find_Y(self.Yhat_index)

		# selecting ytilde randomly
		options = np.arange(self.num_classes)
		self.p_t = np.asarray([self.delta/self.num_classes + 
					(1-self.delta)*self.Ind(l==self.Yhat_index)
				for l in options])
		self.Ytilde_index = np.random.choice(options, p=self.p_t)

		self.Ytilde = self.find_Y(self.Ytilde_index)
		return self.Ytilde

	def update(self, Y, X=None):
		if X is None:
			X = self.X
		y_index = self.find_Y_index(Y)

		k = self.Ytilde_index
		y_tk = 1 if self.Ytilde_index == y_index else -1
		for i in range(self.num_wls):
			# this algorithm only updates stuff for the class self.Ytilde_index
			'''
			Step 1: calculate w^i_{tk} for the weak learner
			'''
			z_i = np.sum(y_tk*self.wl_scores[k, 0:i-1] - self.theta)
			wbar_i = min((1-self.gamma)**z_i, 1.0)
			w_i = wbar_i / self.p_t[self.Ytilde_index]

			# weight created, now create instance to feed to weak learner
			full_inst = self.make_full_instance(self.X, y_tk)
			full_inst.set_weight(w_i)
			# this is necessary because nested llists
			self.weaklearners[k][i].update_classifier(full_inst)


		'''
		Step 2: update alphas
		'''
		# the non-zero portion of L_tk
		L_tk_sub = self.theta - y_tk*np.dot(self.alphas[k], self.wl_scores[k])
		grad_k = y_tk * self.wl_scores[k] if L_tk_sub > 0 else 0
		self.alphas[k] = self.alphas[k] * np.exp(-self.nu * grad_k)
		self.alphas[k] /= np.sum(self.alphas[k]) # normalize
