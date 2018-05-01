import numpy as np
import random

M = 1000
k = 4
gamma = 0.3
def mc_potential(t, b, s):
	'''Approximate potential via Monte Carlo simulation
	Arbs:
		t (int)     : number of weak learners until final decision
		b (list)    : baseline distribution
		s (list)    : current state
	Returns:
		potential value (float)
	'''
	k = len(b)
	r = 0
	cnt = 0
	for _ in xrange(M):
		x = np.random.multinomial(t, b)
		x = x + s
		tmp = [z for z in range(k)
				if x[z] == max(x)]
		i = np.random.choice(tmp)
		if i != r:
			cnt += 1

	return float(cnt) / M

def arbitrary_mc_potential(t, b, s, L):
	'''Approximate potential via Monte Carlo simulation
	Arbs:
		t (int)     : number of weak learners until final decision
		b (list)    : baseline distribution
		s (list)    : current state
		L (array)	: loss function
	Returns:
		potential value (float)
	'''
	k = len(b)
	cnt = 0
	for _ in xrange(M):
		x = np.random.multinomial(t, b)
		x = x + s

		tmp = [z for z in range(k)
					if x[z] == max(x)]
		i = np.random.choice(tmp)
		cnt += L[i]
	return float(cnt) / M


def get_potential(r, n, s):
	'''Compute potential
	Args:
		r (int): True label index
		n (int): Number of weak learners until final decision
		s (list): Current state
	Returns:
		(float) potential function
	'''

	new_s = list(s)
	new_s[r] = -np.inf
	new_s.sort()
	new_s[0] = s[r]
	print new_s

	biased_uniform = np.ones(k)*(1-gamma)/k
	biased_uniform[0] += gamma
	value = mc_potential(n, biased_uniform, new_s)
	return value

def get_arbitrary_potential(r, n, s, L):
	'''Compute potential
	Args:
		r (int): True label index
		n (int): Number of weak learners until final decision
		s (list): Current state
		L (array): the loss at each index
	Returns:
		(float) potential function
	'''
	distribution = np.ones(k) * (1-gamma)/float(k)
	distribution[r] += gamma
	return arbitrary_mc_potential(n, distribution, s, L)


n = 1
s = [1, 0, 0, 0, 0]
r = 2
L = np.ones(k)
L[r] = 0

r = 0
n = 0
s = [1, 0, 1, 0]
L = [0, 1.08, 0, 0]
print get_arbitrary_potential(r, n, s, L)

# sum = 0
# rounds = 100
# rho = 0.1
# for _ in range(rounds):
# 	# first, draw a "prediction"
# 	y = random.randint(0, k-1)

# 	# create a distribution using it
# 	p = np.ones(k) * (1-rho)/k
# 	p[y] += rho

# 	Lhat = np.zeros(k)
# 	i = np.random.choice(k, p=p)
# 	Lhat[i] = L[i] / p[i]
# 	sum += get_arbitrary_potential(r, n, s, Lhat)
# print sum / float(rounds) - get_potential(r, n, s)