import os

datasets = [
				'balance-scale.csv',
				'car.data.csv',
				'nursery.csv',
				'mice_protein.csv',
				'isolet50.csv',
				'movement.csv',
			]


seeds = list(range(4, 24))
# runs with seed=4 have already been completed
# seeds = list(range(5, 14))

rhos = [1e-3, 1e-3, 1e-3, 1e-1, 1e-1, 1e-3]
num_wls_opt = [20, 15, 10, 10, 10, 10]
num_wls_ada = [15, 15, 5, 20, 20, 20]


template = './run.sh --dataset={} --num_wls={} --gamma=0.1 --rho={} --algorithm={} --seed={}'

# seed = 4
# cmd = template.format('car.data.csv', 20, 0.1, 'optfull', seed)
# print(cmd)
# os.system(cmd)

# cmd = template.format('car.data.csv', 15, 0.001, 'optbandit', seed)
# print(cmd)
# os.system(cmd)


# for dataset, rho, num_wls in zip(datasets, rhos, num_wls_opt):
# 	cmd = template.format(dataset, num_wls, rho, 'optbandit', seed)
# 	print(cmd)
# 	os.system(cmd)

# for seed in seeds:
# 	for dataset, rho, num_wls in zip(datasets, rhos, num_wls_opt):
# 		if (dataset == 'isolet50.csv' or dataset == 'movement') and seed >13:
# 			continue
# 		cmd = template.format(dataset, num_wls, rho, 'optbandit', seed)
# 		print(cmd)
# 		os.system(cmd)

# for seed in seeds:
# 	for dataset, rho, num_wls in zip(datasets, rhos, num_wls_ada):
# 		if (dataset == 'isolet50.csv' or dataset == 'movement') and seed >13:
# 			continue
# 		cmd = template.format(dataset, num_wls, rho, 'adabandit', seed)
# 		print(cmd)
# 		os.system(cmd)

# for seed in seeds:
# 	for dataset, rho in zip(datasets, rhos):
# 		if (dataset == 'isolet50.csv' or dataset == 'movement') and seed >13:
# 			continue
# 		cmd = template.format(dataset, 10, rho, 'bin', seed)
# 		print(cmd)
# 		os.system(cmd)

for seed in seeds:
	if seed > 13:
		continue
	cmd = template.format('movement.csv', 20, 0.1, 'optfull', seed)
	print(cmd)
	os.system(cmd)

# for seed in seeds:
# 	for dataset, rho in zip(datasets, rhos):
# 		if (dataset == 'isolet50.csv' or dataset == 'movement') and seed >13:
# 			continue
# 		cmd = template.format(dataset, 20, rho, 'optfull', seed)
# 		print(cmd)
# 		os.system(cmd)

# for seed in seeds:
# 	for dataset, rho in zip(datasets, rhos):
# 		if (dataset == 'isolet50.csv' or dataset == 'movement') and seed >13:
# 			continue
# 		cmd = template.format(dataset, 20, rho, 'adafull', seed)
# 		print(cmd)
# 		os.system(cmd)