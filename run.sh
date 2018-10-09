####  PBS preamble

#PBS -N bandit_boosting
#PBS -M dtzhang@umich.edu
#PBS -m bae

#PBS -A tewaria_fluxm
#PBS -l qos=flux
#PBS -q fluxm

#PBS -l nodes=1:ppn=2,pmem=16gb
#PBS -l walltime=30:00:00
#PBS -j oe
#PBS -V

####  End PBS preamble
# if [ -s "$PBS_NODEFILE" ] ; then
# 	echo "Running on"
# 	uniq -c $PBS_NODEFILE
# fi

# if [ -d "$PBS_O_WORKDIR" ] ; then
# 	cd $PBS_O_WORKDIR
# 	echo "Running from $PBS_O_WORKDIR"
# fi


# module load python-anaconda2

python2 run.py 	"$@"

