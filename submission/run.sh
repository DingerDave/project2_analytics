#!/bin/bash

########################################
############# CSCI 2951-O ##############
########################################
E_BADARGS=65
if [ $# -ne 1 ]
then
	echo "Usage: `basename $0` <input>"
	exit $E_BADARGS
fi
	
input=$1

source p2_venv/bin/activate
# change this to point to your local installation
# CHANGE it back to this value before submitting
export CP_SOLVER_EXEC=/local/projects/cplex/CPLEX_Studio2211/cpoptimizer/bin/x86-64_linux/cpoptimizer
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/course/cs2951o/deps/python3.9/lib

# run the solver
python3.9 src/main.py $input
