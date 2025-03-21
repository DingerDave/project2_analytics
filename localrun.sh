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
sed -i 's/\r$//' analytics/Scripts/Activate

source analytics/Scripts/Activate
# source p2_venv/bin/activate
# change this to point to your local installation
# CHANGE it back to this value before submitting
#export CP_SOLVER_EXEC="C:/Program Files/IBM/ILOG/CPLEX_Studio2211/cpoptimizer/bin/x64_win64/cpoptimizer.exe"
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"C:/Program Files/IBM/ILOG/CPLEX_Studio2211/cpoptimizer/bin/x64_win64":"C:/Program Files/IBM/ILOG/CPLEX_Studio2211/cpoptimizer/bin/x64_win64"

# run the solver
analytics/Scripts/python.exe src/main.py $input
