import os
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

from docplex.cp.config import context

from scheduler_old import Schedule, Scheduler,sudoku_example , generateVisualizerInput,solveAustraliaBinary_example
from model_timer import Timer 
import json
import sys
import numpy as np
# Stencil created by Anirudh Narsipur February 2023
# Adapted from code written by Alexander Ding and Anirudh Narsipur


#DO NOT change this. Change run.sh to point to your local CPLEX installation

def set_context():
    solver_exec = Path(os.environ["CP_SOLVER_EXEC"])
    if not solver_exec.exists():
        print("Error: Set CP Solver Exec\n")
        exit(1)
    print(f"Using cp installation at {solver_exec}")
    context.solver.agent = "local"
    context.solver.local.execfile = str(solver_exec)
    context.verbose = 1 #Change this value to increase/decrease log verbosity 0 is no log, 9 is max log 
    context.log_output = sys.stdout



#  Poor man's Gantt chart.
#  Displays the employee schedules on the command line. 
#  Each row corresponds to a single employee. 
#  A "+" refers to a working hour and "." means no work
#  The shifts are separated with a "|"
#  The days are separated with "||"
#  This might help you analyze your solutions. 
def visualize(schedule: Schedule):
    for e in range(len(schedule)):
        print("E" + str(e + 1) + ": ", end="")
        if e < 9:
            print(" ", end="")
        for d in range(len(schedule[0])):
            for i in range(24):
                if i % 8 == 0:
                    print("|", end="")
                if (
                    schedule[e][d][0] != schedule[e][d][1]
                    and i >= schedule[e][d][0]
                    and i < schedule[e][d][1]
                ):
                    print("+", end="")
                else:
                    print(".", end="")

            print("|", end="")
        print(" ")



def main(args):
    #set_context()
    # sudoku_example() Uncomment to run sudoku example
    #solveAustraliaBinary_example()

    input_file = Path(args.input_file)
    filename = input_file.name
   
    scheduler = Scheduler.from_file(args.input_file)
    timer = Timer()
    timer.start() 
    solution = scheduler.solve()
    timer.stop()
    
    resultdict = {}
    resultdict["Instance"] = filename
    resultdict["Time"] = str(round(timer.getElapsed(),2))
    resultdict["Result"] = str(solution.n_fails)
    if solution.is_solution:
        serialized_schedule = solution.schedule #Serialize Schedule TODO get model solution
        #print(serialized_schedule)
        #visualize(solution.schedule) # Uncomment to see Gant chart
        #generateVisualizerInput(scheduler.config.n_employees,scheduler.config.n_days,solution.schedule) # Uncomment to generate vis file
        resultdict["Solution"] =  " ".join([str(time) for time in np.array(serialized_schedule).flatten().tolist()]) # TODO  write solution to result dictionary
    print(json.dumps(resultdict))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input_file", type=str)
    args = parser.parse_args()
    main(args)
