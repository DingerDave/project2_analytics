from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from docplex.cp.config import *
from docplex.cp.model import *
from cpinstance import CPInstance
# silence logs
context.set_attribute("log_output", None)



def sudoku_example():
    def get_box(grid, i):
    #get the i'th box
        box_row = (i // 3) * 3
        box_col = (i % 3) * 3
        box = []
        for row in range(box_row, box_row + 3):
            for col in range(box_col, box_col + 3):
                box.append(grid[row][col])
        return box

    model = CpoModel()
    int_vars = np.array([np.array([integer_var(1,9) for j in range(0,9)]) for i in range(0,9)])
    #Columns are different
    for row in int_vars:
        model.add(all_diff(row.tolist()))
    #Rows are different
    for col_index in range(0,9):
        model.add(all_diff(int_vars[:,col_index].tolist()))
    
    for box in range(0,9):
        model.add(all_diff(get_box(int_vars,box)))
    sol = model.solve()
    if not sol.is_solution():
        print("ERROR")
    else:
        
        for i in range(0,9):
            for j in range(0,9):
                print(sol[int_vars[i,j]],end=" ")
            print()
    

def solveAustraliaBinary_example():
    Colors = ["red", "green", "blue"]
    try: 
        cp = CpoModel() 
        
        WesternAustralia =  integer_var(0,3)
        NorthernTerritory = integer_var(0,3)
        SouthAustralia = integer_var(0,3)
        Queensland = integer_var(0,3)
        NewSouthWales = integer_var(0,3)
        Victoria = integer_var(0,3)
        
        cp.add(WesternAustralia != NorthernTerritory)
        cp.add(WesternAustralia != SouthAustralia)
        cp.add(NorthernTerritory != SouthAustralia)
        cp.add(NorthernTerritory != Queensland)
        cp.add(SouthAustralia != Queensland)
        cp.add(SouthAustralia != NewSouthWales)
        cp.add(SouthAustralia != Victoria)
        cp.add(Queensland != NewSouthWales)
        cp.add(NewSouthWales != Victoria)

        params = CpoParameters(
            Workers = 1,
            TimeLimit = 300,
            SearchType="DepthFirst" 
        )
        cp.set_parameters(params)
        
        sol = cp.solve() 
        if sol.is_solution(): 
            
            print( "\nWesternAustralia:    " + Colors[sol[WesternAustralia]])
            print( "NorthernTerritory:   " +   Colors[sol[NorthernTerritory]])
            print( "SouthAustralia:      " +   Colors[sol[SouthAustralia]])
            print( "Queensland:          " +   Colors[sol[Queensland]])
            print( "NewSouthWales:       " +   Colors[sol[NewSouthWales]])
            print( "Victoria:            " +   Colors[sol[Victoria]])
        else:
            print("No Solution found!");
        
    except Exception as e:
        print(f"Error: {e}")




# [Employee][Days][startTime][EndTime]
Schedule = list[list[tuple[int, int]]]


@dataclass
class Solution:
    is_solution: bool #Is this a solution
    n_fails: int # Number of failures reported by the model
    schedule: Optional[Schedule] #The Employee Schedule. Should not be None if is_solution is true


class Scheduler:
    OFF_SHIFT = 0
    NIGHT_SHIFT = 1

    def __init__(self, config: CPInstance):
        self.config = config
        self.model = CpoModel()
        self.build_constraints()

    def build_employee_constraints(self):
        """
        Build the constraints for a specific employee
        """
        for employee in self.shifts:
            # ASSUMPTION: Assuming there is at least four days according to edstem post (format: [night_shifts (4 days), day_shifts (4 days), evening_shifts (4 days)])
            # We treat the off-shift as implict again since it will be held up by the constraints from 2.1 combined with this
            first_four_days_shifts = [[],[],[]]
            for i, day in enumerate(employee):

                # TODO: Check if this is rewriteable using numpy array functionality
                # 2.1 - Can only work one shift ----------------------------------------
                shift_length = self.config.n_intervals_per_shift
                shifts = [day[i:i+shift_length] for i in range(0, len(day), shift_length)]

                # a shift being worked is represented by a 1 in the array
                shift_worked = [max(shifts[i].tolist()) for i in range(self.config.n_shifts-1)]

                # Max shifts worked should be 1, otherwise 0 (implicit off-shift)
                self.model.add(sum(shift_worked) <= 1)
                # ------------------------------------------------------------------------
                
                # make sure the employee works enough every day that they work, and don't work too much
                total_sum = sum(day.tolist())
                self.model.add((total_sum >= self.config.employee_min_daily) | (total_sum == 0))
                self.model.add(total_sum <= self.config.employee_max_daily)
                
                # -------------------------------------------
                # make sure the employee works continuously every day
                # first_on = integer_var(-1, self.config.n_intervals_in_day-1)
                # last_on = integer_var(-1, self.config.n_intervals_in_day-1)
                # self.model.add(first_on <= last_on)
                # for i in range(self.config.n_intervals_in_day):
                #     self.model.add(if_then((first_on <= i) & (i <= last_on), day.tolist()[i] == 1))

            # 2.3 - Training Requirement ------------------------------------------------
                if i<4:
                    for j in range(len(first_four_days_shifts)):
                        first_four_days_shifts[j].append(shift_worked[j])
            
            # Check that each shift category only has a sum of 1 for the first four days 
            # (implicitly, this will cause off-shift to hold since there needs to be an off-shift for the sum 
            # of the first four days to be 1 for each of the 3 explicit shift category)
            for j in range(0, len(first_four_days_shifts)):
                self.model.add(sum(first_four_days_shifts[j]) == 1)
            # ------------------------------------------------------------------------

            # 2.4 - Weekly Constraints ------------------------------------------------
            if i % 7 == 0:
                self.model.add(sum(day[i:i+7].tolist()) >= self.config.employee_min_weekly)
                self.model.add(sum(day[i:i+7].tolist()) <= self.config.employee_max_weekly)

            # 2.5 - Night Shift Constraints ----------------------------------------
            

    def build_day_constraints(self):
        for day in range(self.config.n_days):

            # 2.2 - minDailyOperation ------------------------------------------------
            flattened_shifts = self.shifts[:,day,:].flatten()
            self.model.add(sum(flattened_shifts.tolist()) >= self.config.min_daily)
            # ------------------------------------------------------------------------

            for shift in range(self.config.n_shifts-1):
                # Add the constraints for the shifts

                # 2.2 - minDemandDayShift ------------------------------------------------
                shift_start = shift*self.config.n_intervals_per_shift
                for interval in range(self.config.n_intervals_per_shift):
                    self.model.add(sum(self.shifts[:,day,shift_start+interval].tolist()) >= self.config.min_shifts[day][shift+1])
                # ------------------------------------------------------------------------

    def build_constraints(self):
        """Build the constraints for the model
        """
        
        # ASSUMPTION 
        self.config.n_intervals_per_shift = self.config.n_intervals_in_day // (self.config.n_shifts-1)
        # Construct employee shift variables (usage: self.shifts[employee][day][interval])
        self.shifts = np.array([[[integer_var(0,1) for j in range(self.config.n_intervals_in_day)] \
                        for j in range(self.config.n_days)] \
                            for j in range(self.config.n_employees)])

        self.build_employee_constraints()
        self.build_day_constraints()



    def solve(self) -> Solution:
        params = CpoParameters(
            Workers = 1,
            TimeLimit = 300,
            #Do not change the above values 
            # SearchType="DepthFirst" Uncomment for part 2
            # LogVerbosity = "Verbose"
        )
        self.model.set_parameters(params)       

        solution = self.model.solve()
        n_fails = solution.get_solver_info(CpoSolverInfos.NUMBER_OF_FAILS)
        if not solution.is_solution():
            return Solution(False, n_fails, None)
        else:
            schedule = self.construct_schedule(solution)
            return Solution(True, n_fails, schedule)

    def construct_schedule(self, solution: CpoSolveResult) -> Schedule:
        """Convert the solution as reported by the model
        to an employee schedule (see handout) that can be returned

        Args:
            solution (CpoSolveResult): The solution as returned by the model

        Returns:
            Schedule: An output schedule that can returned
            NOTE: Schedule must be in format [Employee][Days][startTime][EndTime]
        """
        
        schedule = []
        for i in range(self.config.n_employees):
            employee = []
            for j in range(self.config.n_days):
                # Get the start and end times for the employee
                employee_day = [solution[self.shifts[i,j,k]] for k in range(self.config.n_intervals_in_day)]
                # get the first index with a 1
                if 1 not in employee_day:
                    start_time = -1
                    end_time = -1
                else:
                    print(employee_day)
                    start_time = employee_day.index(1)
                    # get the last index with a 1
                    end_time = len(employee_day) - employee_day[::-1].index(1)
                # Set the schedule
                employee.append((start_time, end_time))
            schedule.append(employee)
        return schedule

    @staticmethod
    def from_file(f) -> Scheduler:
        # Create a scheduler instance from a config file
        config = CPInstance.load(f)
        return Scheduler(config)

'''
   * Generate Visualizer Input
   * author: Adapted from code written by Lily Mayo
   *
   * Generates an input solution file for the visualizer. 
   * The file name is numDays_numEmployees_sol.txt
   * The file will be overwritten if it already exists.
   * 
   * @param numEmployees the number of employees
   * @param numDays the number of days
   * @param beginED int[e][d] the hour employee e begins work on day d, -1 if not working
   * @param endED   int[e][d] the hour employee e ends work on day d, -1 if not working
   '''
def generateVisualizerInput(numEmployees : int, numDays :int,  sched : Schedule ):
    solString = f"{numDays} {numEmployees}\n"

    for d in range(0,numDays):
        for e in range(0,numEmployees):
            solString += f"{sched[e][d][0]} {sched[e][d][1]}\n"

    fileName = f"{str(numDays)}_{str(numEmployees)}_sol.txt"

    try:
        with open(fileName,"w") as fl:
            fl.write(solString)
        fl.close()
    except IOError as e:
        print(f"An error occured: {e}")
        

if __name__ == "__main__":
    model = Scheduler.from_file("./input/7_14.sched")
    solution = model.solve()
    generateVisualizerInput(model.config.n_employees, model.config.n_days, solution.schedule)
    
