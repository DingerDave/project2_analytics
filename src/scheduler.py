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

    def __init__(self, config: CPInstance, fail_limit: int = 2300, growth_rate: int = 2.5):
        self.config = config
        self.model = CpoModel()
        self.fail_limit = fail_limit
        self.growth_rate = growth_rate
        self.build_constraints()

    def build_employee_constraints(self):
        """
        Build the constraints for a specific employee
        """
        ### For each employee
        for em_idx in range(self.config.n_employees):
            ### For each day
            employee_shifts_worked = []
            for day_idx in range(self.config.n_days):
                employee_duration = self.shift_durations[em_idx, day_idx] ### This is an integer of shift duration

                # create an index which represents the shift_worked
                shift_worked = self.shift_worked[em_idx, day_idx] ### This is an integer of shift worked
                employee_shifts_worked.append(shift_worked)
                self.model.add(if_then(shift_worked == 0, employee_duration == 0)) # If the employee is off shift, they cannot work
                self.model.add(if_then(shift_worked != 0, employee_duration != 0)) # If the employee is not off shift, they must work at least the minimum daily hours
                
                # Night Shift Constraints
                if day_idx != 0:
                    self.model.add(if_then(sum(self.shift_worked[em_idx, i] == 1 for i in range(max(0, day_idx-self.config.employee_max_consecutive_night_shifts), day_idx)) == self.config.employee_max_consecutive_night_shifts, shift_worked != 1))
                if day_idx != self.config.n_days - 1:
                    self.model.add(if_then(sum(self.shift_worked[em_idx, i] == 1 for i in range(day_idx+1, min(day_idx+self.config.employee_max_consecutive_night_shifts+1, self.config.n_days))) == self.config.employee_max_consecutive_night_shifts, shift_worked != 1))
            # Training constraint                 
            self.model.add(all_diff(employee_shifts_worked[:4]))  

    def build_day_constraints(self):
        for day in range(self.config.n_days):
                # Min number of hours worked on a day
                num_hours_worked_on_given_day = sum(self.shift_durations[employee][day] for employee in range(self.config.n_employees))
                self.model.add(num_hours_worked_on_given_day >= self.config.min_daily)

                for shift in range(self.config.n_shifts):
                    if shift == 0:
                        continue
                    # Min number of employees per shift per day
                    num_employees_working_shift_on_given_day = sum(self.shift_worked[employee][day] == shift for employee in range(self.config.n_employees))
                    self.model.add(num_employees_working_shift_on_given_day >= self.config.min_shifts[day][shift])

                    # This is an additional constraint to ensure that the number of employees working at all times is at least the minimum required
                    #   Not in the handout but we discussed it in our report, CP is fully functional with this constraint
                    # total_employees_per_time = [sum([self.shift_durations[employee][day] >= time for employee in range(self.config.n_employees)])
                    #                             for time in range(self.config.employee_min_daily, self.config.employee_max_daily+1)]
                    # for time in total_employees_per_time:
                    #     self.model.add(time >= self.config.min_shifts[day][shift])

        # max night shifts
        for employee in range(self.config.n_employees):
            employee_total_night_shifts = sum(self.shift_worked[employee][day] == 1 for day in range(self.config.n_days))
            self.model.add(employee_total_night_shifts <= self.config.employee_max_total_night_shifts)
                            
            # Weekly constraints
            for week_index in range(0, self.config.n_days, 7):
                total_week_sum_hours = sum(self.shift_durations[employee][week_index:week_index+7].tolist())
                self.model.add(total_week_sum_hours <= self.config.employee_max_weekly)
                self.model.add(total_week_sum_hours >= self.config.employee_min_weekly)

                # This is an additional constraint to ensure that the employee has at least 2 off shifts in a week
                #   Not in the handout but we discussed it in our report, CP is fully functional with this constraint
                # number_of_off_shifts = sum([shift == 0 for shift in self.shift_worked[employee][week_index:week_index+7].tolist()])
                # self.model.add(number_of_off_shifts >= 2)
           
                

    def build_constraints(self):
        """Build the constraints for the model
        """
        
        
        # ASSUMPTION 
        self.config.n_intervals_per_shift = self.config.n_intervals_in_day // (self.config.n_shifts-1)
        # Construct employee shift variables (usage: self.shifts[employee][day][interval])
        
        self.shift_worked = np.array([
                                [integer_var(0, self.config.n_shifts-1) for _ in range(self.config.n_days)]
                                for _ in range(self.config.n_employees)])
        
        self.shift_durations = np.array([
                                [integer_var(domain = [0]+[i for i in range(self.config.employee_min_daily, self.config.employee_max_daily+1)]) for _ in range(self.config.n_days)]
                                    for _ in range(self.config.n_employees)])

        self.build_employee_constraints()
        self.build_day_constraints()



    def solve(self) -> Solution:
        fail_limit = 100
        params = CpoParameters(
            Workers = 1,
            TimeLimit = 300,
            #Do not change the above values 
            SearchType="DepthFirst", # Uncomment for part 2
            # LogVerbosity = "Verbose"
            FailLimit = fail_limit
        )
        self.model.set_parameters(params)

        flat_shift_worked_vars = self.shift_worked.flatten().tolist()
        var_sel = [select_smallest(domain_size()), select_random_var()]
        val_sel = select_largest(value())
        shift_worked_phase = search_phase(flat_shift_worked_vars, var_sel, val_sel)

        flat_shift_duration_vars = self.shift_durations.flatten().tolist()
        var_sel = [select_smallest(domain_size()), select_random_var()]
        val_sel = select_largest(value())
        shift_duration_phase = search_phase(flat_shift_duration_vars, var_sel, val_sel)

        my_search_phases = [shift_worked_phase, shift_duration_phase]
        self.model.set_search_phases(my_search_phases)

        fail_limit = self.fail_limit
        growth_rate = self.growth_rate
        solution = self.model.solve()
        n_fails = 0
        while not solution:
            fail_limit = int(fail_limit * growth_rate)
            
            self.model.set_parameters({"FailLimit":fail_limit, "RandomSeed": np.random.randint(0,100000)})
            solution = self.model.solve()
            n_fails = solution.get_solver_info(CpoSolverInfos.NUMBER_OF_FAILS)

        
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
                
                start_time = (solution[self.shift_worked[i,j]]-1)
                
                if start_time == -1:
                    employee.append((-1, -1)) # Employee is off shift
                    continue
                else:
                    start_time *= self.config.n_intervals_per_shift
                
                
                # get the last index with a 1
                end_time = start_time + solution[self.shift_durations[i,j]]
                # Set the schedule
                employee.append((start_time, end_time))
            schedule.append(employee)
        return schedule

    @staticmethod
    def from_file(f, limit=2300, rate=2.5) -> Scheduler:
        # Create a scheduler instance from a config file
        config = CPInstance.load(f)
        return Scheduler(config, limit, rate)

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
    
