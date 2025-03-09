from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CPInstance:
    n_weeks: int
    n_days: int
    n_days_in_week: int  # computed as n_days // n_weeks
    n_employees: int
    n_shifts: int
    n_intervals_in_day: int
    min_shifts: list[list[int]]  # demand per day per shift
    min_daily: int
    employee_min_daily: int  # except on off-shifts
    employee_max_daily: int
    employee_min_weekly: int
    employee_max_weekly: int
    employee_max_consecutive_night_shifts: int
    employee_max_total_night_shifts: int

    @staticmethod
    def load(f) -> CPInstance:
        """
        Reads in a file of the following format:
        Business_numWeeks: 1
        Business_numDays: 7
        Business_numEmployees: 14
        Business_numShifts: 4
        Business_numIntervalsInDay: 24
        Business_minDemandDayShift: 0 1 3 2 0 1 3 2 0 1 3 2 0 1 5 2 0 1 5 2 0 1 5 2 0 1 5 2
        Business_minDailyOperation: 60
        Employee_minConsecutiveWork: 4
        Employee_maxDailyWork: 8
        Employee_minWeeklyWork: 20
        Employee_maxWeeklyWork: 40
        Employee_maxConsecutiveNigthShift: 1
        Employee_maxTotalNigthShift: 2
        :param f: path to the file
        :return: a dictionary with the parameters
        """
        with open(f, "r") as fl:
            lines = fl.readlines()
            params = {}
            for line in lines:
                line = line.strip()
                if line.startswith("#"):
                    continue
                if line.startswith("Business_"):
                    key, value = line.split(":")
                    if key != "Business_minDemandDayShift":
                        params[key] = int(value)
                    else:
                        params[key] = [int(x) for x in value.split()]
                elif line.startswith("Employee_"):
                    key, value = line.split(":")
                    params[key] = int(value)
                else:
                    raise ValueError("Invalid line in file")
        n_weeks = params["Business_numWeeks"]
        n_days = params["Business_numDays"]
        n_days_in_week = n_days // n_weeks
        n_employees = params["Business_numEmployees"]
        n_shifts = params["Business_numShifts"]
        n_intervals_in_day = params["Business_numIntervalsInDay"]
        min_shifts = []
        for i in range(0, n_days * n_shifts, n_shifts):
            min_shifts.append(params["Business_minDemandDayShift"][i : i + n_shifts])
        min_daily = params["Business_minDailyOperation"]
        employee_min_daily = params["Employee_minConsecutiveWork"]
        employee_max_daily = params["Employee_maxDailyWork"]
        employee_min_weekly = params["Employee_minWeeklyWork"]
        employee_max_weekly = params["Employee_maxWeeklyWork"]
        employee_max_consecutive_night_shifts = params[
            "Employee_maxConsecutiveNigthShift"
        ]
        employee_max_total_night_shifts = params["Employee_maxTotalNigthShift"]
        return CPInstance(
            n_weeks,
            n_days,
            n_days_in_week,
            n_employees,
            n_shifts,
            n_intervals_in_day,
            min_shifts,
            min_daily,
            employee_min_daily,
            employee_max_daily,
            employee_min_weekly,
            employee_max_weekly,
            employee_max_consecutive_night_shifts,
            employee_max_total_night_shifts,
        )