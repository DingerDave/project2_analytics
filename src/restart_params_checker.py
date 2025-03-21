import os
import subprocess
import itertools
import time

def run_main(file_path, fail_limit, growth_rate):
    """Run main.py with the given parameters and return the execution time."""
    try:
        result = subprocess.run(
            ["py", "./src/main.py", file_path, "--fail_limit", str(fail_limit), "--growth_rate", str(growth_rate)],
            capture_output=True,
            text=True,
            check=True
        )

        time_taken = float(result.stdout.strip().split()[-1])
        return time_taken
    except Exception as e:
        print(f"Error running main.py on {file_path} with fail_limit={fail_limit} and growth_rate={growth_rate}: {e}")
        return float('inf')  

def find_best_parameters(input_folder):
    """Find the best fail_limit and growth_rate combination."""
    fail_limits = range(2100, 3201, 100)  
    growth_rates = [round(x, 1) for x in list(frange(2.3, 3.6, 0.1))]  

    best_combination = None
    lowest_total_time = float('inf')

    i = 0

    for fail_limit in fail_limits:
        for growth_rate in growth_rates:
            print(f"Running with fail_limit={fail_limit} and growth_rate={growth_rate} - Best so far: {best_combination} with time {lowest_total_time:.2f} seconds")
            total_time = 0
            for file_name in os.listdir(input_folder):
                # print(f"Running main.py on {file_name} with fail_limit={fail_limit} and growth_rate={growth_rate}")
                file_path = os.path.join(input_folder, file_name)
                if os.path.isfile(file_path):
                    
                    try:
                        start_time = time.time()
                        time_taken = subprocess.run(
                            ["py", "./src/main.py", file_path, "--fail_limit", str(fail_limit), "--growth_rate", str(growth_rate)],
                            capture_output=True,
                            text=True,
                            timeout=5, 
                            check=True
                        ).stdout.strip().split()[-1]
                        time_taken = float(time_taken)
                        elapsed_time = time.time() - start_time

                        if elapsed_time > 5:
                            print(f"Skipping fail_limit={fail_limit} and growth_rate={growth_rate} as {file_name} took too long ({elapsed_time:.2f} seconds)")
                            total_time = float('inf') 
                            break

                        total_time += time_taken
                    except subprocess.TimeoutExpired:
                        print(f"Terminating fail_limit={fail_limit} and growth_rate={growth_rate} as {file_name} exceeded 40 seconds")
                        total_time = float('inf')  
                        break
                    except Exception as e:
                        print(f"Error running main.py on {file_path} with fail_limit={fail_limit} and growth_rate={growth_rate}: {e}")
                        total_time = float('inf')  
                        break

            if total_time < lowest_total_time:
                lowest_total_time = total_time
                best_combination = (fail_limit, growth_rate)

    return best_combination, lowest_total_time

def frange(start, stop, step):
    while start < stop:
        yield start
        start += step

if __name__ == "__main__":
    input_folder = "./input/"  
    best_params, best_time = find_best_parameters(input_folder)
    print(f"Final best parameters: Fail Limit = {best_params[0]}, Growth Rate = {best_params[1]}")
    print(f"Lowest total time: {best_time}")