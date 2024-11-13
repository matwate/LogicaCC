# We will be benchmarking the performance of walkSat with different parameters.
from time import time
from Queens import *
from tree_less_logic import *
import threading

# We will be using the first rule of a 4 queens problem as a benchmark
# It will be tseitin transformed and then we will run walkSat with different parameters

# We will do a grid test with different values of max_flips and max_tries
# After that we'll find the best solution and do an array test with different values of temperature 

# Queens object we're going to use

q = Reinas(4)

# Grid test
max_flips =  [10, 50, 100, 500]
max_tries = [10, 50, 100, 500]

# Function to run walkSat with given parameters
def run_walkSat(max_flip, max_try, temp=None):
    start_time = time()
    # Assuming walkSat is a function from tree_less_logic module
    w = WalkSat(q.r1, max_flip, max_try, temp)
    sat, result = w.SAT_till_SAT() # This will ensure that every sat returns a valid solution
    end_time = time()
    print(f"max_flips: {max_flip}, max_tries: {max_try}, temperature:{temp}, time: {end_time - start_time}")

# Now spawn a thread for each combination of max_flips and max_tries and run them in parallel
def grid_test():
    threads = []
    for max_flip in max_flips:
        for max_try in max_tries:
            thread = threading.Thread(target=run_walkSat, args=(max_flip, max_try, 0.5))
            threads.append(thread)
            thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

def array_test_for_temp():
    temps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    threads = []
    for temp in temps:
        thread = threading.Thread(target=run_walkSat, args=(10, 500, temp))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

# Grid test was run and it found 10, 500 to be the best combination
# Now we will run array test for temperature

array_test_for_temp()