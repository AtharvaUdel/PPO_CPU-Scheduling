import numpy as np
from round_robin import RoundRobin

rr = RoundRobin()
rr.gantt = [-1, 0, 1, 2, 3]

# Load data from CSV file into a NumPy array
rr.data = np.genfromtxt("./dataset/example_data.csv", delimiter=',', skip_header=1)
#print(rr.data)


