import numpy as np
from round_robin import RoundRobin

rr = RoundRobin()

# Load data from CSV file into a NumPy array
rr.data = np.genfromtxt("./dataset/test.csv", delimiter=',', skip_header=1)
rr.run()
print(rr.gantt)

#print(rr.data)


