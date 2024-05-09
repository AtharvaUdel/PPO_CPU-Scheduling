import numpy as np
from round_robin import RoundRobin
from fifo import FIFO

rr = RoundRobin()
fifo = FIFO()

# Load data from CSV file into a NumPy array
rr.data = np.genfromtxt("./dataset/test.csv", delimiter=',', skip_header=1)
rr.run()
print("RR", rr.gantt)

fifo.data = np.genfromtxt("./dataset/test.csv", delimiter=',', skip_header=1)
fifo.run()
print("FIFO", fifo.gantt)



