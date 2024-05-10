import numpy as np
from schedulers.round_robin import RoundRobin
from schedulers.fifo import FIFO
from schedulers.ml_prio import MLPriority


rr = RoundRobin()
fifo = FIFO()

# Load data from CSV file into a NumPy array
rr.data = np.genfromtxt("./dataset/test.csv", delimiter=',', skip_header=1)
rr.run()
print("RR", rr.gantt)

fifo.data = np.genfromtxt("./dataset/test.csv", delimiter=',', skip_header=1)
fifo.run()
print("FIFO", fifo.gantt)

data = np.genfromtxt("./dataset/test.csv", delimiter=',', skip_header=1)
ml_prio = MLPriority(data=data, encoder_context=10, max_priority=10)
prio_time = ml_prio.time_run()
print('ML Priority:', ml_prio.gantt)
print('ML Priority time:', prio_time)


