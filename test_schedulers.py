import numpy as np
from schedulers.round_robin import RoundRobin
from schedulers.fifo import FIFO
from schedulers.cfs import CFS
from schedulers.ml_prio import MLPriority
from schedulers.mlq import MLQ
#from schedulers.mfq import MFQ

def test_scheduler(scheduler, csv="./dataset/test/test1-5.csv", **kwargs):
    data = np.genfromtxt(csv, delimiter=',', skip_header=1)
    sched = scheduler(data, **kwargs)
    sched.time_run()
    sched.calc_stats()
    return sched
'''
print('mlq')
mlq = test_scheduler(MLQ)
mlq.print_stats()

#print('mfq')
#mlq = test_scheduler(MFQ)
#mlq.print_stats()

print('rr')
rr = test_scheduler(RoundRobin)
rr.print_stats()
'''
print('fifo')
fifo = test_scheduler(FIFO)
fifo.print_stats()
'''
print('cfs')
cfs = test_scheduler(CFS)
cfs.print_stats()

print('ml prio')
ml_prio = test_scheduler(MLPriority, encoder_context=10, max_priority=10)
ml_prio.print_stats()
'''