from round_robin import RoundRobin

rr = RoundRobin()
rr.gantt = [-1, 0, 1, 2, 3]
print(rr.cpu_util())