from abc import ABC, abstractmethod
import numpy as np
import time

class Scheduler(ABC):
    def __init__(self):
        self.gantt = []
        self.data = None

    def cpu_util(self):
        "Ranges from 0 to 100 percent"
        cpu_utilization = (len(self.gantt) - self.gantt.count(-1)) / len(self.gantt) * 100
        return f"{cpu_utilization}%"

    def throughput(self):
        """Number of processes completed per time unit"""
        print("data size", self.data.shape[0])
        return self.data.shape[0] / len(self.gantt)

    def turnaround_time(self):
        """Average turnaround time - time between submission to completion"""
        turnaround_times = []
        for pid in self.data[:,0]:
            first_index = self.gantt.index(pid)
            last_index = len(self.gantt) - 1 - self.gantt[::-1].index(pid)
            turnaround_times.append(last_index - first_index)
        return sum(turnaround_times) / len(turnaround_times)

    def waiting_time(self):
        "Average waiting time - amount of time a process has been waiting in the ready queue not including execution and I/O"
        waiting_times = []
        for pid in self.data[:,0]:
            row_index = np.where(self.data[:, 0] == pid)[0]
            arrival_time = self.data[row_index[0], 1]
            instruction_count = self.data[row_index[0], 2]
            finish_time = len(self.gantt) - 1 - self.gantt[::-1].index(pid)
            waiting_times.append(finish_time - arrival_time - instruction_count)
        return sum(waiting_times) / len(waiting_times)

    def response_time(self):
        "Average response time - amount of time it takes from when a request was submitted until the first response is produced"
        response_times = []
        for pid in self.data[:,0]:
            row_index = np.where(self.data[:, 0] == pid)[0]
            arrival_time = self.data[row_index[0], 1]
            first_index = self.gantt.index(pid)
            response_times.append(first_index - arrival_time)
        return sum(response_times) / len(response_times)
    
    def time_run(self):
        "Overhead - amount of time (real) to complete running a particular dataset"
        start_time = time.time()
        self.run()
        stop_time = time.time()

        return stop_time - start_time

    @abstractmethod
    def run():
        pass