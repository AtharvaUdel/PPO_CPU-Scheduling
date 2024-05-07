from abc import ABC, abstractmethod

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
        return self.data.size[0] / len(self.gantt)

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
            waiting_times.append(1)
        pass

    def response_time(self):
        pass

    @abstractmethod
    def run():
        pass