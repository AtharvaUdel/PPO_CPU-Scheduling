from abc import ABC, abstractmethod

class Scheduler(ABC):
    def __init__():
        pass

    def cpu_util(self):
        pass

    def throughput(self):
        pass

    def turnaround_time(self):
        pass

    def waiting_time(self):
        pass

    def response_time(self):
        pass

    @abstractmethod
    def run():
        pass