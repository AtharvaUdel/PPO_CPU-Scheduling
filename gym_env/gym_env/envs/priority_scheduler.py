import gymnasium as gym
from gymnasium import spaces
from queue import PriorityQueue
import numpy as np

class PrioritySchedulerEnv(gym.Env):
    def __init__(self, data, encoder_context, max_priority) -> None:
        super(PrioritySchedulerEnv, self).__init__()

        self.data = data
        self.encoder_context = encoder_context
        self.max_priority = max_priority

        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(encoder_context, 4))

        self.action_space = spaces.Discrete(max_priority)

        self.reset()

    def _get_info(self):
        return None

    def reset(self):
        self.total_instructions = 0
        self.processes = []
        for pid in range(self.data.shape[0]):
            arrival_time = self.data[pid,1]
            instructions = self.data[pid,2]
            self.total_instructions += instructions
            self.processes.append([pid, arrival_time, instructions, instructions])

        self.current_time = -1
        self.data_pointer = 0
        self.completed_processes = []
        self.current_processes = []
        self.execution_queue = PriorityQueue()

        info = self._get_info()

        return self.execution_queue[:self.encoder_context], info
    
    def step(self, action):
        # execute as much as possible between last step and now of current process
        # new process arrives
        # assign process priority with action
        # place process into prio queue
        # repeat until action list empty
        delta_time = self.processes[self.data_pointer, 1] - self.current_time  # get time difference between last and this step
        
        # Add next process to current observation, add to priority queue, remove from list of processes
        self.current_processes.append(self.processes[self.data_pointer])
        self.execution_queue.put((action, self.processes[self.data_pointer]))
        self.processes.pop(self.data_pointer)

        # Update current time to arrival time of this process
        self.current_time = self.processes[self.data_pointer, 1]
        self.data_pointer += 1 # increment data pointer

        # Update highest priority process based on change in time
        if len(self.processes) == 0:
            delta_time = self.total_instructions
        for _ in range(delta_time):
            if self.execution_queue.not_empty:
                current_process = self.execution_queue.queue[0]
                remaining_instructions = current_process[1][3]
                remaining_instructions -= 1
                if remaining_instructions == 0:
                    _, proc = self.execution_queue.get()
                    pid = proc[0]
                    working_index = self.current_processes[:,0].index(pid)
                    turnaround_time = self.current_time - proc[1] 
                    self.completed_processes.append((pid, turnaround_time))
                    self.current_processes.pop(working_index)
                else:
                    priority = current_process[0]
                    pid = current_process[1][0]
                    arrival = current_process[1][1]
                    instructions = current_process[1][2]
                    self.execution_queue.queue[0] = (priority, (pid, arrival, instructions, remaining_instructions))
            else:
                break

        # Calculate Reward
        reward = len(self.completed_processes)

        # Check if all processes completed
        terminated = (len(self.processes) == 0) & (len(self.current_processes) == 0)

        info = self._get_info()

        return self.current_processes[:self.encoder_context], reward, terminated, False, info
    
    def render(self, mode='human'):
        print(f"Current Time: {self.current_time}")
        print("Running Processes:")
        for priority, process in self.execution_queue.queue:
            print(f"  Priority: {process[0]}, PID: {process[0]}, Arrival Time: {process[1]}, Instructions: {process[2]}, Remaining: {process[3]}")
        if self.execution_queue.not_empty:
            print(f"Current Process: PID {self.execution_queue.queue[0]}")
        print("Completed Processes:")
        for pid, turnaround_time in self.completed_processes:
            print(f"  PID: {pid}, Turnaround Time: {turnaround_time}")
        print()
