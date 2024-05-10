from scheduler import Scheduler
from queue import PriorityQueue
from priority_prediction.network import FeedForwardNN
import numpy as np

class MLPriority(Scheduler):
    def __init__(self, data, encoder_context, max_priority):
        self.data = np.zeros(shape=(data.shape[0], 4))
        self.data[:,:3] = data
        self.data[:,4] = self.data[:,3]
        self.encoder_context = encoder_context
        self.max_priority = max_priority
        self.model = FeedForwardNN(self.encoder_context * 4, self.max_priority)
        self.model.load_state_dict('model_weights/ml_priority_scheduler.pt')
        super.__init__()

    def run(self):
        self.execution_queue = PriorityQueue()
        time = 0

        while len(self.execution_queue.queue) > 0 or self.data.size > 0: # if priority queue is not empty or data is not empty

            while self.data.size > 0 and self.data[0,1] == time: # add all processes to queue that arrrive at time
                priority = self.get_priority()
                self.execution_queue.put((priority, self.data[0]))
                self.data = self.data[1:]
            
            if len(self.execution_queue.queue) > 0: # if queue is not empty
                priority, process = self.execution_queue.queue[0]
                self.gantt.append(int(process[0]))
                process[3] -= 1
                self.execution_queue.queue[0] = (priority, process)

                if process[3] == 0: # task is complete
                    _ = self.execution_queue.get()
                    time += 1
                    continue

            else: # if queue is empty, store -1 in gantt chart
                self.gantt.append(-1)

            time += 1
        return

    def get_priority(self):
        obs = self.get_observation().ravel()
        priority = self.model(obs)
        return priority

    def get_observation(self):
        obs = np.zeros((self.encoder_context, 4), dtype=np.int16)
        for i in range(self.encoder_context):
            if i < len(self.execution_queue.queue):
                obs[i,:] = self.execution_queue.queue[i][1]
            else:
                break
        return obs