# CPU Scheduling with Deep Reinforcement Learning

## Introduction

Hello, my name is Kyle O'Donnell, and myself and a team of classmates, [Ethan Chang](https://github.com/ethanchang34) and [Atharva Vichare](https://github.com/AtharvaUdel), set out to create a CPU scheduling algorithm using a Deep Reinforcement learning technique, Proximal Policy Optimization ([PPO](https://arxiv.org/abs/1707.06347)).  This project served as a final project for our graduate Operating Systems class at the University of Delaware, CISC 663.

To replicate results, the following steps are detailed below:
1) Environment setup
2) Custom gym environment
3) Generate synthetic data
4) Train PPO reinforcement model
5) Evaluate results

Special Thanks:
1) Eric Yu for his [implementation of PPO from scratch](https://medium.com/analytics-vidhya/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8) ([Github](https://github.com/ericyangyu/PPO-for-Beginners))
2) Sanchith Hedge for his Python implementation of the [Completely Fair Scheduler](https://github.com/SanchithHegde/completely-fair-scheduler)

## Environment Setup
We used the Anaconda package manager for this project.  Install required packages with the following:
``` 
conda env create -f environment.yml
```

## Custom Gym Environment
This project required the implementation of a custom Gymnasium environment.  This needs to be installed with the following:
```
pip install -e gym_env
```

## Generate Synthetic Data
We used synthetic data to simulate processes arriving in the ready queue of the kernel.  This synthetic data was generated randomly using the file `make_dataset.py`.  This file takes the following parameters from the command line:

`-n` - number (integer) : the number of processes to simulate

`-mi` - max instructions (integer) : the maximum number of instructions any process can have

`-ma` - maximum arrival time (integer) : the latest time unit a process can arrive

`-d` - distribution ('n'|'u'|'f'|'cs') : the type of statistical distribution to use to generate number of instructions.  The strings correspond to (Normal, Uniform, Fischer(F), Chi-Squared).

`-s` - seed (integer) : seed for intantiation of random generator

`-f` - filename (string) : filename to save generated data to

`-dir` - data directory (string) : path to data directory 

The datasets used to train our PPO model can be replicated with the bash script `make_training_data.sh`

To generate custom data, use `python make_dataset.py [options]`

Data will follow the format seen in `dataset/example_data.csv`

## Train PPO Model
The training process we used can be replicated with the following code:
```
python priority_prediction/train_ppo_priority_scheduler.py
```

The model weights for the actor model will be saved to `model_weights/ml_priority_scheduler.py`

An basic example for training can be seen at `priority_prediction/run_ppo_example.py` 

## Evaluate Results
Three test suites were generated for the evaluation of the model. To generate these test files, run the command 
```
bash make_testing_data.sh
```

- Test 1 contains datasets of increasing number of tasks.
- Test 2 contains datasets of varying statistical distribution.
- Test 3 contains datasets of increasing average burst time of processes.


The following algoritms were used to compare against the PPO priority scheduler:
- First in First Out (FIFO)
- Multilevel Queue (MLQ)
- Round Robin (RR)
- Completely Fair Scheduler (CFS)

The following statistics are tracked for each algorithm:
- CPU Utilization
- Throughput
- Average Turnaround Time
- Average Waiting Time
- Average Response Time
- Overhead

Code to generate tabulated results is ;ocated within the jupyter notebook `results_report.ipynb`, and tables are saved to `results/test#`

Figure generation code is contained within `plot figures.ipynb`

