import argparse
import numpy as np

def getRandomGenerator(name, seed):
    generator = np.random.default_rng(seed=seed)
    generators = {'n':generator.normal,'u':generator.uniform,'f':generator.f,'cs':generator.chisquare}
    return generators[name]

def execute(args):
    random_dist = args.distribution
    random_seed = args.seed
    print(args)

    n = args.number
    max_instructions = args.max_instructions
    max_arrival = args.max_arrival

    data = np.zeros(shape=(n, 3), dtype=np.int16)
    pid = np.array(list(range(n)))
    arrival_gen = np.random.default_rng(seed=random_seed)
    arrivals = arrival_gen.uniform(0, max_arrival, n)
    arrivals.sort()
    distribution_gen = np.random.default_rng(seed = random_seed)
    if random_dist == 'n':
        n_instructions = distribution_gen.normal(0,1,size=n) # standard normal distribution
        n_instructions = np.clip(n_instructions, -3, 3) + 3 # normal distribution is 0 mean and has infinite support, so must be clipped and recentered. 99.83% of data falls within (-3,3) bounds
        n_instructions = ((n_instructions / 6) * max_instructions).astype(np.int16) # scale 0-1 and rescale to max, then cast to int
    elif random_dist == 'u':
        n_instructions = distribution_gen.uniform(low=0, high=max_instructions, size=n).astype(np.int16)
    elif random_dist == 'f':
        n_instructions = distribution_gen.f(5, 10, size=n) # F-distribution, range (0, inf)
        n_instructions = np.clip(n_instructions, 0, 10) # 99.88% of values fall within (0,10)
        n_instructions = ((n_instructions / 10) * max_instructions).astype(np.int16) # scale 0-1 and rescale to max, then cast to int
    elif random_dist == 'cs':
        n_instructions = distribution_gen.chisquare(3, size=n) # chi squared distribution, range (0, inf)
        n_instructions = np.clip(n_instructions, 0, 10) # 98.14% of values fall within (0,10)
        n_instructions = ((n_instructions / 10) * max_instructions).astype(np.int16) # scale 0-1 and rescale to max, then cast to int
    else:
        print('Specified distribution not implemented')
        exit

    #print(pid[:5])
    #print(arrivals[:5])
    #print(n_instructions[:5])

    data[:,0] = pid
    data[:,1] = arrivals
    data[:,2] = n_instructions

    savepath = args.data_directory + args.filename + '.csv'
    np.savetxt(savepath, data, fmt='%i', delimiter=',', header='PID,ArrivalTime,InstructionCount')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='make_dataset',
        description='generates a synthetic dataset for use in training or testing a CPU scheduling algorithm',
    )
    parser.add_argument('-n', '--number', help="number of data entries to generate", required=True, type=int)
    parser.add_argument('-mi', '--max_instructions', help='maximum number of instructions a process can have', type=int, default=40)
    parser.add_argument('-ma', '--max_arrival', help='maximum time of process arrival', required=True, type=int)
    parser.add_argument('-d', '--distribution', help='type of random distribution: \n\tn(normal)\n\tu(uniform)\n\tf(F) - default\n\tcs(chi-square)', choices=['n','u','f','cs'], default='f')
    parser.add_argument('-s', '--seed', help='random seed', type=int, default=42)
    parser.add_argument('-f', '--filename', help='name of file to save', required=True)
    parser.add_argument('-dir', '--data_directory', help='directory to save dataset', default='dataset/')

    args = parser.parse_args()
    execute(args)
