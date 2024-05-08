import argparse
import numpy as np

def getRandomGenerator(name, seed):
    generator = np.random.default_rng(seed=seed)
    generators = {'n':generator.normal,'u':generator.uniform,'f':generator.f,'cs':generator.chisquare}
    return generators[name]

def execute(args):
    rng = getRandomGenerator(args.distribution, args.seed)
    print(args)

    n = args.number
    max_instructions = args.max_instructions
    max_arrival = args.max_arrival

    data = np.zeros(shape=(n, 3), dtype=np.int16)
    arrival_gen = np.random.default_rng(seed=args.seed)
    arrival = arrival_gen.uniform(0, max_arrival, n).sort()
    


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
