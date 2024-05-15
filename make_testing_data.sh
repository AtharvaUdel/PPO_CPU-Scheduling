# create test directory
mkdir -p dataset/test
# test 1: datasets with increasing number of processes
python make_dataset.py -n 10 -mi 20 -ma 30 -f test/test1-1
python make_dataset.py -n 100 -mi 20 -ma 300 -f test/test1-2
python make_dataset.py -n 1000 -mi 20 -ma 3000 -f test/test1-3
python make_dataset.py -n 10000 -mi 20 -ma 30000 -f test/test1-4
python make_dataset.py -n 100000 -mi 20 -ma 300000 -f test/test1-5
# test 2: datasets with identical parameters but different distributions
python make_dataset.py -n 1000 -mi 20 -ma 20000 -d n -f test/test2-1 
python make_dataset.py -n 1000 -mi 20 -ma 20000 -d u -f test/test2-2
python make_dataset.py -n 1000 -mi 20 -ma 20000 -d f -f test/test2-3
python make_dataset.py -n 1000 -mi 20 -ma 20000 -d cs -f test/test2-4
# test 3: datasets with varying number of instructions
python make_dataset.py -n 1000 -mi 2 -ma 2000 -f test/test3-1
python make_dataset.py -n 1000 -mi 20 -ma 20000 -f test/test3-2
python make_dataset.py -n 1000 -mi 200 -ma 200000 -f test/test3-3
python make_dataset.py -n 1000 -mi 2000 -ma 2000000 -f test/test3-4
