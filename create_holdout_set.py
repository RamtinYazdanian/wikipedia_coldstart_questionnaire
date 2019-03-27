from pyspark import SparkConf, SparkContext
from utils.save_load_utils import load_count_vector_matrix
import sys
import random
import pickle
from utils.common_utils import add_slash_to_dir

def holdout_row_both(x, row_index, max_frac = 0.5, max_num = 20):
    if (int(max_frac*len(x.indices)) < max_num):
        return ([],[])
    indices_set = set(x.indices)
    #nonzero_sample_count =  min([int(max_frac*len(x.indices)), max_num])
    nonzero_sample_count = max_num
    zero_sample_count = 2*max_num - nonzero_sample_count
    nonzero_sample = list(x.indices[random.sample(range(len(x.indices)), nonzero_sample_count)])

    zero_sample = []
    properly_sampled = False
    while not properly_sampled:
        zero_sample = random.sample(range(x.size), zero_sample_count)
        if len(set(zero_sample) & indices_set) == 0:
            properly_sampled = True

    samples = nonzero_sample + zero_sample
    row_indices = [row_index]*len(samples)
    return (row_indices, samples)

def holdout_row_nonzero(x, row_index, max_frac = 0.5, max_num = 20):
    if (int(max_frac*len(x.indices)) < max_num):
        return ([],[])
    #nonzero_sample_count =  min([int(max_frac*len(x.indices)), max_num])
    nonzero_sample_count = max_num
    nonzero_sample = list(x.indices[random.sample(range(len(x.indices)), nonzero_sample_count)])

    row_indices = [row_index]*len(nonzero_sample)
    return (row_indices, nonzero_sample)

def main():

    usage_str = 'Receives a matrix in rdd form and outputs a list of holdout pairs.\n' \
                'Args:\n' \
                '1. Dir of matrix in HDFS\n' \
                '2. Output dir (non-HDFS)\n' \
                '3. -b for both nonzeros and zeros, -nz for nonzeros only. (any other input will default to -nz)'

    if (len(sys.argv) != 4):
        print usage_str
        return

    hdfs_dir = sys.argv[1]
    output_dir = sys.argv[2]
    nonzeros_option = sys.argv[3]

    conf = SparkConf().set("spark.driver.maxResultSize", "4G").\
            set('spark.default.parallelism', '200')
    spark = SparkContext(conf=conf)
    test_rdd_matrix = load_count_vector_matrix(spark, hdfs_dir).map(lambda x: x[1]).zipWithIndex()
    if (nonzeros_option == '-b'):
        holdout_rdd = test_rdd_matrix.map(lambda x: holdout_row_both(x[0], x[1]))
    else:
        holdout_rdd = test_rdd_matrix.map(lambda x: holdout_row_nonzero(x[0], x[1]))

    holdout_pairs = holdout_rdd.reduce(lambda x,y: (x[0]+y[0], x[1]+y[1]))

    pickle.dump(holdout_pairs, open(add_slash_to_dir(output_dir)+'holdout_pairs.pkl', mode='w'))

if __name__ == '__main__':
    main()