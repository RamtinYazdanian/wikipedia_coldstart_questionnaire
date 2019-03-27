from pyspark import SparkConf, SparkContext
import sys
from utils.save_load_utils import load_three_tuple_rdd, load_dict
from utils.spark_utils import tuples_rdd_to_csr
from utils.common_utils import add_slash_to_dir
import pickle

def main():
    usage_str = 'Usage:\n' \
                '1. Dir for pairs rdd.\n' \
                '2. Dir for matrix output - should also contain the info_dict json with num of rows and cols.\n' \
                '3. Name of output matrix.'
    if (len(sys.argv) != 4):
        print(usage_str)
        return

    input_rdd_dir = sys.argv[1]
    output_dir = sys.argv[2]
    out_name = sys.argv[3]

    conf = SparkConf().set("spark.driver.maxResultSize", "30G"). \
        set("spark.hadoop.validateOutputSpecs", "false"). \
        set('spark.default.parallelism', '100')
    spark = SparkContext.getOrCreate(conf=conf)

    rdd = load_three_tuple_rdd(spark, input_rdd_dir)
    info_dict = load_dict(add_slash_to_dir(output_dir)+'info_dict.json')
    n_rows = int(info_dict['rows'])
    n_cols = int(info_dict['cols'])
    result_mat = tuples_rdd_to_csr(rdd, (n_rows, n_cols))

    f1 = open(add_slash_to_dir(output_dir) + out_name + '_sparse_scipy.pickle', mode='wb')
    pickle.dump(result_mat, f1)
    f1.close()

if __name__ == '__main__':
    main()