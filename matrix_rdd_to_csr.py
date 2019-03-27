import pickle
import json
import sys

from pyspark import SparkContext, SparkConf
from utils.common_utils import add_slash_to_dir, intify_dict
from utils.save_load_utils import load_count_vector_matrix, load_dict

from utils.spark_utils import mat_rdd_to_csr, binarise_rdd


def main():
    usage_str = 'Takes an rdd matrix, converts it into a csr matrix. The rdd matrix can either be (id, SparseVector)' \
                ', in which case we use indices instead of the ids, or already indexed as (index, SparseVector), and' \
                'in the latter case there is also the option of filtering the rows using a dictionary ' \
                '(this is for the doc-term matrix), which maps ' \
                'the ids to indices. We also save the row dict.\n' \
                '1. rdd dir name\n' \
                '2. output dir for matrix\n' \
                '3. -b for binarise, -n otherwise\n' \
                '4. name for the output file\n' \
                '5. dir for doc filtering dict, optional, -invert for special case of already indexed input rdd'

    if (len(sys.argv) < 5 or len(sys.argv) > 6):
        print(usage_str)
        return

    input_rdd_dir = sys.argv[1]
    output_dir = sys.argv[2]
    to_bin = True

    if (sys.argv[3] == '-n'):
        to_bin = False
    elif (sys.argv[3] == '-b'):
        to_bin = True
    else:
        print(usage_str)
        return

    out_name = sys.argv[4]

    input_dict_dir = None
    already_indexed = False
    invert_rdd = False
    if (len(sys.argv) == 6):
        input_dict_dir = sys.argv[5]
        if input_dict_dir == '-invert':
            already_indexed = True
            invert_rdd = True
            input_dict_dir = None


    conf = SparkConf().set("spark.driver.maxResultSize", "30G").\
        set("spark.hadoop.validateOutputSpecs", "false").\
        set('spark.default.parallelism', '100')
    spark = SparkContext.getOrCreate(conf=conf)

    count_vec_matrix = load_count_vector_matrix(spark, add_slash_to_dir(input_rdd_dir)+'count_vector_matrix')
    if (to_bin):
        count_vec_matrix = binarise_rdd(count_vec_matrix)

    if (input_dict_dir is not None):
        in_dict = intify_dict(load_dict(add_slash_to_dir(input_dict_dir)+'col_dict.json'))
        dict_broadcast = spark.broadcast(in_dict)
        count_vec_matrix = count_vec_matrix.filter(lambda x: x[0] in dict_broadcast.value)
        count_vec_matrix = count_vec_matrix.map(lambda x: (x[1], dict_broadcast.value[x[0]]))
        already_indexed = True
        cols_num = count_vec_matrix.first()[0].size
    else:
        cols_num = count_vec_matrix.first()[1].size
        if (invert_rdd):
            count_vec_matrix = count_vec_matrix.map(lambda x: (x[1], x[0]))

    rows_num = count_vec_matrix.count()
    print('shape calculated!')
    result = mat_rdd_to_csr(count_vec_matrix, (rows_num, cols_num), already_indexed=already_indexed)

    row_dict = count_vec_matrix.map(lambda x:x[0]).zipWithIndex().collectAsMap()

    json.dump(row_dict, open(add_slash_to_dir(output_dir)+out_name+'_row_dict.json', mode='w'))

    f1 = open(add_slash_to_dir(output_dir)+out_name+'_sparse_scipy.pickle', mode='wb')
    pickle.dump(result, f1)
    f1.close()

if __name__ == '__main__':
    main()
