import json
import pickle

from pyspark.mllib.linalg import SparseVector

from common_utils import add_slash_to_dir, make_sure_path_exists

"""
Used in loading the count-vector RDD saved as text files.
"""

def label_and_sparse_vec(x):
    split1 = x.strip('(').strip(')').split(',',1)
    label = int(split1[0].strip('u\'').strip('\''))
    sparse_vec_str = split1[1].strip().strip('SparseVector').strip('(').strip(')').split(',', 1)
    dim = int(sparse_vec_str[0])
    counts = {int(y.split(':')[0]):float(y.split(':')[1]) for y in sparse_vec_str[1].strip().strip('{').strip('}').split(',')}
    return (label, SparseVector(dim, counts))

"""
Use to get the original column dictionary from the saved json file (created by matrix_builders.py).
"""

def load_dict(filename):
    return json.load(open(filename, 'r'))

"""
Use to get the original count-vector matrix from the text files saved using Spark's 'saveAsTextFile'.
SparkSession must be provided to the function.
"""

def load_count_vector_matrix(spark, dir_name):
    loaded_data = spark.textFile(dir_name)
    converted_rdd = loaded_data.map(lambda x: label_and_sparse_vec(x))
    return converted_rdd

"""
Common function for loading count-vectorised data and their column dictionaries.
Inputs:

spark: SparkContext instance
hdfs_dir_name: For the matrix itself, loaded using Spark.
dict_dir_name: For the dict which is a JSON file, loaded in the usual Pythonic manner.

Outputs:

col_dict
count_vector_matrix
"""

def load_count_vec_data(spark, hdfs_dir_name, dict_dir_name):

    hdfs_dir_name = add_slash_to_dir(hdfs_dir_name)
    dict_dir_name = add_slash_to_dir(dict_dir_name)

    count_vector_matrix = load_count_vector_matrix(spark, hdfs_dir_name + 'count_vector_matrix')
    col_dict = load_dict(dict_dir_name + 'col_dict.json')
    return col_dict, count_vector_matrix

"""
Loads tuples of the form (doc_id, word_id, count) into an RDD.
"""

def load_three_tuple_rdd(spark, dir_name):
    rdd = spark.textFile(dir_name)
    rdd = rdd.map(lambda x: [u.strip() for u in x.strip('u').strip('\'').strip('(').strip(')').split(',')])
    rdd = rdd.map(lambda x: (int(x[0]), int(x[1]), float(x[2])))
    return rdd

"""
Receives file names (for the rdd it's the dir, but for the json it's the name), and saves the rdd and dict.
"""

def save_rdd_mat(output_dir_rdd, rdd_matrix, filename ='count_vector_matrix'):
    output_dir_rdd = add_slash_to_dir(output_dir_rdd)
    output_name_rdd = output_dir_rdd + filename
    rdd_matrix.saveAsTextFile(output_name_rdd)
    print('***** RDD matrix saved. *****')

def save_dict(output_dir_dict, out_dict, filename ='col_dict.json'):
    output_dir_dict = add_slash_to_dir(output_dir_dict)
    output_name_dict = output_dir_dict + filename
    make_sure_path_exists(output_dir_dict)
    json.dump(out_dict, open(output_name_dict, mode='w'))

def tsv_to_rdd(spark, filename):
    data = spark.textFile(filename)
    data_rdd = data.map(lambda x: x.strip('u').strip('\'').split())
    return data_rdd

def save_id_list(id_list, dir_name):
    pickle.dump(id_list, open(add_slash_to_dir(dir_name)+'test_ids.pickle', mode='wb'))

def load_id_list(dir_name):
    return pickle.load(open(add_slash_to_dir(dir_name)+'test_ids.pickle', mode='rb'))

