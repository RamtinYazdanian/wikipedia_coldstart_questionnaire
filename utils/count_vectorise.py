from collections import Counter
from pyspark import SparkContext
from pyspark.mllib.linalg import SparseVector

"""
Receives a SparkContext and an rdd that has been tokenised (more on that later), and performs count-vectorisation.

Inputs:

spark: the SparkContext

tokenised_rdd: An rdd consisting of tuples, in which the first is the key (e.g. user id for user-doc,
or doc_id for doc-term) and the second is the list of values (e.g. doc_id's of docs the user has edited, or
terms contained in the 'key' document). An example (for the doc-term case, doc_id as key) is as follows:
(12, ['Russia', 'Soviet Union'])
(895, ['Institute', 'Technology'])

Outputs:

col_dict: a dictionary that maps values (e.g. docs in the user-doc case, terms in the doc-term case) to their
respective columns.

count_vector_matrix: a matrix in which each row is a tuple consisting of one of the keys, and
a sparse vector showing containing the value of each non-zero column at that row.
"""


def count_vectorise_simple(spark, tokenised_rdd):
    # zipWithIndex creates tuples in which the index is member 1 and your data is member 0.
    col_dict = tokenised_rdd.map(lambda x: x[1]).flatMap(lambda x: x) \
        .distinct().zipWithIndex().collectAsMap()
    #If we don't broadcast it, all machines will have to query the machine that has it, slowing the algorithm down.
    col_map = spark.broadcast(col_dict)
    col_map_size = spark.broadcast(len(col_dict))
    count_vector_matrix = tokenised_rdd.map(lambda x: (x[0], Counter(x[1]))) \
        .map(lambda x: (x[0], {col_map.value[token]: float(x[1][token]) for token in x[1]})) \
        .map(lambda x: (x[0], SparseVector(col_map_size.value, x[1])))

    return col_dict, count_vector_matrix