import sys
from operator import add

import numpy as np
from pyspark import SparkContext, SparkConf
from utils.save_load_utils import tsv_to_rdd, load_dict, save_dict
from utils.constants import DOC_FREQ_UPPER_REL, DOC_FREQ_LOWER_ABS
from utils.common_utils import intify_dict, add_slash_to_dir


def generate_concept_doc_dict(rdd):
    rdd = rdd.map(lambda x: (int(x[0]), int(x[1]))).distinct()
    return rdd.collectAsMap()

def main():
    usage_str = '1. Concept-doc file name, -none if there isn\'t any.\n' \
                '2. Concept-word-count file name\n' \
                '3. Output rdd dir\n' \
                '4. Dir for doc filtering dict, optional\n' \
                '5. Dir for output dict'

    if (len(sys.argv) != 6):
        print(usage_str)
        return
    concept_doc_filename = sys.argv[1]
    concept_term_filename = sys.argv[2]
    output_dir_rdd = sys.argv[3]
    filter_dict_dir = sys.argv[4]
    output_dir_dict = sys.argv[5]

    conf = SparkConf().set("spark.driver.maxResultSize", "2G").\
        set("spark.hadoop.validateOutputSpecs", "false").\
        set('spark.default.parallelism', '100')
    spark = SparkContext.getOrCreate(conf=conf)

    #Making the concept-doc map to convert concept values in the concept-doc rdd to doc-term... or not.

    c2dmap_bc = None
    if concept_doc_filename != '-none':
        rdd1 = tsv_to_rdd(spark, concept_doc_filename)
        concept_to_doc_map = generate_concept_doc_dict(rdd1)
        c2dmap_bc = spark.broadcast(concept_to_doc_map)

    #Loading the concept-term rdd and mapping the concepts to docs.

    rdd2 = tsv_to_rdd(spark, concept_term_filename)
    rdd2 = rdd2.map(lambda x: (int(x[0]), (int(x[1]), int(x[2]))))

    if c2dmap_bc is not None:
        rdd2 = rdd2.map(lambda x: (c2dmap_bc.value[x[0]], x[1]))

    doc_row_dict = None

    #Here we filter the docs if a dict of ids and indices of docs we want to keep is given as input
    # (e.g. when docs are already columns of a user-doc matrix).
    if (filter_dict_dir != '-none'):
        filtering_dict = intify_dict(load_dict(add_slash_to_dir(filter_dict_dir) + 'col_dict.json'))
        filter_dict_bc = spark.broadcast(filtering_dict)
        rdd2 = rdd2.filter(lambda x: x[0] in filter_dict_bc.value)
        rdd2 = rdd2.map(lambda x: (filter_dict_bc.value[x[0]], x[1]))
        #very important, because filtering with a dict means that we want the number of docs here
        #to be equal to number of docs in that dict (for matrix multiplication).
        docs_num = len(filtering_dict)
    else:
        doc_row_dict = rdd2.map(lambda x: x[0]).distinct().zipWithIndex().collectAsMap()
        doc_row_dict_bc = spark.broadcast(doc_row_dict)
        rdd2 = rdd2.map(lambda x: (doc_row_dict_bc.value[x[0]], x[1]))
        docs_num = len(doc_row_dict)


    #Now we calculate word frequencies. This will be used both for filtering and for calculating IDF.
    word_doc_freq = rdd2.map(lambda x: (x[1][0], 1)).reduceByKey(add).collectAsMap()
    #Filtering now
    doc_freq_bc = spark.broadcast(word_doc_freq)
    rdd2 = rdd2.filter(lambda x: doc_freq_bc.value[x[1][0]] < DOC_FREQ_UPPER_REL * docs_num and
                                 doc_freq_bc.value[x[1][0]] > DOC_FREQ_LOWER_ABS)

    #Now we calculate TF-IDF and instead of doc-word-count we have doc-word-tf_idf. Also filtering out those docs
    #with 0 tf_idf value.

    tf_idf_pairs = rdd2.map(lambda x: (x[0], x[1][0], np.log(1+x[1][1]) * np.log(docs_num / doc_freq_bc.value[x[1][0]])))
    tf_idf_pairs = tf_idf_pairs.filter(lambda x: x[2] > 0)

    #Now we want to make map the word ids of the rdd to word indices which are actually column indices for a
    #doc-term matrix.

    word_index_dict = tf_idf_pairs.map(lambda x: x[1]).distinct().zipWithIndex().collectAsMap()
    word_index_d_bc = spark.broadcast(word_index_dict)
    tf_idf_pairs = tf_idf_pairs.map(lambda x: (x[0], word_index_d_bc.value[x[1]], x[2]))

    words_num = len(word_index_dict)
    total_count = tf_idf_pairs.count()

    #count_vec_matrix = tf_idf_pairs.reduceByKey(add).map(lambda x: (x[0], SparseVector(words_num, {a[0]:a[1] for a in x[1]})))

    #Now saving.
    tf_idf_pairs.saveAsTextFile(add_slash_to_dir(output_dir_rdd))
    print('************************Saved RDD, now saving dicts*****************************')
    info_dict = {'rows': docs_num, 'cols': words_num, 'vals': total_count, 'upper_df_thresh':DOC_FREQ_UPPER_REL,
                 'lower_df_thresh':DOC_FREQ_LOWER_ABS}
    save_dict(output_dir_dict, info_dict, 'info_dict.json')
    save_dict(output_dir_dict, word_index_dict)
    if (doc_row_dict is not None):
        save_dict(output_dir_dict, doc_row_dict, 'row_dict.json')

if __name__=='__main__':
    main()
