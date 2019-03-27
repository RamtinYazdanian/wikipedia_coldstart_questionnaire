from pyspark import SparkConf, SparkContext
from utils.split_and_filter import filter_user_doc_data
import sys
import pickle
from utils.save_load_utils import tsv_to_rdd
from pairs_tf_idf import generate_concept_doc_dict
from utils.common_utils import add_slash_to_dir
from operator import add

def main():
    usage_str = 'This script produces doc and user edit counts (total and distinct) and doc sizes, using the ' \
                'following steps:\n' \
                '\t* Removing bots.\n' \
                '\t* Removing administrative pages.\n' \
                '\t* Calculating the doc lengths by counting the number of their words.' \
                '\t* Calculating total and distinct edit counts for users and docs.' \
                'Args:\n' \
                '1. The input revision history file.\n' \
                '2. The directory for the output files (all are dicts).\n' \
                '3. The name of the non-admin pages file.\n' \
                '4. Name of bot names file to filter them out.\n' \
                '5. Concept-doc file name, -none if nonexistent.\n' \
                '6. Concept-word-count file name\n' \
                '7. Whether to filter out by name or not. -f filters, -n does not.'
    if (len(sys.argv) != 8):
        print(usage_str)
        return
    input_filename = sys.argv[1]
    output_dir = sys.argv[2]
    nonadmin_pages_filename = sys.argv[3]
    bots_filename = sys.argv[4]
    concept_doc_filename = sys.argv[5]
    concept_term_filename = sys.argv[6]
    filter_by_name = False
    if (sys.argv[7]=='-f'):
        filter_by_name = True
    elif (sys.argv[7] == '-n'):
        filter_by_name = False
    else:
        print(usage_str)
        return
    conf = SparkConf().set("spark.driver.maxResultSize", "10G").set('spark.default.parallelism', '100')
    spark = SparkContext.getOrCreate(conf=conf)
    input_rdd = tsv_to_rdd(spark, input_filename)

    filtered_rdd = filter_user_doc_data(spark, input_rdd, pages_filename=nonadmin_pages_filename, admin_filtration=True,
                                        bots_filename=bots_filename, doc_discard_list_filename=None, user_discard_list_filename=None,
                                        user_freq_filtration=False, doc_freq_filtration=False, discard_by_name=filter_by_name)
    remaining_docs = filtered_rdd.map(lambda x: (x[1],1)).distinct().collectAsMap()

    c2dmap_bc = None
    if concept_doc_filename != '-none':
        rdd1 = tsv_to_rdd(spark, concept_doc_filename)
        concept_to_doc_map = generate_concept_doc_dict(rdd1)
        c2dmap_bc = spark.broadcast(concept_to_doc_map)

    remaining_docs_bc = spark.broadcast(remaining_docs)
    rdd2 = tsv_to_rdd(spark, concept_term_filename)
    rdd2 = rdd2.map(lambda x: (int(x[0]), int(x[2])))

    if c2dmap_bc is not None:
        rdd2 = rdd2.map(lambda x: (c2dmap_bc.value[x[0]], x[1]))

    rdd2 = rdd2.filter(lambda x: x[0] in remaining_docs_bc.value)

    doc_lengths = rdd2.reduceByKey(add).collectAsMap()
    doc_edit_total = filtered_rdd.map(lambda x: (x[1], 1)).reduceByKey(add).collectAsMap()
    doc_edit_distinct = filtered_rdd.distinct().map(lambda x: (x[1], 1)).reduceByKey(add).collectAsMap()
    user_edit_total = filtered_rdd.map(lambda x: (x[0], 1)).reduceByKey(add).collectAsMap()
    user_edit_distinct = filtered_rdd.distinct().map(lambda x: (x[0], 1)).reduceByKey(add).collectAsMap()

    print('Doc total edit average')
    print(sum([doc_edit_total[x] for x in doc_edit_total])/len(doc_edit_total))
    print('Doc distinct edit average')
    print(sum([doc_edit_distinct[x] for x in doc_edit_distinct]) / len(doc_edit_distinct))
    print('User total edit average')
    print(sum([user_edit_total[x] for x in user_edit_total]) / len(user_edit_total))
    print('User distinct edit average')
    print(sum([user_edit_distinct[x] for x in user_edit_distinct]) / len(user_edit_distinct))
    print('Doc length average')
    print(sum([doc_lengths[x] for x in doc_lengths])/len(doc_lengths))

    pickle.dump(doc_lengths, open(add_slash_to_dir(output_dir)+'doc_lengths.pkl', mode='wb'))
    pickle.dump(doc_edit_total, open(add_slash_to_dir(output_dir) + 'doc_edit_total.pkl', mode='wb'))
    pickle.dump(doc_edit_distinct, open(add_slash_to_dir(output_dir) + 'doc_edit_distinct.pkl', mode='wb'))
    pickle.dump(user_edit_total, open(add_slash_to_dir(output_dir) + 'user_edit_total.pkl', mode='wb'))
    pickle.dump(user_edit_distinct, open(add_slash_to_dir(output_dir) + 'user_edit_distinct.pkl', mode='wb'))


if __name__ == '__main__':
    main()
