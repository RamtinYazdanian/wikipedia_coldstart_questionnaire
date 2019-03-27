import sys

from utils.matrix_builders import create_user_doc_count
from pyspark import SparkContext
from utils.save_load_utils import tsv_to_rdd, save_rdd_mat, save_dict, save_id_list

from utils.common_utils import add_slash_to_dir
from utils.constants import DOC_INACTIVITY_THRESH, USER_INACTIVITY_THRESH, USER_BOT_THRESH, DOC_STUB_THRESH
from utils.split_and_filter import train_test_split_ids, get_sub_rdd_by_id_list,\
    filter_user_doc_data, iterative_keeplist_filtering


def main():
    usage_str = 'Creates a user-doc count-vectorised matrix out of revision history data. Can do a train-test split. ' \
                'Has two filtering schemes, which are determined by the first argument. If first arg is -old, the ' \
                'old approach is used, which is deprecated and will be removed. If it is -new, the new approach ' \
                'is used. For -old, the args are:\n' \
                '\t1. The input file.\n' \
                '\t2. The directory for the RDD output.\n' \
                '\t3. The directory for the json dictionary and test id list (if to be saved).\n' \
                '\t4. The name of the non-admin pages file.\n' \
                '\t5. Name of bot names file to filter them out, -none for no such filtering\n' \
                '\t6. Name of list of docs to discard, -none for no such thing\n' \
                '\t7. Train/test split fraction. Must be a float in the range [0,1). If it\'s 0, no split is performed.\n' \
                '\t8. If it is -f then inactive users are filtered, settings in constants.py. -nf for no filtering.\n' \
                'For -new, the args are:\n' \
                '\t1. The input file.\n' \
                '\t2. The directory for the RDD output.\n' \
                '\t3. The directory for the json dictionary and test id list (if to be saved).\n' \
                '\t4. Name of doc keep-list. -none to keep all.\n' \
                '\t5. Name of user keep-list. -none to keep all.\n' \
                '\t6. Train/test split fraction. Must be a float in the range [0,1). If 0, no split is performed.\n' \
                'In this case, user and doc lower thresholds and the doc upper threshold are set in constants.py.'
    if (len(sys.argv) < 2):
        print(usage_str)
        return
    if (sys.argv[1] == '-old'):
        old_mode = True
        if len(sys.argv) != 10:
            print(usage_str)
            return

        filter_inactives = False

        input_filename = sys.argv[2]
        output_dir_rdd = sys.argv[3]
        output_dir_dict = sys.argv[4]
        nonadmin_pages_filename = sys.argv[5]
        bots_filename = sys.argv[6]
        discard_list_filename = sys.argv[7]
        if (bots_filename == '-none'):
            bots_filename = None
        if (discard_list_filename == '-none'):
            discard_list_filename = None
        if (nonadmin_pages_filename == '-none'):
            nonadmin_pages_filename = None
        split_frac = 0
        try:
            split_frac = float(sys.argv[8])
            if (split_frac > 1 or split_frac < 0):
                print(usage_str)
                return
        except:
            print(usage_str)
            return

        if (sys.argv[9] == '-f'):
            filter_inactives = True
        elif (sys.argv[9] == '-nf'):
            filter_inactives = False
        else:
            print(usage_str)
            return
    else:
        # TODO Add an option for getting an edit count matrix vs getting an edit size matrix.
        # Edit count option just uses the existing code, while edit size matrix uses the new code.
        old_mode=False
        if len(sys.argv) != 8:
            print(usage_str)
            return
        input_filename = sys.argv[2]
        output_dir_rdd = sys.argv[3]
        output_dir_dict = sys.argv[4]
        doc_keep_list = sys.argv[5]
        user_keep_list = sys.argv[6]
        if (doc_keep_list == '-none'):
            doc_keep_list = None
        if (user_keep_list == '-none'):
            user_keep_list = None
        split_frac = 0
        try:
            split_frac = float(sys.argv[7])
            if (split_frac > 1 or split_frac < 0):
                print(usage_str)
                return
        except:
            print(usage_str)
            return

    spark = SparkContext.getOrCreate()
    input_rdd = tsv_to_rdd(spark, input_filename)



    if (old_mode):
        filtered_rdd = filter_user_doc_data(spark, input_rdd, pages_filename=nonadmin_pages_filename,
                                        user_freq_filtration=filter_inactives, bots_filename=bots_filename,
                                        doc_discard_list_filename= discard_list_filename)
    else:
        filtered_rdd = iterative_keeplist_filtering(spark, input_rdd,
                                    user_keep_list_filename=user_keep_list, doc_keep_list_filename=doc_keep_list)
    print('**********************Filtering complete************************')
    col_dict, count_vec_matrix = create_user_doc_count(spark, filtered_rdd, removeAnonymous=True)
    print('******************Data matrix creation completed!*************************')

    train_ids_list = None
    test_ids_list = None
    if (split_frac != 0):
        train_ids_list, test_ids_list = train_test_split_ids(spark, count_vec_matrix, test_frac=split_frac, id_col=0)

    save_dict(output_dir_dict, col_dict)

    if (split_frac == 0):
        save_rdd_mat(output_dir_rdd, count_vec_matrix)
    else:
        train_count_vec = get_sub_rdd_by_id_list(spark, count_vec_matrix, train_ids_list)
        test_count_vec = get_sub_rdd_by_id_list(spark, count_vec_matrix, test_ids_list)
        save_rdd_mat(add_slash_to_dir(output_dir_rdd) + 'train', train_count_vec)
        save_rdd_mat(add_slash_to_dir(output_dir_rdd) + 'test', test_count_vec)


    if (test_ids_list is not None):
        save_id_list([int(test_id) for test_id in test_ids_list], output_dir_dict)

    n_rows = count_vec_matrix.count()
    n_cols = count_vec_matrix.first()[1].size

    info_dict = {'rows': n_rows, 'cols': n_cols, 'split fraction': split_frac,
                 'doc_filter_thresh': DOC_INACTIVITY_THRESH, 'user_inactivity_thresh': USER_INACTIVITY_THRESH,
                 'user_bot_thresh': USER_BOT_THRESH, 'doc_stub_thresh': DOC_STUB_THRESH}
    save_dict(output_dir_dict,info_dict,'info_dict.json')

    print('Total number of rows:\n'+str(n_rows)+'\nTotal number of cols:\n'+str(n_cols))

if __name__ == '__main__':
    main()