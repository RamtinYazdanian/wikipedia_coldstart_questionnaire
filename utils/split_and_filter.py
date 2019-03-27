import numpy as np
import random
from operator import add
import pickle
import re
from utils.constants import USER_ID, USER_NAME, USER_INACTIVITY_THRESH, DOC_PERCENTILE, DOC_INACTIVITY_THRESH,\
                                PAGE_ID, DOC_FILTER_METHOD, USER_FILTER_METHOD, DISCARD_LISTS, DISCARD_YEAR_LISTS, \
                                USER_BOT_THRESH

"""
Randomly divides the set of user_ids into a test_ids and a train_ids set. Both are sets, i.e. no repetition.
To be used on an rdd.
"""

def train_test_split_ids(spark, rdd, test_frac=0.1, id_col = USER_ID):
    user_ids_rdd = rdd.map(lambda x: int(x[id_col])).distinct()
    user_ids = user_ids_rdd.collect()
    test_indices = random.sample(range(len(user_ids)), int(test_frac * len(user_ids)))
    test_ids = spark.parallelize([user_ids[i] for i in test_indices])
    train_ids = user_ids_rdd.subtract(test_ids)
    return train_ids.collect(), test_ids.collect()

def split_csr_mat(sparse_mat, split_frac = 0.1):
    n_rows = sparse_mat.shape[0]
    test_indices = random.sample(range(n_rows), int(split_frac * n_rows))
    train_indices = [x for x in np.arange(n_rows) if x not in test_indices]
    train_mat = sparse_mat[train_indices, :]
    test_mat = sparse_mat[test_indices, :]
    return train_mat, test_mat, test_indices

"""
Gets a list of ids and returns only the part of the dataset with those ids. Used in train/test splitting and
filtering out the administrative pages. col_num is 0 by default, which means user_id (in all of our data, the
first column is user_id), but for filtering admin pages we need to set col_num = 1.
"""

def get_sub_rdd_by_id_list(spark, rdd, id_list, col_num=0):
    id_set = set(id_list)
    id_set_bc = spark.broadcast(id_set)
    return rdd.filter(lambda x: x[col_num] in id_set_bc.value)

"""
Takes a user-doc matrix and filters users whose activity has been below the designated threshold.
"""

def remove_inactive_users(rdd, threshold = USER_INACTIVITY_THRESH, binarise=True):
    if (binarise):
        return rdd.filter(lambda x: len(x[1].values) >= threshold)
    else:
        return rdd.filter(lambda x: sum(x[1].values) >= threshold)

"""
Takes file containing list of non-admin docs (id, size, name) and only keeps those rows of rdd whose page_id
is in that file.
"""

def remove_admin_pages(spark, rdd, pages_filename):
    name_file = open(pages_filename, mode='r')
    lines = name_file.readlines()
    valid_ids = [int(line.strip().split('\t')[0]) for line in lines]
    return get_sub_rdd_by_id_list(spark, rdd, valid_ids, col_num=1)

"""
Receives list of docs as pickled list and filters them out from rdd.
"""

def discard_by_list(spark, rdd, discard_list_filename, col_num=1):
    try:
        discard_list = pickle.load(open(discard_list_filename, mode='rb'))
    except:
        discard_list = pickle.load(open(discard_list_filename, mode='rb'), encoding='latin1')

    discard_bc = spark.broadcast(set(discard_list))
    rdd = rdd.filter(lambda x: not (x[col_num] in discard_bc.value))
    return rdd

"""
Discards articles whose names
    * start with "List of" or "list of".
    * contain a year.
"""
def discard_docs_by_names(spark, rdd, pages_filename, col_num=1):
    name_file = open(pages_filename, mode='r')
    lines = name_file.readlines()
    art_id_and_names = [(int(line.strip().split('\t')[0]), line.strip().split('\t')[2]) for line in lines]
    # If the setting for discarding list-like articles is true, then we discard the following:
    if (DISCARD_LISTS):
        list_article_ids = {x[0] for x in art_id_and_names if 'list of ' in x[1].lower() or 'lists of ' in x[1].lower()
                                                        or 'index of ' in x[1].lower()
                                                        # or 'timeline of ' in x[1].lower()
                                                        or 'rosters of ' in x[1].lower() or 'roster of ' in x[1].lower()
                                                        or 'glossary of ' in x[1].lower()
                                                        #or 'outline of ' in x[1].lower()
                                                        or 'discography' in x[1].lower()
                                                        or 'line-ups' in x[1].lower() or 'line-up' in x[1].lower()
                                                        # or 'chronology of ' in x[1].lower()
                                                        }
        print('*****************************************')
        print('*****************************************')
        print('Number of articles identified as "list-like articles"')
        print(len(list_article_ids))
        print('*****************************************')
        print('Out of')
        print(len(art_id_and_names))
        print('*****************************************')
        print('*****************************************')
        list_ids_bc = spark.broadcast(list_article_ids)
        rdd = rdd.filter(lambda x: not (x[col_num] in list_ids_bc.value))
    # If the setting for discarding yearly events is set to true, we remove everything listed below.
    # Notice that we don't detect "years", we also remove any article whose name begins with a number. Which is
    # a good idea since there are also articles like '1 (number)', etc.
    if (DISCARD_YEAR_LISTS):
        # 1953 Detroit Lions season
        year_pattern_1 = re.compile(r"\?[0-9]{1,4}.*")
        # Monthly lists
        year_pattern_2 = re.compile(r"(Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|Jun(e)?|Jul(y)?|Aug(ust)?|Sep(tember)?|Oct(ober)?|Nov(ember)?|Dec(ember)?)\s+[0-9]{1,4}")
        # China at the 2014 Olympics (or whatever)
        year_pattern_3 = re.compile(r"\s+(in|before|at the)\s+[0-9]{4}")
        # Switzerland in the Eurovision Song Contest 2014
        year_pattern_4 = re.compile(r",\s+[0-9]{1,4}")
        yearlist_ids = {x[0] for x in art_id_and_names if len(re.findall(year_pattern_1,'?'+x[1])) > 0
                                                            or len(re.findall(year_pattern_2,x[1])) > 0
                                                            or len(re.findall(year_pattern_3,x[1])) > 0
                                                            or len(re.findall(year_pattern_4, x[1])) > 0
                                                        }
        yearlists_bc = spark.broadcast(yearlist_ids)
        rdd = rdd.filter(lambda x: not (x[col_num] in yearlists_bc.value))

    return rdd

"""
Takes an rdd of (user_id, page_id) pairs, a column number (to choose whether we'll be filtering on the user_ids
or on the page_ids), and a percentile threshold. Returns only those pairs whose col_num values' repetition count
in the original rdd is above the thresh percentile.
"""

def filter_below_percentile(rdd, spark, thresh=DOC_PERCENTILE, col_num=1):
    counts_rdd = rdd.map(lambda x: (x[col_num], 1)).reduceByKey(add).sortBy(lambda x: x[1]).zipWithIndex()
    total_count = counts_rdd.count()
    keep_list = counts_rdd.filter(lambda x: x[1] >= thresh*total_count).map(lambda x:x[0][0]).collect()
    keep_list = set(keep_list)
    keep_list_bc = spark.broadcast(keep_list)
    return rdd.filter(lambda x: x[col_num] in keep_list_bc.value)

"""
Similar to the above function, but does not filter based on percentiles, but based on raw value. Those with their
count under the thresh are discarded.
"""

def filter_by_values(rdd, spark, thresh=DOC_INACTIVITY_THRESH, col_num=1, binarise = False, how='lower'):
    if (binarise):
        counts_rdd = rdd.distinct().map(lambda x: (x[col_num], 1)).reduceByKey(add)
    else:
        counts_rdd = rdd.map(lambda x: (x[col_num], 1)).reduceByKey(add)

    if (how == 'upper'):
        # Upper bounding
        keep_list = counts_rdd.filter(lambda x: x[1] <= thresh).map(lambda x: x[0]).collect()
    else:
        # Lower bounding
        keep_list = counts_rdd.filter(lambda x: x[1] >= thresh).map(lambda x: x[0]).collect()
    keep_list = set(keep_list)
    keep_list_bc = spark.broadcast(keep_list)
    return rdd.filter(lambda x: x[col_num] in keep_list_bc.value)

"""
Removes pesky bots.
"""

def remove_bots(spark, rdd, bots_filename):
    bot_names = open(bots_filename, mode='r').readlines()
    bots_set = set([x.strip() for x in bot_names])
    bots_bc = spark.broadcast(bots_set)
    return rdd.filter(lambda x: not(x[2] in bots_bc.value))

"""
The main guy. Used in creation of the histograms in create_doc_edit_histograms.
Removes (all of them are optional and depend either on flags or on values not being null):
1. Bots
2. Administrative pages
3. Users with few edits
4. Docs with few edits
5. Preset lists of users and docs
"""

def filter_user_doc_data(spark, rdd, doc_freq_filtration = True, user_freq_filtration = True,
                         admin_filtration = True, pages_filename = None, bots_filename=None, discard_by_name = True,
                         doc_discard_list_filename=None, user_discard_list_filename=None):

    rdd = rdd.map(lambda x: (int(x[USER_ID]), int(x[PAGE_ID]), x[USER_NAME]))
    #Bot removal
    if (bots_filename is not None):
        rdd = remove_bots(spark, rdd, bots_filename)
    rdd = rdd.map(lambda x: (x[0], x[1]))

    #Doc removal by names and types (i.e. removal of administrative pages).
    if (pages_filename is not None):
        if (admin_filtration):
            rdd = remove_admin_pages(spark, rdd, pages_filename)
        if discard_by_name:
            rdd = discard_docs_by_names(spark, rdd, pages_filename, col_num=1)

    #A list of docs to discard. Can be used to discard stubs, i.e. overly small docs.
    if (doc_discard_list_filename is not None):
        rdd = discard_by_list(spark, rdd, doc_discard_list_filename, col_num=1)

    # A list of users to discard. Can be used to discard bots, i.e. people with way too many edits.
    if (user_discard_list_filename is not None):
        rdd = discard_by_list(spark, rdd, user_discard_list_filename, col_num=0)

    #Doc removal by frequency
    if (doc_freq_filtration):
        if (DOC_FILTER_METHOD == 'value'):
            rdd = filter_by_values(rdd, spark, thresh=DOC_INACTIVITY_THRESH)
        elif (DOC_FILTER_METHOD == 'value_bin'):
            rdd = filter_by_values(rdd, spark, binarise=True, thresh=DOC_INACTIVITY_THRESH)
        else:
            rdd = filter_below_percentile(rdd, spark)

    #User removal by frequency
    if (user_freq_filtration):
        if (USER_FILTER_METHOD == 'value'):
            rdd = filter_by_values(rdd, spark, thresh=USER_INACTIVITY_THRESH, col_num = 0)
        elif (USER_FILTER_METHOD == 'value_bin'):
            rdd = filter_by_values(rdd, spark, binarise=True, thresh=USER_INACTIVITY_THRESH, col_num = 0)

    return rdd

"""
Keeps a list of users/docs and discards the rest.
"""

def keep_list(spark, rdd, k_list, col_num):
    k_list_bc = spark.broadcast(set(k_list))
    rdd = rdd.filter(lambda x: x[col_num] in k_list_bc.value)
    return rdd

"""
Also a main guy. This is for -new in create_user_doc_matrix, and is different from the other main guy.
Performs iterative threshold-based filtering after discarding any users and docs not in the provided "keep-lists".
"""

def iterative_keeplist_filtering(spark, rdd, user_keep_list_filename, doc_keep_list_filename, bot_thresh=USER_BOT_THRESH
                                 , user_thresh=USER_INACTIVITY_THRESH, doc_thresh=DOC_INACTIVITY_THRESH,
                                 user_mode=USER_FILTER_METHOD, doc_mode=DOC_FILTER_METHOD):

    rdd = rdd.map(lambda x: (int(x[USER_ID]), int(x[PAGE_ID])))

    #Using the "keep lists"
    if (user_keep_list_filename is not None):
        user_keep_list = pickle.load(open(user_keep_list_filename, mode='rb'))
        rdd = keep_list(spark, rdd, user_keep_list, col_num=0)
    if (doc_keep_list_filename is not None):
        doc_keep_list = pickle.load(open(doc_keep_list_filename, mode='rb'))
        rdd = keep_list(spark, rdd, doc_keep_list, col_num=1)

    #Using the upper threshold for users
    rdd = filter_by_values(rdd, spark, thresh=bot_thresh, col_num=0, binarise=True, how='upper')

    done = False
    n_users = len(user_keep_list)
    n_docs = len(doc_keep_list)
    if user_mode not in ['value', 'value_bin'] and doc_mode not in ['value', 'value_bin']:
        return rdd
    if user_thresh == 0 and doc_thresh == 0:
        return rdd

    #Performing the iterative frequency-based filtering
    while not done:
        if (doc_thresh > 0):
            # Doc removal by frequency
            if (doc_mode == 'value'):
                rdd = filter_by_values(rdd, spark, thresh=doc_thresh, col_num=1)
            elif (doc_mode == 'value_bin'):
                rdd = filter_by_values(rdd, spark, binarise=True, thresh=doc_thresh, col_num=1)
            n_docs_new = rdd.map(lambda x: x[1]).distinct().count()
            if (n_docs_new == n_docs or user_thresh == 0):
                done = True
                break
            else:
                n_docs = n_docs_new
        if (user_thresh > 0):
            # User removal by frequency
            if (user_mode == 'value'):
                rdd = filter_by_values(rdd, spark, thresh=user_thresh, col_num=0)
            elif (user_mode == 'value_bin'):
                rdd = filter_by_values(rdd, spark, binarise=True, thresh=user_thresh, col_num=0)
            n_users_new = rdd.map(lambda x: x[0]).distinct().count()
            if (n_users_new == n_users or doc_thresh == 0):
                done = True
                break
            else:
                n_users = n_users_new

    print('Final doc and user count:')
    print(n_docs, n_users)

    return rdd