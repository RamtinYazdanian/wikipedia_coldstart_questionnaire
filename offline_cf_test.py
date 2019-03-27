from utils.common_utils import add_slash_to_dir, make_sure_path_exists
from offline_nonadaptive_test import rankings_to_recommendation_tuples, erase_heldout, save_textual_desc_prec_and_recall
import sys
import pickle
import numpy as np
from scipy.sparse import vstack
import implicit
from itertools import groupby
from operator import itemgetter
from functools import reduce

def py3_recoms_to_prec_recall(recommended_pairs, at_ks, user_holdout_size=20):
    result_dict = {}

    for at_k in at_ks:
        current_recommended_pairs = [(x[0], x[1]) for x in recommended_pairs if x[2] <= at_k]
        true_positive_counts = [reduce(lambda x, y: (x[0], x[1] + y[1]), group) for _, group in
                                groupby(current_recommended_pairs, key=itemgetter(0))]
        micro_precisions = [(x[0], 1.0 * x[1] / at_k) for x in true_positive_counts]
        micro_recalls = [(x[0], 1.0 * x[1] / user_holdout_size) for x in true_positive_counts]

        result_dict[at_k] = {'mic_prec': micro_precisions, 'mic_recl': micro_recalls}

    return result_dict

def main():
    usage_str = 'Performs offline testing of CF recommendations using user edit histories.\n' \
                'ATTENTION: The library "implicit" used in this script requires Python 3. Do not attempt running ' \
                'with Python 2.\n' \
                'Args:\n' \
                '1. Training user-doc matrix.\n' \
                '2. Test user-doc matrix.\n' \
                '3. Holdout pairs.\n' \
                '4. Output dir'

    if len(sys.argv) != 5:
        print(usage_str)
        return

    train_ud_filename = sys.argv[1]
    test_ud_filename = sys.argv[2]
    holdout_filename = sys.argv[3]
    output_dir = sys.argv[4]

    at_ks = [20, 50, 100, 200, 300]

    E_train = pickle.load(open(train_ud_filename, 'rb'), encoding='latin1')
    E_test = pickle.load(open(test_ud_filename, mode='rb'), encoding='latin1')
    heldout_pairs = pickle.load(open(holdout_filename, mode='rb'), encoding='latin1')

    test_users, test_docs = heldout_pairs
    E_test_modified = erase_heldout(E_test, heldout_pairs)

    print('Data loaded, starting creation of training matrix...')

    n_train_users = E_train.shape[0]
    training_mat = vstack([E_train, E_test_modified]).transpose().tocsr()

    print('Starting training...')

    model = implicit.als.AlternatingLeastSquares(factors=50)
    model.fit(training_mat)

    max_at_k = max(at_ks)
    recommended_pairs = []
    user_counter = 0

    recommendation_test_mat = training_mat.transpose().tocsr()

    print('Calculating recommendations')

    for user_index in np.unique(test_users):
        user_counter += 1
        if user_counter % 100 == 0:
            print(user_counter)

        nonzero_indices = set(E_test_modified[user_index, :].nonzero()[1])
        user_index_in_training_mat = n_train_users+user_index
        article_index_ranking = model.recommend(user_index_in_training_mat, recommendation_test_mat,
                                                N=max_at_k, filter_already_liked_items=True)
        article_index_ranking = [x[0] for x in article_index_ranking]
        new_recommended_pairs = rankings_to_recommendation_tuples(article_index_ranking,
                                                                  max_at_k, user_index, E_test, nonzero_indices,
                                                                  ascending=False)
        recommended_pairs.extend(new_recommended_pairs)

    result_dict = py3_recoms_to_prec_recall(recommended_pairs, at_ks)
    make_sure_path_exists(add_slash_to_dir(output_dir))

    output_text = open(
        add_slash_to_dir(output_dir) + 'prec_and_recall_cf.txt', mode='w')

    save_textual_desc_prec_and_recall(at_ks, output_text, result_dict)

    pickle.dump(result_dict,
                open(add_slash_to_dir(output_dir) + 'out_dict_cf.pkl', mode='wb'))


if __name__ == '__main__':
    main()