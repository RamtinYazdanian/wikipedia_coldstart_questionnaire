from utils.common_utils import add_slash_to_dir, make_sure_path_exists, invert_dict
from offline_nonadaptive_test import rankings_to_recommendation_tuples, recommendation_tuples_to_prec_recall, \
    erase_heldout, save_textual_desc_prec_and_recall
import sys
import pickle
import json
import numpy as np

def main():
    usage_str = 'Calculates editing or viewing popularity-based recommendations for the existing users (offline users).' \
                ' Has separate modes for edit-pop and view-pop.\n' \
                'Args:\n' \
                '1. -edit for edit-pop recoms, -view for view-pop recoms.\n' \
                '* For edit-pop:\n' \
                '\t2. User-doc matrix filename for the calculation of most popular articles editing-wise\n' \
                '* For view-pop:\n' \
                '\t2. The JSON file containing the most viewed articles (in some period of time)\n' \
                '\t3. The file containing the mapping from article ids to article names\n' \
                '\t4. The mapping from article ids to article indices.\n' \
                '* For both:' \
                '\t3 (or 5). Test user-doc matrix file\n' \
                '\t4 (or 6). Holdout indices file\n' \
                '\t5 (or 7). Output dir\n'

    if len(sys.argv) < 2:
        print(usage_str)
        return

    oper_mode = sys.argv[1]
    if oper_mode not in ['-edit', '-view']:
        print(usage_str)
        return

    if oper_mode == '-view':
        if len(sys.argv) != 8:
            print(usage_str)
            return
        article_views_file = sys.argv[2]
        docid_docname_file = sys.argv[3]
        docid_index_file = sys.argv[4]
        input_matrix_name = sys.argv[5]
        holdout_indices_file = sys.argv[6]
        output_dir = sys.argv[7]

        article_ranking = json.load(open(article_views_file, 'r'))
        docid_docname_dict = json.load(open(docid_docname_file, 'r'))
        docname_docid_dict = invert_dict(docid_docname_dict)
        docid_docindex_dict = json.load(open(docid_index_file, 'r'))

        article_ranking = [str(x['article'].encode('utf8')).replace('_', ' ')
                           for x in article_ranking['items'][0]['articles']]
        article_index_ranking = []
        for article_name in article_ranking:
            if article_name in docname_docid_dict and docname_docid_dict[article_name] in docid_docindex_dict:
                article_index_ranking.append(docid_docindex_dict[docname_docid_dict[article_name]])
            else:
                print(article_name)

        print(len(article_index_ranking))
        ascending = False

    else:
        if len(sys.argv) != 6:
            print(usage_str)
            return

        train_user_doc_matrix_file = sys.argv[2]
        input_matrix_name = sys.argv[3]
        holdout_indices_file = sys.argv[4]
        output_dir = sys.argv[5]

        train_ud_mat = pickle.load(open(train_user_doc_matrix_file, 'rb'))
        article_edit_pops = np.array(train_ud_mat.sum(axis=0)).flatten()
        article_index_ranking = np.argsort(article_edit_pops)
        ascending = True

#    at_ks = [20, 50, 100, 200, 300, 400, 500, 600, 700, 800]
    at_ks = [20, 50, 100, 200, 300]

    E_test = pickle.load(open(input_matrix_name, mode='rb'))
    heldout_pairs = pickle.load(open(holdout_indices_file, mode='rb'))

    test_users, test_docs = heldout_pairs
    E_test_modified = erase_heldout(E_test, heldout_pairs)
    max_at_k = max(at_ks)
    recommended_pairs = []
    user_counter = 0
    for user_index in np.unique(test_users):
        user_counter += 1
        if user_counter % 500 == 0:
            print(user_counter)

        nonzero_indices = set(E_test_modified[user_index, :].nonzero()[1])

        new_recommended_pairs = rankings_to_recommendation_tuples(article_index_ranking,
                                              max_at_k, user_index, E_test, nonzero_indices, ascending=ascending)
        recommended_pairs.extend(new_recommended_pairs)

    result_dict = recommendation_tuples_to_prec_recall(recommended_pairs, at_ks)

    make_sure_path_exists(add_slash_to_dir(output_dir))
    pickle.dump(result_dict, open(add_slash_to_dir(output_dir) + 'out_dict_' + oper_mode[1:] + '_pop' + '.pkl', mode='wb'))

    output_text = open(
        add_slash_to_dir(output_dir) + 'prec_and_recall' + oper_mode[1:] + '_pop' + '_.txt', mode='w')
    save_textual_desc_prec_and_recall(at_ks, output_text, result_dict)


if __name__ == '__main__':
    main()
