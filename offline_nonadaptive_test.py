import pickle
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from itertools import groupby
from operator import itemgetter
import sys
from utils.common_utils import add_slash_to_dir, make_sure_path_exists
from utils.constants import DEFAULT_HOLDOUT_SIZE

def calculate_random_expectation(n_true, n_total, at_k):
    p = 1.0*n_true/n_total
    expectation = 0
    variance = 0
    for i in range(1,n_true+1):
        current_prob = p**i * (1-p)**(at_k-i)
        current_n = 1
        for j in range(i):
            current_n *= 1.0*(at_k - j)/(j+1)
        expectation += i*current_n*current_prob
        variance += (i**2)*current_n*current_prob
    variance -= expectation**2
    return expectation, np.sqrt(variance)

def erase_heldout(E_test, heldout_pairs):
    E_test_modified = E_test.copy()
    E_test_modified[heldout_pairs[0], heldout_pairs[1]] = 0
    E_test_modified.eliminate_zeros()
    return E_test_modified

def majority_based_questioning(E_test, Q, width=20):
    Q_new_data = []
    Q_new_rows = []
    Q_new_cols = []

    #binarising
    E_test[E_test > 0] = 1

    for i in range(Q.shape[1]):
        v = Q[:, i].flatten()
        sorting_indices = np.argsort(v)
        highest_row_indices = sorting_indices[v.size - width:v.size]
        lowest_row_indices = sorting_indices[0:width]
        col_indices = [i] * (2*width)
        row_indices = []
        row_indices.extend(highest_row_indices)
        row_indices.extend(lowest_row_indices)
        data_values = [1] * width
        data_values.extend([-1] * width)
        Q_new_data.extend(data_values)
        Q_new_rows.extend(row_indices)
        Q_new_cols.extend(col_indices)

    #calculating the new Q matrix (the question matrix)
    Q_new = csr_matrix((Q_new_data, (Q_new_rows, Q_new_cols)), shape=Q.shape)
    #calculating the answer matrix
    answers_mat = E_test.dot(Q_new)
    #binarising the answer matrix
    answers_mat[answers_mat < 0] = -1
    answers_mat[answers_mat > 0] = 1

    return answers_mat

def projection_based_questioning(E_test, Q, n_q = 50):
    answers_mat = E_test.dot(Q[:,:n_q])
    answers_mat[answers_mat < 0] = -1
    answers_mat[answers_mat > 0] = 1
    return answers_mat

def extended_proj_based_questioning(E_test, Q, actual_test_users, n_q = 50, levels=-1, col_normalise=False):
    print('Questioning simulation started...')
    if levels != -1 and levels % 2 == 0:
        levels += 1

    # This line is commented out because we are not calculating the cosine: we're calculating the dot product.
    # E_test = normalize(E_test, norm='l2', axis=1)
    if col_normalise:
        Q = Q / np.reshape(np.linalg.norm(Q, axis=0), (1, Q.shape[1]))
    answers_mat = E_test.dot(Q[:, :n_q])

    if levels == -1:
        return answers_mat
    else:
        # This array will contain the percentiles and the maximum (not including the minimum).
        # Therefore, it will contain levels elements.
        # The answer of the person for a question, where the cosine is denoted by c,
        # will be ('the number of levels_array elements c is greater than' - (levels - 1)/2) / ((levels-1)/2).
        # So e.g. if levels = 7 and something is greater than 3 elements of the list, its response is 0. If it's
        # greater than 6 of them, its response is 1. It cannot be greater than all 7
        # because the last element is the max.
        test_users_answer_submat = answers_mat[actual_test_users, :]
        levels_array = []
        print('Calculating all the percentiles...')
        for i in range(1,levels):
            levels_array.append(np.percentile(test_users_answer_submat, i*100/levels))
        levels_array.append(np.max(test_users_answer_submat))

        copied_answers_mat = answers_mat.copy()
        print('Generating the answers\' matrix using rank normalisation with ' + str(levels) + ' levels...')
        for i in range(len(levels_array)):
            if i == 0:
                answers_mat[copied_answers_mat <= levels_array[i]] = (i - (levels-1)/2) / ((levels-1)/2)
            else:
                answers_mat[(copied_answers_mat > levels_array[i-1]) & (copied_answers_mat <= levels_array[i])] = \
                    (i - (levels-1)/2) / ((levels-1)/2)

    return answers_mat

def answers_to_latent_profile(answers_mat, Q, question_indices, n_q=50):
    binarised_answers = lil_matrix((answers_mat.shape[0], answers_mat.shape[1]*2))
    top_vec_indices = np.array(list(range(0,binarised_answers.shape[1],2)))
    bottom_vec_indices = np.array(list(range(1,binarised_answers.shape[1], 2)))
    binarised_answers[:,top_vec_indices] = answers_mat[:,[x //2 for x in top_vec_indices]]
    binarised_answers[:,bottom_vec_indices] = -answers_mat[:,[x//2 for x in bottom_vec_indices]]
    binarised_answers[binarised_answers < 0] = 0
    binarised_answers = binarised_answers.tocsr()
    binarised_answers.eliminate_zeros()

    question_centroids = np.zeros((answers_mat.shape[1]*2, Q.shape[1]))
    for i in range(min([n_q, Q.shape[1], question_indices.shape[1]])):
        v = np.array(question_indices[:, i].todense()).flatten()
        # sorting_indices = np.argsort(v)
        # highest_row_indices = sorting_indices[v.size - width:v.size]
        # lowest_row_indices = sorting_indices[0:width]
        highest_row_indices = np.argwhere(v == 1)
        lowest_row_indices = np.argwhere(v == -1)
        question_centroids[2*i,:] = np.mean(Q[highest_row_indices,:], axis=0)
        question_centroids[2*i+1,:] = np.mean(Q[lowest_row_indices,:], axis=0)

    latent_profiles = binarised_answers.dot(question_centroids)
    return latent_profiles

"""
DEPRECATED

This function assumes that holdout size is equal for all given holdout users, and that the holdout is purely
nonzeros; it calculates predictions over all zeros and the heldout nonzeros.

Uses a latent space profile and dot product with Q to recommend articles.
"""

def evaluate_test_nonzero_holdout(E_test, heldout_pairs, Q, question_indices, n_q=50, at_ks=[10], user_holdout_size=20):
    E_test_modified = E_test.copy()
    E_test_modified[heldout_pairs[0], heldout_pairs[1]] = 0
    E_test_modified.eliminate_zeros()
    answers_mat = projection_based_questioning(E_test_modified, Q, n_q)
    latent_profiles = answers_to_latent_profile(answers_mat, Q, question_indices, n_q)
    test_users, test_docs = heldout_pairs
    max_at_k = max(at_ks)
    recommended_pairs = []

    for user_index in np.unique(test_users):
        print(user_index)
        user_vec = latent_profiles[user_index,:]
        nonzero_indices = set(E_test_modified[user_index,:].nonzero()[1])
        ratings = np.reshape(user_vec, (1, user_vec.size)).dot(Q.transpose()).flatten()
        rankings = np.argsort(ratings)
        ind = rankings.size - 1
        count = 0
        top_max_k_indices = []
        while count < max_at_k:
            if (rankings[ind] not in nonzero_indices):
                top_max_k_indices.append((rankings[ind], rankings.size-ind))
                count += 1
            ind -= 1
        recommended_pairs.extend([(user_index, int(E_test[user_index, x[0]]>0), x[1]) for x in top_max_k_indices])

    result_dict = {}

    for at_k in at_ks:
        current_recommended_pairs = [(x[0],x[1]) for x in recommended_pairs if x[2] <= at_k]
        true_positive_counts = [reduce(lambda x, y: (x[0], x[1] + y[1]), group) for _, group in
                                groupby(current_recommended_pairs, key=itemgetter(0))]
        micro_precisions = map(lambda x: (x[0], 1.0 * x[1] / at_k), true_positive_counts)
        micro_recalls = map(lambda x: (x[0], 1.0 * x[1] / user_holdout_size), true_positive_counts)

        result_dict[at_k] = {'mic_prec': micro_precisions, 'mic_recl': micro_recalls}

    return result_dict

"""
This function assumes that holdout size is equal for all given holdout users, and that the holdout is purely
nonzeros; it calculates predictions over all zeros and the heldout nonzeros.

Uses the document space and the ordering of articles induced by the sum of answers*topic vectors.
"""

def dotproduct_test_article_space(E_test, heldout_pairs, Q, n_q=50, levels=7, at_ks=[10],
                                  user_holdout_size=20, col_normalise=False, stratify=True):
    test_users, test_docs = heldout_pairs

    E_test_modified = erase_heldout(E_test, heldout_pairs)

    if not stratify:
        levels = -1

    answers_mat = extended_proj_based_questioning(E_test_modified, Q, np.unique(test_users), n_q=n_q, levels=levels,
                                                  col_normalise=col_normalise)

    max_at_k = max(at_ks)
    recommended_pairs = []
    if col_normalise:
        Q = Q / np.reshape(np.linalg.norm(Q, axis=0), (1, Q.shape[1]))
    user_counter = 0
    for user_index in np.unique(test_users):
        user_counter += 1
        if user_counter % 500 == 0:
            print(user_counter)

        nonzero_indices = set(E_test_modified[user_index, :].nonzero()[1])

        user_answer = answers_mat[user_index,:]
        ratings = Q[:, :n_q].dot(user_answer.reshape((user_answer.size, 1))).flatten()
        rankings = np.argsort(ratings)

        new_recommended_pairs = rankings_to_recommendation_tuples(rankings,
                                              max_at_k, user_index, E_test, nonzero_indices)
        recommended_pairs.extend(new_recommended_pairs)

    return recommendation_tuples_to_prec_recall(recommended_pairs, at_ks, user_holdout_size)

"""
Gets a ranking among articles for a particular user (where each article is represented by its index), 
the user's index, the maximum @k for which you want the precision and recall values, the E_test matrix 
(test user-doc matrix), and the nonzero indices of the desired user (user_index) in the modified E_test matrix 
(i.e. in the E_test matrix with the holdout edits erased), 
and,
returns recommendation tuples of the form:
(user_index, 1 if this recommendation was an actual holdout edit and 0 otherwise, position in recommendations' list 
for this user)

The argument 'ascending' tells us whether the ranking is ascending or descending.
"""

def rankings_to_recommendation_tuples(rankings, max_at_k, user_index, E_test, nonzero_indices, ascending=True):
    if ascending:
        ind = rankings.size - 1
    else:
        ind = 0
    count = 0
    top_max_k_indices = []
    while count < max_at_k:
        # We skip over the nonzeros of the modified E_test, because we simply do not want to predict
        # articles that we know they edited and that we have used for the question-answering simulation.
        # Therefore, we only record those articles that are zeros in the modified E_test, and then the
        # next step is to see, @k, how many of them are the ones we held out, and how many are garbage.
        if (rankings[ind] not in nonzero_indices):
            if ascending:
                top_max_k_indices.append((rankings[ind], rankings.size - ind))
            else:
                top_max_k_indices.append((rankings[ind], ind+1))
            count += 1
        if ascending:
            ind -= 1
        else:
            ind += 1
    # user_index, whether the point we recorded was a heldout edit (1) or garbage (0) and the index of the article
    # in the recommendations (i.e. the position in the list of recommendations at which it appears).
    recommended_pairs = [(user_index, int(E_test[user_index, x[0]] > 0), x[1]) for x in top_max_k_indices]
    return recommended_pairs

"""
Takes tuples of recommendations, the k's at which we want precision and recall, and user holdout size, and returns 
a dictionary containing the prec@k and recall@k values.
The recommended tuples are as follows: 
(user_index, 1 if this recommendation was an actual holdout edit and 0 otherwise, position in recommendations' list 
for this user)
"""

def recommendation_tuples_to_prec_recall(recommended_pairs, at_ks, user_holdout_size=20):
    result_dict = {}

    for at_k in at_ks:
        current_recommended_pairs = [(x[0], x[1]) for x in recommended_pairs if x[2] <= at_k]
        true_positive_counts = [reduce(lambda x, y: (x[0], x[1] + y[1]), group) for _, group in
                                groupby(current_recommended_pairs, key=itemgetter(0))]
        micro_precisions = map(lambda x: (x[0], 1.0 * x[1] / at_k), true_positive_counts)
        micro_recalls = map(lambda x: (x[0], 1.0 * x[1] / user_holdout_size), true_positive_counts)

        result_dict[at_k] = {'mic_prec': micro_precisions, 'mic_recl': micro_recalls}

    return result_dict

def save_textual_desc_prec_and_recall(at_ks, output_text, result_dict):
    for k in at_ks:
        current_mean_prec = sum([x[1] for x in result_dict[k]['mic_prec']]) / len(result_dict[k]['mic_prec'])
        current_variance_prec = sum([(x[1] - current_mean_prec) ** 2 for x in result_dict[k]['mic_prec']]) / (
                len(result_dict[k]['mic_prec']) - 1)
        current_mean_recl = sum([x[1] for x in result_dict[k]['mic_recl']]) / len(result_dict[k]['mic_recl'])
        current_variance_recl = sum([(x[1] - current_mean_recl) ** 2 for x in result_dict[k]['mic_recl']]) / (
                len(result_dict[k]['mic_recl']) - 1)
        output_text.write('Micro-avg precision at ' + str(k) + '\n')
        output_text.write(str(current_mean_prec) + ',' + str(np.sqrt(current_variance_prec)) + '\n')
        output_text.write('Micro-avg recall at ' + str(k) + '\n')
        output_text.write(str(current_mean_recl) + ',' + str(np.sqrt(current_variance_recl)) + '\n')
    output_text.close()

def main():
    usage_str = 'Takes the test user-doc matrix, heldout pairs (a tuple of two lists) and the doc latent matrix, ' \
                'calculates answers to questionnaire based on non-heldout data for test users, and evaluates on ' \
                'the heldout pairs, with precision@k and recall@k.\n' \
                'Args:\n' \
                '1. Test user-doc matrix\n' \
                '2. Heldout pairs\n' \
                '3. Doc latent matrix\n' \
                '4. Output dir\n' \
                '5. Number of questions to consider.\n' \
                '6. Whether to turn off stratification. -s to stratify, -n otherwise.'


    if (len(sys.argv) != 7):
        print(usage_str)
        return

    input_matrix_name = sys.argv[1]
    heldout_pairs_name = sys.argv[2]
    doc_latent_name = sys.argv[3]
    #question_filename = sys.argv[4]
    output_dir = sys.argv[4]
    holdout_option = '-nz'
    n_q = int(sys.argv[5])
    stratify = True
    if sys.argv[6] == '-s':
        stratify = True
    elif sys.argv[6] == '-n':
        stratify = False
    else:
        print(usage_str)
        return

    E_test = pickle.load(open(input_matrix_name, mode='rb'))
    print(E_test.shape)
    heldout_pairs = pickle.load(open(heldout_pairs_name, mode='r'))
    print('Number of test users:')
    print(np.unique(heldout_pairs[0]).size)
    doc_latent = pickle.load(open(doc_latent_name, mode='rb'))
    print(doc_latent.shape)

    if (n_q > doc_latent.shape[1]):
        n_q = doc_latent.shape[1]

    doc_latent_d = n_q
    doc_latent = doc_latent[:,:doc_latent_d]

    at_ks = [20,50,100,200,300]
    levels = 7
    result_dict = dotproduct_test_article_space(E_test, heldout_pairs, doc_latent, n_q=n_q, at_ks=at_ks,
                                                user_holdout_size=DEFAULT_HOLDOUT_SIZE, levels=levels,
                                                col_normalise=False, stratify=stratify)


    make_sure_path_exists(add_slash_to_dir(output_dir))
    output_filename = 'q_based_out_dict'+holdout_option
    if stratify:
        output_filename = output_filename + '_stratified.pkl'
    else:
        output_filename = output_filename + '_unstratified.pkl'
    pickle.dump(result_dict, open(add_slash_to_dir(output_dir) + output_filename, mode='wb'))

    output_text = open(add_slash_to_dir(output_dir) + 'prec_and_recall' + holdout_option +
                       '_' + str(levels) + '_' + str(n_q) + '_.txt', mode='w')
    save_textual_desc_prec_and_recall(at_ks, output_text, result_dict)

if __name__ == '__main__':
    main()