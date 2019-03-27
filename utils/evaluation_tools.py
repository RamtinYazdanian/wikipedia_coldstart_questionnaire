import numpy as np
import sys
import pickle
from sklearn.utils import resample

def intra_set_dotproduct(vecs):
    total_score = 0
    for i in range(vecs.shape[0]):
        for j in range(i+1, vecs.shape[0]):
            total_score += vecs[i,:].dot(vecs[j,:])/(np.linalg.norm(vecs[i,:]) * np.linalg.norm(vecs[j,:])+1e-200)

    return total_score / (vecs.shape[0]*(vecs.shape[0]-1)/2)

def inter_set_dotproduct(vecs1, vecs2):
    total_score = 0
    for i in range(vecs1.shape[0]):
        for j in range(0, vecs2.shape[0]):
            total_score += vecs1[i, :].dot(vecs2[j, :])/(np.linalg.norm(vecs1[i,:]) * np.linalg.norm(vecs2[j,:])+1e-200)

    return total_score / (vecs1.shape[0]*vecs2.shape[0])

def calc_cohesion_score(Q, Qbar, k, width = 20, return_list = False):
    # The top and the bottom have to be cohesive for each column of Q.
    width = int(width)
    total_score = 0
    if (k > Q.shape[1]):
        k = Q.shape[1]
    all_scores = []
    for index in range(k):
        sorted_indices = Q[:,index].argsort()
        top_indices = sorted_indices[-width:]
        bottom_indices = sorted_indices[:width]
        top_vecs = Qbar[top_indices,:]
        bottom_vecs = Qbar[bottom_indices,:]
        top_score = intra_set_dotproduct(top_vecs)
        bottom_scores = intra_set_dotproduct(bottom_vecs)
        all_scores.append((top_score+bottom_scores)/2)

    total_score = sum(all_scores)
    mean_score = total_score / len(all_scores)
    std_score = np.sqrt(sum([(x - mean_score)**2 for x in all_scores])/(len(all_scores)-1))
    if (return_list):
        return total_score, mean_score, std_score, all_scores
    return total_score, mean_score, std_score

def cohesion_confidence(cohesion_score_list, times=1000):
    cohesion_score_list = np.array(cohesion_score_list)
    mean_list = list()
    for i in range(times):
        mean_list.append(np.mean(resample(cohesion_score_list)))
    return np.mean(mean_list), np.std(mean_list), np.percentile(mean_list, 2.5), np.percentile(mean_list, 97.5)

def do_validation_split(E, user_percentage = 0.05, one_in = 5):
    # One in how many? E.g. one in 5.
    one_in = int(one_in)
    if (one_in <= 0):
        one_in = 5
    if (user_percentage == 0):
        return E.copy(), ([],[])
    if (user_percentage > 1 or user_percentage < 0):
        user_percentage = 0.05

    # We take users from the bottom of the matrix.
    starting_user = int(E.shape[0] * (1-user_percentage))

    E_val = E.copy()
    all_nonzeros = E_val.nonzero()
    # Nonzeros belonging to our set of validation users.
    starting_validation_nonzero_index = np.argmax(all_nonzeros[0] >= starting_user)
    # Now we mask one in 'one_in' interactions - these will be the ones on which we measure predictive power.
    validation_users = all_nonzeros[0][starting_validation_nonzero_index::one_in]
    validation_items = all_nonzeros[1][starting_validation_nonzero_index::one_in]
    # Setting them to zero
    E_val[validation_users,validation_items] = 0
    E_val.eliminate_zeros()

    return E_val, (validation_users,validation_items)

def prediction_error_pair_sqerr(E, P, Q, validation_pairs, kappa_, epsilon_):
    validation_users = validation_pairs[0]
    n_distinct_users = np.unique(validation_users).size
    validation_docs = validation_pairs[1]
    validation_es = np.array(E[validation_users, validation_docs]).flatten()
    P_val = P[validation_users, :]
    Q_val = Q[validation_docs, :]
    validation_PQt = np.sum(P_val*Q_val, axis=1).flatten()

    C_val = 1 + kappa_*np.log(1+validation_es/epsilon_)
    R_val = validation_es
    R_val[R_val>0] = 1

    e = R_val - validation_PQt
    return np.sum(C_val*e*e) / n_distinct_users , n_distinct_users

def main():
    usage_str = 'Calculates the cohesion score of a matrix. Args:\n' \
                '1. Name of the question matrix file.\n' \
                '2. Name of the reference matrix file (the file for calculation of cohesion)\n' \
                '3. Number of vectors to consider\n'\
                '4. Number of dimensions to consider, -1 for all.'
    if len(sys.argv) != 5:
        print(usage_str)
        return
    q_input_filename = sys.argv[1]
    Qbar_input_filename = sys.argv[2]
    try:
        n_vecs = int(sys.argv[3])
        n_dims = int(sys.argv[4])
    except:
        print(usage_str)
        return

    q = np.array(pickle.load(open(q_input_filename,mode='rb')))
    Qbar = pickle.load(open(Qbar_input_filename,mode='rb'))
    if (n_dims > -1):
        Qbar = Qbar[:,:n_dims]
    _, _, _, score_list = calc_cohesion_score(q, Qbar, n_vecs, return_list=True)
    score_mean, score_std, score_conf_lower, score_conf_upper = cohesion_confidence(score_list)
    print(score_mean, score_std, score_conf_lower, score_conf_upper)

    pickle.dump(score_list, open(q_input_filename+'_all_cohesions_'+str(n_vecs)+'.pkl',mode='wb'))

if __name__ == '__main__':
    main()
