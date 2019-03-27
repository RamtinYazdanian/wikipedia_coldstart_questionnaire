import pickle
import json
import sys
import numpy as np
import random
from scipy.sparse import csr_matrix
from utils.common_utils import add_slash_to_dir, make_sure_path_exists
from utils.evaluation_tools import do_validation_split, calc_cohesion_score, prediction_error_pair_sqerr

GAMMA_DECREASE_STEP = 500000
BATCH_SIZE = 20000

def calc_error(user_doc_edit, user_latent, doc_latent, doc_original_latent, kappa_, epsilon_, alpha_, lambda_):
    user_sample = random.sample(range(user_latent.shape[0]), k=int(user_latent.shape[0]/50))
    doc_sample = random.sample(range(doc_latent.shape[0]), k=int(doc_latent.shape[0]/1000))
    error1 = alpha_ * np.linalg.norm(user_latent)**2
    error2 = lambda_ * np.linalg.norm(doc_latent - doc_original_latent) ** 2
    ud_sampled = np.array(user_doc_edit[user_sample][:, doc_sample].todense())
    r_sampled = ud_sampled.copy()
    r_sampled[r_sampled > 0] = 1
    c_sampled = ud_sampled
    c_sampled = np.array(1+kappa_*+np.log(1+c_sampled/epsilon_))
    term_1 = np.array(r_sampled - user_latent[user_sample,:].dot(doc_latent[doc_sample,:].transpose()))
    term_1 = term_1*term_1*c_sampled
    error3 = np.sum(term_1)
    return error1,error2,error3,error1+error2+error3

def main():
    usage_str = 'This script receives a user-doc edit matrix (non-binary) and a doc-term latent representation matrix. ' \
                'It computes two matrices, a latent user matrix and a latent doc matrix, both in the same latent ' \
                'space as the input doc-term matrix.\n' \
                'Running modes are -b for batch and -i for individual. ' \
                'If first arg is -i, then the rest of the args are:\n' \
                '1. Dir and name of the user-doc matrix.\n' \
                '2. Dir and name of the doc-term latent matrix.\n' \
                '3. Output directory.\n' \
                '4. Number of latent dimensions.\n' \
                '5. Name of file containing list of user indices to keep. Use -none if you don\'t want any such filtering.\n' \
                '6. User matrix init mode: -p for (0,1), -s for (-1,1) (both will then have their rows normalised)\n' \
                '7. -f for full training, -v for validation set separation (saves a file for the errors too)\n' \
                '8. alpha and lambda and theta in the format alpha,lambda,theta, e.g. 1,1e-3,1e-4\n' \
                '\nIf the first arg is -b, you should give the address of a file that contains the arguments ' \
                'listed above (each in one line).'
    if (len(sys.argv) < 2):
        print(usage_str)
        return
    mode_arg = sys.argv[1]
    if (mode_arg == '-i'):
        if (len(sys.argv) != 10):
            print(usage_str)
            return
        input_user_doc = sys.argv[2]
        input_doc_term = sys.argv[3]
        output_dir = sys.argv[4]
        n_latent = sys.argv[5]
        user_filter_filename = sys.argv[6]
        symmetry_arg = sys.argv[7]
        validation_arg = sys.argv[8]
        alphalambda = sys.argv[9]
    elif (mode_arg == '-b'):
        if (len(sys.argv) != 3):
            print(usage_str)
            return
        args_filename = sys.argv[2]
        args_file = open(args_filename, mode='r')
        arg_contents = args_file.readlines()
        arg_contents = [x.strip() for x in arg_contents]
        arg_contents = [x for x in arg_contents if len(x) > 0]
        if (len(arg_contents) != 8):
            print(usage_str)
            return
        input_user_doc = arg_contents[0]
        input_doc_term = arg_contents[1]
        output_dir = arg_contents[2]
        n_latent = arg_contents[3]
        user_filter_filename = arg_contents[4]
        symmetry_arg = arg_contents[5]
        validation_arg = arg_contents[6]
        alphalambda = arg_contents[7]
    else:
        print(usage_str)
        return

    try:
        n_latent = int(n_latent)
    except:
        print(usage_str)
        return

    user_init_symmetric = False
    if (symmetry_arg == '-p'):
        user_init_symmetric = False
    elif (symmetry_arg == '-s'):
        user_init_symmetric = True
    else:
        print(usage_str)
        return

    do_validation = False
    if (validation_arg == '-f'):
        do_validation = False
    elif (validation_arg == '-v'):
        do_validation = True
    else:
        print(usage_str)
        return


    split_alphalambda = alphalambda.strip('(').strip(')').split(',')
    gamma_ = -1
    if (len(split_alphalambda) < 3 or len(split_alphalambda) > 4):
        print(usage_str)
        return
    try:
        alpha_ = float(split_alphalambda[0])
        lambda_ = float(split_alphalambda[1])
        theta_ = float(split_alphalambda[2])
        if (len(split_alphalambda) == 4):
            gamma_ = float(split_alphalambda[3])
    except:
        print(usage_str)
        return
    # Loading the input matrices

    print('Loading...')
    ud_in_file = open(input_user_doc, mode='rb')
    user_doc_sparse_mat_original = csr_matrix(pickle.load(ud_in_file))
    ud_in_file.close()

    if (user_filter_filename != '-none'):
        user_filter_list = pickle.load(open(user_filter_filename, mode='rb'))
        user_doc_sparse_mat_original = user_doc_sparse_mat_original[user_filter_list, :]

    validation_pairs = None
    if (do_validation):
        user_doc_sparse_mat, validation_pairs = do_validation_split(user_doc_sparse_mat_original)
    else:
        user_doc_sparse_mat = user_doc_sparse_mat_original

    print('User-doc matrix loaded')
    dt_in_file = open(input_doc_term, mode='rb')
    doc_original_latent = np.array(pickle.load(dt_in_file))
    doc_original_latent -= np.mean(doc_original_latent,axis=0)
    #doc_original_latent = doc_original_latent / np.linalg.norm(doc_original_latent, axis=0)
    dt_in_file.close()
    print('Loading completed')

    user_doc_nonzero_indices = user_doc_sparse_mat.nonzero()
    n_nonzeros = len(user_doc_nonzero_indices[0])

    if n_latent < doc_original_latent.shape[1] and n_latent > 0:
        doc_original_latent = doc_original_latent[:,0:n_latent]
    else:
        n_latent = doc_original_latent.shape[1]

    n_users = user_doc_sparse_mat.shape[0]
    n_docs = user_doc_sparse_mat.shape[1]

    doc_original_latent /= (np.linalg.norm(doc_original_latent, axis=1).reshape((doc_original_latent.shape[0],1))+1e-60)

    print('Number of users: '+str(n_users))
    print('Number of docs: ' + str(n_docs))
    print('Number of latent dimensions: ' + str(n_latent))

    # Initialising
    print('Initialising')
    if (user_init_symmetric):
        user_latent = np.random.rand(n_users, n_latent) * 2 - 1
    else:
        user_latent = np.random.rand(n_users, n_latent)
    #user_latent /= np.linalg.norm(user_latent)
    #user_latent *= np.sqrt(np.sum(user_doc_sparse_mat.data*user_doc_sparse_mat.data))
    user_latent -= np.mean(user_latent, axis=0)
    user_latent /= (np.linalg.norm(user_latent, axis = 1).reshape((user_latent.shape[0], 1))+1e-60)
    #user_latent = user_doc_sparse_mat.dot(doc_original_latent) / (1+np.array(user_doc_sparse_mat.sum(axis=1)).reshape(user_doc_sparse_mat.shape[0],1))
    #user_latent /= ((np.linalg.norm(user_latent, axis = 1)*np.linalg.norm(user_latent, axis = 1)).reshape((user_latent.shape[0], 1))+1e-60)
    #user_latent *= 50
    #user_latent /= np.linalg.norm(user_latent,axis=0)
    doc_latent = doc_original_latent.copy()
    #doc_latent = doc_latent / np.linalg.norm(doc_latent, axis=0)

    # kappa_ and epsilon_ and gamma_ and n_iter are read from json; but alpha_ and lambda_ are given as inputs.

    params_dict = json.load(open('minibatch_settings.json', mode='r'))
    kappa_ = params_dict['kappa_']
    epsilon_ = params_dict['epsilon_']
    if gamma_ == -1:
        gamma_ = params_dict['gamma_']
    n_iter = params_dict['n_iter']
    zeta_ = params_dict['zeta_']
    #theta_ = params_dict['theta_']

    # These are the old values we used to use.
    # kappa_ = 10
    # epsilon_ = 20
    # gamma_ = 5e-2
    # n_iter = 200000
    # alpha_ = 1
    # lambda_ = 1e-3
    # theta_ = 1e-4

    errors_list = []
    print('Initialisation complete.')

    i = 0
    while i<n_iter:
        user_old = user_latent.copy()
        doc_old = doc_latent.copy()
        step_size = gamma_ / (1 + int(i / GAMMA_DECREASE_STEP))

        rand_choices = random.sample(range(0,n_nonzeros), int(0.9*BATCH_SIZE))
        user_rand_indices = [user_doc_nonzero_indices[0][rand_index] for rand_index in rand_choices]
        doc_rand_indices = [user_doc_nonzero_indices[1][rand_index] for rand_index in rand_choices]
        user_rand_indices.extend([random.randint(0, n_users-1) for j in range(0, BATCH_SIZE-len(rand_choices))])
        doc_rand_indices.extend([random.randint(0, n_docs-1) for j in range(0, BATCH_SIZE-len(rand_choices))])

        for rand_index in range(0, len(user_rand_indices)):
            user_index = user_rand_indices[rand_index]
            doc_index = doc_rand_indices[rand_index]
            e_ui = user_doc_sparse_mat[user_index, doc_index]
            r_ui = int(e_ui > 0)
            c_ui = 1 + kappa_ * np.log(1 + e_ui / epsilon_)
            coef1 = -2 * c_ui * (r_ui - np.dot(user_old[user_index,:], doc_old[doc_index,:]))
            user_latent[user_index, :] -= step_size * (coef1 * doc_old[doc_index,:])
            doc_latent[doc_index, :] -= step_size * (coef1 * user_old[user_index,:])

        uri_set = set(user_rand_indices)
        dri_set = set(doc_rand_indices)
        for user_index in uri_set:
            user_latent[user_index, :] -= 2 * step_size * alpha_ * user_old[user_index, :]
        for doc_index in dri_set:
            doc_latent[doc_index, :] -= step_size * (2 * lambda_ * (doc_old[doc_index,:] - doc_original_latent[doc_index, :])
                                        + zeta_*(1.0*(doc_old[doc_index,:]>0).astype(int) - 1.0*(doc_old[doc_index,:]<0).astype(int)))

        qtq_minus_diag = doc_old.transpose().dot(doc_old)
        qtq_minus_diag -= np.diag(np.diag(qtq_minus_diag))
        doc_latent -= 4 * theta_ * step_size * doc_old.dot(qtq_minus_diag)

        i += BATCH_SIZE
        print(i,alpha_,lambda_,theta_,gamma_,zeta_)
        #do_gd_step(user_doc_sparse_mat, user_latent, doc_latent, doc_original_latent, user_index,doc_index, i,
        #           kappa_, epsilon_, alpha_, lambda_, gamma_, theta_)

        if (i % (20*BATCH_SIZE) == 0):
            print('Errors:')
            current_error = calc_error(user_doc_sparse_mat, user_latent, doc_latent, doc_original_latent,
                                       kappa_, epsilon_, alpha_, lambda_)
            print(current_error)
            #errors_list.append(current_error)
            print('----------------')

    print('Saving')

    make_sure_path_exists(add_slash_to_dir(output_dir))
    if (not do_validation or do_validation):
        f1 = open(add_slash_to_dir(output_dir) + 'user_latent.pickle', mode='wb')
        pickle.dump(user_latent, f1)
        f1.close()

        f2 = open(add_slash_to_dir(output_dir) + 'doc_latent.pickle', mode='wb')
        pickle.dump(doc_latent, f2)
        f2.close()

    f3 = open(add_slash_to_dir(output_dir) + 'params.json', mode='w')
    json.dump({'alpha_': alpha_, 'lambda_': lambda_, 'gamma_': gamma_,
               'n_iter': n_iter, 'kappa_': kappa_, 'epsilon_': epsilon_, 'theta_':theta_, 'zeta_':zeta_, 'BATCH_SIZE':BATCH_SIZE}, f3)
    f3.close()

    if (do_validation):
        k_topics = 50
        print('Calculating validation errors:')
        pred_error, n_distinct_val_users = prediction_error_pair_sqerr(user_doc_sparse_mat_original, user_latent,
                                                                       doc_latent, validation_pairs, kappa_, epsilon_)
        #The maximum cohesion score ever possible is 2*k_topics (possible if all top and bottom scores come out as 1)
        cohesion_score, _, _ = calc_cohesion_score(doc_latent, doc_original_latent, k=k_topics)
        error_dict = {'prediction_error': pred_error, 'cohesion_score': cohesion_score,
                      'n_validation_users': n_distinct_val_users, 'k_topics': k_topics,
                      'n_validation_nonzeros': len(validation_pairs[0])}
        json.dump(error_dict, open(add_slash_to_dir(output_dir)+'errors.json', mode='w'))
        print('Prediction error:')
        print(pred_error)
        print('Cohesion score:')
        print(cohesion_score)

if __name__ == '__main__':
    main()
