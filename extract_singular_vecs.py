import pickle
import sys

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import TruncatedSVD

from utils.common_utils import add_slash_to_dir

"""
Performs inverse row frequency weighting (using log(N_rows/1+row_freq) for each col).
"""

def inv_row_freq_weighting(X):
    row_freqs = np.array(X.sum(axis = 0)).flatten()
    n_rows = X.shape[0]
    inv_row_freq = np.log(1.0*n_rows/(1+row_freqs))
    inv_rf_diag = lil_matrix((inv_row_freq.size, inv_row_freq.size))
    inv_rf_diag.setdiag(inv_row_freq)
    return X*inv_rf_diag

"""
Calculates k top singular vectors of a sparse matrix.
The return value has shape (num_features, k) in which num_features equals X.shape[1]
"""

def get_singular_vecs(X, k):
    model = TruncatedSVD(n_components=k)
    model.fit(X)
    singular_vecs = model.components_
    return singular_vecs.transpose()

"""
Calculates k top eigenvectors of a symmetric (Hermitian actually) matrix
w has the eigenvalues and v the eigenvectors, with shape (n_features, k) where n_features = covariance_mat.shape[0]
(or [1]).
"""

def get_eigenvectors(covariance_mat, k):
    w, v = eigsh(covariance_mat, k, return_eigenvectors=True, which='LM')
    return w, v

"""
Takes an eigenvector of the reverse covariance mat (X.X^T instead of X^T.X) and transforms it into
an eigenvector of the regular covariance mat.
"""

def transform_reverse_eigenvector(X, mean_vector, eigenvec):
    transformed_eigenvec = X.transpose().dot(np.reshape(eigenvec, (eigenvec.size, 1)))
    #print(np.shape(eigenvec))
    #print(np.shape(transformed_eigenvec))
    transformed_eigenvec = transformed_eigenvec - np.reshape(mean_vector*np.sum(eigenvec),
                                                             newshape=transformed_eigenvec.shape)
    return np.ndarray.flatten(transformed_eigenvec)

"""
Calculates X.X^T (assumes rows of X to be data points) with mean-centering
"""

def calculate_reverse_cov_mat(X):
    mean_vector = X.mean(axis=0)
    mean_vector = np.reshape(mean_vector, (1, mean_vector.size))
    print('mean calculated')
    m_dot_mt = mean_vector.dot(mean_vector.transpose())
    print('m_dot_mt calculated')
    x_dot_mt = X.dot(mean_vector.transpose())
    print('x_dot_mt calculated')
    m_dot_xt = x_dot_mt.transpose()
    x_dot_xt = X.dot(X.transpose())
    print('x_dot_xt calculated')
    covariance_mat = x_dot_xt + m_dot_mt - x_dot_mt - m_dot_xt
    return mean_vector, covariance_mat

"""
Calculates X^T.X with mean-centering
"""

def calculate_cov_mat(X):
    mean_vector = X.mean(axis=0)
    mean_vector = np.reshape(mean_vector, (1, mean_vector.size))
    X = X - mean_vector
    return mean_vector, X.transpose().dot(X)

def main():
    usage_str = 'Gets a scipy sparse matrix in pickle form, calculates its k top singular vectors ' \
                '(w/o mean-centering).\n' \
                'Args:\n' \
                '1. Input file dir\n' \
                '2. Number of singular vectors desired (max 1000, default 20)\n' \
                '3. -w for inverse row freq weighting, -n for normal\n' \
                '4. List of rows to keep (-none for none)\n' \
                '5. -b to binarise the input, -nb to leave it be.\n' \
                '6. Name of the input file.\n' \
                '7. Whether or not to transpose the original matrix. -t to transpose, -n otherwise.'
    if (len(sys.argv) != 8):
        print(usage_str)
        return

    input_dir = sys.argv[1]
    k = 20
    try:
        k = int(sys.argv[2])
    except:
        k = 20

    if (k>1000 or k<0):
         k = 20
    do_weighting = False
    if (sys.argv[3] == '-w'):
        do_weighting = True
    elif (sys.argv[3] == '-n'):
        do_weighting = False
    else:
        print(usage_str)
        return
    keep_users_filename = sys.argv[4]
    binarise = False
    if (sys.argv[5] == '-b'):
        binarise = True
    elif (sys.argv[5] == '-nb'):
        binarise = False
    else:
        print(usage_str)
        return
    if (keep_users_filename != '-none'):
        keep_rows = np.array(pickle.load(open(keep_users_filename, mode='rb')))
    else:
        keep_rows = None
    input_filename = sys.argv[6]
    transpose_input = False
    if sys.argv[7] == '-t':
        transpose_input = True
    elif sys.argv[7] == '-n':
        transpose_input = False
    else:
        print(usage_str)
        return


    in_file = open(add_slash_to_dir(input_dir)+input_filename, mode='rb')
    X = pickle.load(in_file)
    if (keep_rows is not None):
        X = X[keep_rows, :]
    if transpose_input:
        X = X.transpose()
    if (binarise):
        X = csr_matrix((np.array([1]*len(X.data)), X.indices, X.indptr), shape = X.shape)
    if (do_weighting):
        X = inv_row_freq_weighting(X)

    singular_vecs = get_singular_vecs(X, k)
    f1 = open(add_slash_to_dir(input_dir) + 'v_sing_vecs.pkl', mode='wb')
    pickle.dump(singular_vecs, f1)
    f1.close()
    print('V singular vecs saved')
    f2 = open(add_slash_to_dir(input_dir) + 'utimessigma_sing_vecs.pkl', mode='wb')
    utimessigma = X.dot(singular_vecs[:,1:])
    pickle.dump(utimessigma,f2)
    f2.close()
    print('U*Sigma saved')

if __name__=='__main__':
    main()
