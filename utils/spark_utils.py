import numpy as np
from pyspark.mllib.linalg import SparseVector
from scipy.sparse import coo_matrix

from common_utils import invert_dict

"""
Given a count-vector matrix and a dictionary containing the labels of each row, creates a
labeled count-vector matrix of the form (label, SparseVector(...)).
"""


def wrap_label_and_matrix(row_dict, count_vector_mat):
    inv_dict = invert_dict(row_dict)
    wrapped_count_vector_mat = count_vector_mat.zipWithIndex().map(lambda x: (inv_dict[x[1]], x[0]))
    return wrapped_count_vector_mat

"""
Given a labeled count-vector matrix of the form (label, SparseVector(...)), returns a row dict mapping labels to
their row indices, and a matrix without the labels.
"""


def separate_label_and_matrix(labeled_count_vector_matrix):
    row_dict = create_row_dict(labeled_count_vector_matrix)
    return row_dict, labeled_count_vector_matrix.map(lambda x: x[1])

"""
Given a count-vector matrix, creates a ROW dictionary, mapping row labels (e.g. user_id in the user-doc case)
to row indices.
"""


def create_row_dict(labeled_count_vector_matrix):
    row_dict = labeled_count_vector_matrix.map(lambda x: x[0]).zipWithIndex().collectAsMap()
    return row_dict


"""
Receives a dense vector as input and makes it a sparse row vector.
"""


def make_sparse(vec):
    vec_size = np.shape(vec)[0]
    vec_dict = {i:val_i for i,val_i in enumerate(vec) if val_i != 0}
    return SparseVector(vec_size, vec_dict)

"""
Receives two SparseVectors and returns their sum as a new SparseVector.
"""

def add_sparse_vecs(x, y):
    indices = set(x.indices).union(set(y.indices))
    x_dict = dict(zip(x.indices, x.values))
    y_dict = dict(zip(y.indices, y.values))
    values =  {i: x_dict.get(i, 0.0) + y_dict.get(i, 0.0) for i in indices if x_dict.get(i, 0.0) + y_dict.get(i, 0.0) != 0.0}
    return SparseVector(x.size, values)

"""
Gets a sparse vector v and a scalar number or a dense vector a (could also be sparse) and calculates a*v, where
the multiplication is point-wise.
"""

def multiply_sparse_vec(sparse_vec, multiplier):
    if (not isinstance(multiplier, np.ndarray)):
        result_values = list(np.array(sparse_vec.values) * multiplier)
    else:
        multiplier = np.array(multiplier).flatten()
        assert multiplier.size == sparse_vec.size
        sparse_vec_dict = dict(zip(sparse_vec.indices, sparse_vec.values))
        result_values = [multiplier[i]*sparse_vec_dict[i] for i in sparse_vec.indices]

    result_dict = dict(zip(sparse_vec.indices, result_values))
    return SparseVector(sparse_vec.size, result_dict)

"""
Used in the multiplication of two matrices.
If we want to calculate A*B, then if we name the columns of A a_i and the rows of B b_j, then A*B = SUM(a_i.b_i)
This method assumes that x is a_i and y is b_i. Therefore, to use this method, you need to join the transpose of
A with B on row indices of B (which are column indices of A) and then map the resulting rdd using this function.
"""

def outer_product_sparse(x, y):
    assert isinstance(x, SparseVector)
    assert isinstance(y, SparseVector)
    row_size = x.size
    col_size = y.size
    new_indices = [(int(i),int(j)) for i in x.indices for j in y.indices]
    new_row_indices = [i for i, j in new_indices]
    new_col_indices = [j for i, j in new_indices]
    new_values = [x[i]*y[j] for i,j in new_indices]

    coo_result = coo_matrix((new_values, (new_row_indices, new_col_indices)), shape=(row_size,col_size))
    return coo_result.tocsr()


"""
Binarises the given matrix. The rdd is assumed to be a standard count-vector matrix,
i.e. with pairs of ids and sparse row vectors as its rows, like:
(1, SparseVector(size,dict1))
(445, SparseVector(size,dict2))
...
"""

def binarise_rdd(rdd):
    return rdd.map(lambda x: (x[0], SparseVector(x[1].size, {i: 1.0 for i in x[1].indices})))

def binarise_sparse_vec(sv):
    return SparseVector(sv.size, {i: 1.0 for i in sv.indices})

def expand_sparse_indices(x):
    vec = x[0]
    vec_dict = dict(zip(vec.indices, vec.values))
    return [(vec_dict[i], (x[1], i)) for i in vec.indices]

"""
Receives matrix rdd of form (row_id, SparseVector), or (SparseVector, row_index) if already_indexed = True.
Returns scipy csr matrix.
"""
def mat_rdd_to_csr(rdd, rdd_shape, already_indexed=False, type = 'csr'):
    if (not already_indexed):
        rdd = rdd.map(lambda x: x[1]).zipWithIndex()
    tuples_wrapped = rdd.flatMap(lambda x: expand_sparse_indices(x))
    return data_ind_to_csr(tuples_wrapped, rdd_shape, type)

"""
Receives tuple rdd of form (row_index, column_index, value).
Returns csr_matrix.
"""

def tuples_rdd_to_csr(rdd, rdd_shape, type = 'csr'):
    tuples_rdd = rdd.map(lambda x: (x[2], (x[0], x[1])))
    return data_ind_to_csr(tuples_rdd, rdd_shape, type)

def csr_to_rdd(spark, mat):
    l = [make_sparse(mat.getrow(i).todense().flatten()) for i in range(mat.shape[0])]
    return spark.parallelize(l)

def data_ind_to_csr(tuples_rdd, mat_shape, type):
    data = tuples_rdd.map(lambda x: x[0]).collect()
    i = tuples_rdd.map(lambda x: x[1][0]).collect()
    j = tuples_rdd.map(lambda x: x[1][1]).collect()
    print('***************Number of non-zeros********************')
    print(len(data))
    result_mat = coo_matrix((data, (i, j)), shape=mat_shape)
    if (type == 'csc'):
        return result_mat.tocsc()
    else:
        return result_mat.tocsr()