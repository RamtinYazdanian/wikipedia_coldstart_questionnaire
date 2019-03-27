import errno
import os
from scipy.sparse import csr_matrix

def add_slash_to_dir(dir_name):
    if dir_name[len(dir_name) - 1] != '/':
        return dir_name + '/'
    return dir_name

"""
Inverts the given dictionary.
"""

def invert_dict(d):
    return {d[k]: k for k in d}


def intify_dict(d):
    return {int(k):int(d[k]) for k in d}

"""
Using the Wikipedia csv file, generates the id to name dictionary for the articles.
"""

def get_id_name_dict(name_filename):
    name_file = open(name_filename, mode='r')
    lines = name_file.readlines()
    lines = [line.strip().split('\t') for line in lines]
    names_dict = {int(line[0].strip()):(','.join(line[2:]).strip()).strip('\"') for line in lines}
    return names_dict

"""
Receives a list of indices, an id to index mapping in the 'col_dict', and 'inverted' which is True if col_dict
is of the form id:index and False if it's vice versa. Returns the ids of the indices_list.
"""

def get_ids_from_col_numbers(indices_list, col_dict, inverted = True):
    if (inverted):
        col_dict = invert_dict(col_dict)

    return [int(col_dict[col_number]) for col_number in list(indices_list)]


def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def binarise_sparse_scipy(m):
    a = csr_matrix(([1]*len(m.data), m.indices, m.indptr), shape=m.shape)
    return a