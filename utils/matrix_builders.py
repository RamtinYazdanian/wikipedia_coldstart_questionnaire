from count_vectorise import count_vectorise_simple

"""
This function creates a user-doc edit matrix, without taking whether an edit is major or minor into account.

Return values:
doc_dict: It's a dictionary, mapping page_id to column.
user_doc_matrix: An RDD. Each row is a tuple consisting of the user_id and a SparseVector object.
"""


def create_user_doc_count(spark, edit_data, removeAnonymous = False):
    user_doc_edit_data = edit_data.map(lambda x: (int(x[0]), int(x[1])))
    print('-------columns selected-----------')
    if (removeAnonymous == True):
        user_doc_edit_data = user_doc_edit_data.filter(lambda x: x[0] != '0')
    print('-------------anonymous removed----------------')
    user_doc_edit_data = user_doc_edit_data.groupByKey().mapValues(list)
    print('--------------------count-vectorise starting------------------------')
    doc_dict, user_doc_matrix = count_vectorise_simple(spark, user_doc_edit_data)
    return doc_dict, user_doc_matrix

"""
Like the previous function, but creates the transpose of that matrix from scratch (without actually
creating that one and transposing it).
"""

def create_user_doc_transpose(spark, edit_data, removeAnonymous = False):
    doc_user_edit_data = edit_data.map(lambda x: (int(x[1]), int(x[0])))
    print('-------columns selected-----------')
    if (removeAnonymous == True):
        doc_user_edit_data = doc_user_edit_data.filter(lambda x: x[1] != '0')
    print('-------------anonymous removed----------------')
    doc_user_edit_data = doc_user_edit_data.groupByKey().mapValues(list)
    print('--------------------count-vectorise starting------------------------')
    user_dict, doc_user_matrix = count_vectorise_simple(spark, doc_user_edit_data)
    return user_dict, doc_user_matrix
