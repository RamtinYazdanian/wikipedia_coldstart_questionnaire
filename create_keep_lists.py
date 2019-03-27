import pickle
import sys
from utils.constants import *
from utils.common_utils import add_slash_to_dir

def main():
    if (len(sys.argv) != 2):
        print('Please give the input directory containing the histogram data (the output dir of create_doc_edit_histograms.py).')
        return
    input_dir = sys.argv[1]
    doc_lengths = pickle.load(open(add_slash_to_dir(input_dir)+'doc_lengths.pkl', mode='rb'))
    doc_edit_total = pickle.load(open(add_slash_to_dir(input_dir)+'doc_edit_total.pkl', mode='rb'))
    doc_edit_distinct = pickle.load(open(add_slash_to_dir(input_dir)+'doc_edit_distinct.pkl', mode='rb'))
    user_edit_total = pickle.load(open(add_slash_to_dir(input_dir) + 'user_edit_total.pkl', mode='rb'))
    doc_keep_list = [x for x in doc_lengths if doc_lengths[x] >= DOC_STUB_THRESH]
    user_keep_list = [x for x in user_edit_total]
    print('Doc total edit average after stub removal')
    print(sum([doc_edit_total[x] for x in doc_keep_list])/len(doc_keep_list))
    print('Doc distinct edit average after stub removal')
    print(sum([doc_edit_distinct[x] for x in doc_keep_list])/len(doc_keep_list))
    print('Length of doc keep list:')
    print(len(doc_keep_list))
    print('Length of user keep list:')
    print(len(user_keep_list))
    pickle.dump(user_keep_list, open(add_slash_to_dir(input_dir)+'user_keep_list.pkl',mode='wb'))
    pickle.dump(doc_keep_list, open(add_slash_to_dir(input_dir)+'doc_keep_list.pkl', mode='wb'))

if __name__ == '__main__':
    main()