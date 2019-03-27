import pickle
import sys
import numpy as np
from utils.common_utils import get_ids_from_col_numbers, get_id_name_dict, invert_dict, add_slash_to_dir, intify_dict
import json

def convert_to_question(sv, names_dict, col_dict, rng=20):
    if sv.ndim > 1:
        v = np.ndarray.flatten(sv)
    else:
        v = np.array(sv)
    sorting_indices = np.array(np.argsort(v))
    top_indices = sorting_indices[v.size-rng:v.size]
    top_indices = top_indices[::-1]
    top_ids = get_ids_from_col_numbers(top_indices, col_dict, inverted=False)

    bottom_indices = sorting_indices[0:rng]
    bottom_ids = get_ids_from_col_numbers(bottom_indices, col_dict, inverted=False)

    highest_scores_and_names = [(v[top_indices[i]], names_dict.get(top_ids[i])) for i in range(0,len(top_indices))]
    lowest_scores_and_names = [(v[bottom_indices[i]], names_dict.get(bottom_ids[i])) for i in range(0,len(bottom_indices))]
    
    all_ids = top_ids
    all_ids.extend(bottom_ids)

    return lowest_scores_and_names, highest_scores_and_names, all_ids

def create_questions(col_dict, input_q_matrix, names_dict, out_file_txt, out_file_json, col_normalise=False):
    output_dict = {}
    avoid_list = []
    for i in range(0, input_q_matrix.shape[1]):
        print(i)
        out_file_txt.write('Singular value number '+str(i+1)+'\n')
        lowest, highest, all_ids = convert_to_question(input_q_matrix[:, i], names_dict, col_dict)
        avoid_list.extend(all_ids)
        
        current_dict = {}
        current_dict['top'] = [x[1] for x in highest]
        current_dict['bottom'] = [x[1] for x in lowest]
        
        output_dict[str(i)] = current_dict

        out_file_txt.write('Top docs: \n')
        for item in highest:
            out_file_txt.write(str(item))
            out_file_txt.write('\n')

        out_file_txt.write('Bottom docs: \n')
        for item in lowest:
            out_file_txt.write(str(item))
            out_file_txt.write('\n')

    json.dump(output_dict, out_file_json)

    if col_normalise:
        latent_reps = input_q_matrix / np.reshape(np.linalg.norm(input_q_matrix, axis=0), (1, input_q_matrix.shape[1]))
    else:
        latent_reps = input_q_matrix

    return latent_reps, avoid_list

def main():
    usage_str = 'Shows interpretations of singular vectors or principal components. The vectors are assumed to be in ' \
                'column form, i.e. a 2d array of shape (n_dims, n_components). The output consists of the questions ' \
                'both in txt and json formats, and the column-normalised latent representations ' \
                'used for recommendation plus the list of ' \
                'article ids that should be avoided since they appear in the questions.\n' \
                '1. Name of doc names file (mapping of doc name to doc id)\n' \
                '2. Dir of id to column index mapping dict\n' \
                '3. Name of pickle file containing the Q matrix. (the outputs will be saved in the same dir)\n' \
                '4. Optional, number of questions. If not provided, generates all the questions.'
    if (len(sys.argv) < 4 or len(sys.argv) > 5):
        print(usage_str)
        return
    name_filename = sys.argv[1]
    dict_dir = sys.argv[2]
    vectors_file_name = sys.argv[3]
    n_questions = -1
    if (len(sys.argv) == 5):
        try:
            n_questions = int(sys.argv[4])
            if (n_questions < 1):
                print(usage_str)
                return
        except:
            print(usage_str)
            return

    col_dict = json.load(open(add_slash_to_dir(dict_dir) + 'col_dict.json', mode='r'))
    col_dict = intify_dict(invert_dict(col_dict))
    names_dict = get_id_name_dict(name_filename)

    Q = np.array(pickle.load(open(vectors_file_name, mode='rb')))
    if (n_questions == -1 or n_questions > Q.shape[1]):
        n_questions = Q.shape[1]
    Q = Q[:,:n_questions]

    out_file_txt = open(vectors_file_name + '_interpreted_'+str(n_questions)+'.txt', mode='w')
    out_file_json = open(vectors_file_name + '_questions_dict_'+str(n_questions)+'.json', mode='w')

    latent_reps, id_avoid_list = create_questions(col_dict, Q, names_dict, out_file_txt, out_file_json)

    out_file_txt.close()
    out_file_json.close()

    name_avoid_list = [names_dict[x] for x in id_avoid_list]

    pickle.dump(latent_reps, open(vectors_file_name+'_latent_rep_'+str(n_questions)+'.pkl', mode='wb'))
    pickle.dump(id_avoid_list, open(vectors_file_name+'_id_avoid_list_'+str(n_questions)+'.pkl', mode='wb'))
    pickle.dump(name_avoid_list, open(vectors_file_name + '_name_avoid_list_' + str(n_questions) + '.pkl', mode='wb'))


if __name__ == '__main__':
    main()
