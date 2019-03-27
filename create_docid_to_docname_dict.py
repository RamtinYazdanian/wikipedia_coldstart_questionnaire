from utils.common_utils import get_id_name_dict, make_sure_path_exists, add_slash_to_dir
import sys
import json

def main():
    usage_str = 'Gets the name of the .csv file containing names and ids, and creates and saves the doc id to ' \
                'doc name dictionary.\n' \
                'Args:\n' \
                '1. Input file name.\n' \
                '2. Output dir.'

    if len(sys.argv) != 3:
        print(usage_str)
        return

    input_filename = sys.argv[1]
    output_dir = sys.argv[2]

    dict_result = get_id_name_dict(input_filename)
    print('Dict generated!')
    make_sure_path_exists(add_slash_to_dir(output_dir))

    with open(add_slash_to_dir(output_dir)+'docid_docname.json', 'w') as f:
        json.dump(dict_result, f)

if __name__ == '__main__':
    main()