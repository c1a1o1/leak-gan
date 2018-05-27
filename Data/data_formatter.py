""" Script for extracting only the captions data from json formatting it into a
    simple txt file
    !!! Warning do not execute this script manually (A part of the data_setup.sh) !!!
"""
import json

# define the files to be processed
train_file = "intermediate/captions_train2017.json"
val_file = "intermediate/captions_val2017.json"
formatted_train_file = "captions_train.txt"
formatted_val_file = "captions_val.txt"


def format_captions_json_to_txt(json_file, out_file):
    """
    parse json file and extract only the required data into text file
    :param json_file: path to json file
    :param out_file: path to output txt file
    :return: None
    """
    # parse the input json file
    with open(json_file, 'r') as j_file:
        json_data = json.loads(j_file.read())

    # json_data is a dict with keys: dict_keys(['images', 'licenses', 'info', 'annotations'])
    captions = json_data['annotations']

    # captions is a list of dicts with keys: dict_keys(['caption', 'image_id', 'id'])
    # of which we only need the `caption` field
    # extract this data, lowercase it and write to the out_file

    with open(out_file, 'w') as o_file:
        for caption in captions:
            line = caption['caption'].strip().lower()
            if line != "":  # if line is not an empty string
                if line[-1] != '.':
                    line = line + "."  # add period if doesn't exist
                o_file.write(line + "\n")

    # provide a done message
    print("Data has been converted ... check file: " + out_file)


def main():
    """
    Main function of the script
    :return: None
    """
    format_captions_json_to_txt(train_file, formatted_train_file)
    format_captions_json_to_txt(val_file, formatted_val_file)


if __name__ == '__main__':
    main()
