import os
import argparse
import logging
from pathlib import Path

"""
This script checks whether the results format for Hatespeech task is correct. 
It also provides some warnings about possible errors.

The submission of the result file for subtask 1A and 1B should be in tsv format. 
id \t label \t model


where id is the text id as given in the test file, and label is the predicted label.
For example:
101 \t Abusive \t BERT
102 \t Profane \t BERT
103 \t Religious Hate \t BERT


The submission of the result file for subtask 1C should also be in json format. 
id \t hate_type \t hate_severity \t to_whom \t model

where id is the text id as given in the test file, and label is the predicted label.
101 \t Abusive \t Mild \t Community \t BERT
102 \t Profane \t Severe \t Individual \t BERT
103 \t Religious Hate \t Mild \t Organization \t BERT
"""

logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)


def check_format(file_path):
    hate_labels = ["Abusive", "Political Hate", "Profane", "Religious Hate", "Sexism", "None"]
    to_whom_labels = ["Society", "Organization", "Community", "Individual", "None"]
    severity_labels = ["Little to None", "Mild", "Severe"]

    logging.info("Checking format of prediction file...")

    if not os.path.exists(file_path):
        logging.error("File doesn't exist: {}".format(file_path))
        return False

    with open(file_path, encoding='UTF-8') as out:
        next(out)
        file_content = out.read().strip()
        for i, line in enumerate(file_content.split('\n')):
            if 'subtask_1A' in file_path:
                doc_id, labels, model = line.strip().split('\t')

                if labels.strip() not in hate_labels:
                    logging.error("Unknown label {} in line {}".format(labels, i))
                    return False
            elif 'subtask_1B' in file_path:
                doc_id, labels, model = line.strip().split('\t')

                if labels.strip() not in to_whom_labels:
                    logging.error("Unknown label {} in line {}".format(labels, i))
                    return False
            else:
                doc_id, hate_type, severity, to_whom, model = line.strip().split('\t')

                if hate_type not in hate_labels:
                    logging.error("Unknown label {} in line {}".format(hate_type, i))
                    return False
                if severity not in severity_labels:
                    logging.error("Unknown label {} in line {}".format(severity, i))
                    return False
                if to_whom not in to_whom_labels:
                    logging.error("Unknown label {} in line {}".format(to_whom, i))
                    return False

    logging.info("File passed format checker!")

    return True


def validate_files(pred_files_path):
    logging.info("Validating if passed files exist...")

    for pred_file_path in pred_files_path:
        if not os.path.exists(pred_file_path):
            logging.error("File doesn't exist: {}".format(pred_file_path))
            return False

        # Check if the filename matches what is required by the task
        subtasks = ['1A', '1B', '1C']
        if not any(Path(pred_file_path).name.startswith('subtask_'+st_name) for st_name in subtasks):
            logging.error("The submission file must start by task name! possible prefixes: " + str(subtasks))
            return False

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_files_path", "-p", nargs='+', required=True,
                        help="The absolute path to the files you want to check.", type=str)

    args = parser.parse_args()
    pred_files_path = args.pred_files_path

    if validate_files(pred_files_path):
        for pred_file_path in pred_files_path:
            logging.info("Checking file: {}".format(pred_file_path))

            check_format(pred_file_path)
