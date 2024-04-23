import os
import re
import glob
from natsort import os_sorted


def get_matrix(dir_name: str):
    tmp_dir = dir_name if dir_name[-1] == '\\' else dir_name + '\\'
    true_dir = tmp_dir + 'true\\'
    pred_dir = tmp_dir + 'pred\\'
    list_of_true_files = os_sorted(filter(os.path.isfile, glob.glob(true_dir + '*')))
    list_of_pred_files = os_sorted(filter(os.path.isfile, glob.glob(pred_dir + '*')))




if __name__ == '__main__':
    directory = input("enter path to folder: ")
    get_matrix(directory)
