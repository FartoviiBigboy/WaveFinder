import math
import os
import re
import glob

import numpy as np
from natsort import os_sorted
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt


def compute_score(args):
    i, j, list_of_map_pred_value, list_of_map_true_value = args
    TP, FP, FN, TN = 0, 0, 0, 0
    f1_score, g_mean, p_4, mcc_score = 0, 0, 0, 0

    for k in range(len(list_of_map_pred_value)):
        tmp_list_pred_up = {key for (key, value) in list_of_map_pred_value[k].items() if
                            value[0] >= (float(i) / 100.0) and value[1] <= (float(j) / 100.0)}
        tmp_list_pred_down = {key for (key, value) in list_of_map_pred_value[k].items()} - tmp_list_pred_up

        tmp_list_true = set(list_of_map_true_value[k])
        TP += len(tmp_list_pred_up & tmp_list_true)
        FP += len(tmp_list_pred_up - tmp_list_true)
        FN += len(tmp_list_pred_down & tmp_list_true)
        TN += len(tmp_list_pred_down - tmp_list_true)

    if not ((TP + FP) == 0 or (TP + FN) == 0):
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        beta = 1
        f1_score = (1 + beta ** 2) * precision * recall / (precision * beta ** 2 + recall)

    if not ((TP + FN) == 0 or (TN + FP) == 0):
        g_mean = math.sqrt(TP / (TP + FN)) * math.sqrt(TN / (TN + FP))

    if not ((4 * TP * TN + (TP + TN) * (FP + FN)) == 0):
        p_4 = 4 * TP * TN / (4 * TP * TN + (TP + TN) * (FP + FN))

    if not ((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN) == 0):
        mcc_score = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    return (i, j, f1_score, g_mean, p_4, mcc_score)


def get_matrix(dir_name: str):
    tmp_dir = dir_name if dir_name[-1] == '\\' else dir_name + '\\'
    true_dir = tmp_dir + 'true_40\\'
    pred_dir = tmp_dir + 'pred_40_notall\\'
    list_of_true_files = os_sorted(filter(os.path.isfile, glob.glob(true_dir + '*')))
    list_of_pred_files = os_sorted(filter(os.path.isfile, glob.glob(pred_dir + '*')))
    list_of_map_true_p = []
    list_of_map_pred_p = []
    list_of_map_true_s = []
    list_of_map_pred_s = []

    for i in range(len(list_of_true_files)):
        with open(list_of_true_files[i], 'r') as file:
            flag = True
            tmp_map_true_p = []
            tmp_map_true_s = []
            lines = file.readlines()
            for line in lines:
                if line.rstrip('\n') == 'P':
                    continue
                elif line.rstrip('\n') == 'S':
                    flag = False
                    continue
                if flag:
                    tmp_map_true_p.append(int(line.rstrip('\n')))
                else:
                    tmp_map_true_s.append(int(line.rstrip('\n')))
        list_of_map_true_p.append(tmp_map_true_p)
        list_of_map_true_s.append(tmp_map_true_s)

    for i in range(len(list_of_pred_files)):
        with open(list_of_pred_files[i], 'r') as file:
            flag = True
            tmp_map_pred_p = {}
            tmp_map_pred_s = {}
            lines = file.readlines()
            for line in lines:
                if line.rstrip('\n') == 'P':
                    continue
                elif line.rstrip('\n') == 'S':
                    flag = False
                    continue
                if flag:
                    tmp_map_pred_p[int(line.split(' ')[0])] = [float(line.split(' ')[1]),
                                                               float((line.split(' ')[2]).rstrip('\n'))]
                else:
                    tmp_map_pred_s[int(line.split(' ')[0])] = [float(line.split(' ')[1]),
                                                               float((line.split(' ')[2]).rstrip('\n'))]
        list_of_map_pred_p.append(tmp_map_pred_p)
        list_of_map_pred_s.append(tmp_map_pred_s)

    args_list_p = [(i, j, list_of_map_pred_p, list_of_map_true_p) for i in range(101) for j in range(101)]
    args_list_s = [(i, j, list_of_map_pred_s, list_of_map_true_s) for i in range(101) for j in range(101)]

    with ProcessPoolExecutor() as executor:
        results_p = executor.map(compute_score, args_list_p)
        results_s = executor.map(compute_score, args_list_s)

    f1_p = np.zeros((101, 101))
    g_mean_p = np.zeros((101, 101))
    p_4_p = np.zeros((101, 101))
    mcc_score_p = np.zeros((101, 101))
    for i, j, f1_score, g_mean, p_4, mcc_score in results_p:
        f1_p[i, j] = f1_score
        g_mean_p[i, j] = g_mean
        p_4_p[i, j] = p_4
        mcc_score_p[i, j] = mcc_score

    f1_s = np.zeros((101, 101))
    g_mean_s = np.zeros((101, 101))
    p_4_s = np.zeros((101, 101))
    mcc_score_s = np.zeros((101, 101))
    for i, j, f1_score, g_mean, p_4, mcc_score in results_s:
        f1_s[i, j] = f1_score
        g_mean_s[i, j] = g_mean
        p_4_s[i, j] = p_4
        mcc_score_s[i, j] = mcc_score

    # plt.imshow(f1_p, cmap='coolwarm')
    # plt.colorbar()
    # plt.savefig('f1_p.png', dpi=700)
    # plt.clf()
    # plt.imshow(g_mean_p, cmap='coolwarm')
    # plt.colorbar()
    # plt.savefig('g_mean_p.png', dpi=700)
    # plt.clf()
    # plt.imshow(p_4_p, cmap='coolwarm')
    # plt.colorbar()
    # plt.savefig('p_4_p.png', dpi=700)
    # plt.clf()
    # plt.imshow(mcc_score_p, cmap='coolwarm')
    # plt.colorbar()
    # plt.savefig('mcc_score_p.png', dpi=700)
    # plt.clf()
    #
    # plt.imshow(f1_s, cmap='coolwarm')
    # plt.colorbar()
    # plt.savefig('f1_s.png', dpi=700)
    # plt.clf()
    # plt.imshow(g_mean_s, cmap='coolwarm')
    # plt.colorbar()
    # plt.savefig('g_mean_s.png', dpi=700)
    # plt.clf()
    # plt.imshow(p_4_s, cmap='coolwarm')
    # plt.colorbar()
    # plt.savefig('p_4_s.png', dpi=700)
    # plt.clf()
    # plt.imshow(mcc_score_s, cmap='coolwarm')
    # plt.colorbar()
    # plt.savefig('mcc_score_s.png', dpi=700)
    # plt.clf()
    # plt.show()


if __name__ == '__main__':
    directory = input("enter path to folder: ")
    get_matrix(directory)
