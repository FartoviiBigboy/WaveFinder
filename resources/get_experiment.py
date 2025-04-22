import math
import os
import re
import glob
from concurrent.futures.thread import ThreadPoolExecutor
import cProfile

import numpy as np
import obspy
from natsort import os_sorted
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt

from PredictionFilter import PredictionFilter
from Seismogram import Seismogram

def compute_score_pre_p(args):
    map_pred_p, map_true_p = args
    answer = np.zeros((101, 101, 4), dtype=np.int32)

    for i in range(101):
        for j in range(101):
            tmp_list_pred_up_p = {key for (key, value) in map_pred_p.items() if
                                value[0] >= (float(i) / 100.0) and value[1] <= (float(j) / 100.0)}
            tmp_list_pred_down_p = {key for (key, value) in map_pred_p.items()} - tmp_list_pred_up_p
            tmp_list_true_p = set(map_true_p)
            answer[i, j, 0] = len(tmp_list_pred_up_p & tmp_list_true_p)
            answer[i, j, 1] = len(tmp_list_pred_up_p - tmp_list_true_p)
            answer[i, j, 2] = len(tmp_list_pred_down_p & tmp_list_true_p)
            answer[i, j, 3] = len(tmp_list_pred_down_p - tmp_list_true_p)
    return answer

def compute_score_after_s(args):
    map_pred_p, map_true_p, map_pred_s, map_true_s, list_pred_preds, seismogram, location = args
    answer = np.zeros((101, 101, 4), dtype=np.int32)

    tmp_list_pred_up_p = {key for (key, value) in map_pred_p.items() if
                          value[0] >= (float(location[0]) / 100.0) and value[1] <= (float(location[1]) / 100.0)}

    tmp_list_pred_up_corrected = sorted([int(x / PredictionFilter.DELTA_X) for x in tmp_list_pred_up_p])
    corrected_s_prediction = PredictionFilter.s_wave_correction(seismogram,
                                                                tmp_list_pred_up_corrected,
                                                                list_pred_preds)[:, 1]

    for i in range(101):
        for j in range(101):
            tmp_list_pred_up_s = {key for (key, value) in map_pred_s.items() if
                                  corrected_s_prediction[int(key / PredictionFilter.DELTA_X)] >=
                                  (float(i) / 100.0) and value[1] <= (float(j) / 100.0)}

            tmp_list_pred_down_s = {key for (key, value) in map_pred_s.items()} - tmp_list_pred_up_s
            tmp_list_true_s = set(map_true_s)

            answer[i, j, 0] = len(tmp_list_pred_up_s & tmp_list_true_s)
            answer[i, j, 1] = len(tmp_list_pred_up_s - tmp_list_true_s)
            answer[i, j, 2] = len(tmp_list_pred_down_s & tmp_list_true_s)
            answer[i, j, 3] = len(tmp_list_pred_down_s - tmp_list_true_s)

    print("exit_s")
    return answer


def compute_score_angled(args):
    map_pred_p, map_true_p, map_pred_s, map_true_s, list_pred_preds, seismogram = args
    answer = np.zeros((101, 101, 2, 4), dtype=np.int32)

    prev_map = []
    prev_s_pred = []

    for i in range(101):
        for j in range(101):
            tmp_list_pred_up_p = {key for (key, value) in map_pred_p.items() if
                                value[0] >= (float(i) / 100.0) and value[1] <= (float(j) / 100.0)}
            tmp_list_pred_down_p = {key for (key, value) in map_pred_p.items()} - tmp_list_pred_up_p
            tmp_list_true_p = set(map_true_p)
            answer[i, j, 0, 0] = len(tmp_list_pred_up_p & tmp_list_true_p)
            answer[i, j, 0, 1] = len(tmp_list_pred_up_p - tmp_list_true_p)
            answer[i, j, 0, 2] = len(tmp_list_pred_down_p & tmp_list_true_p)
            answer[i, j, 0, 3] = len(tmp_list_pred_down_p - tmp_list_true_p)

            tmp_list_pred_up_s = {}
            tmp_list_pred_down_s = {}
            tmp_list_true_s = set(map_true_s)

            if prev_map == tmp_list_pred_up_p:
                tmp_list_pred_up_s = {key for (key, value) in map_pred_s.items() if
                                      prev_s_pred[int(key / PredictionFilter.DELTA_X)] >=
                                      (float(i) / 100.0) and value[1] <= (float(j) / 100.0)}

                tmp_list_pred_down_s = {key for (key, value) in map_pred_s.items()} - tmp_list_pred_up_s


            else:
                tmp_list_pred_up_corrected = sorted([int(x / PredictionFilter.DELTA_X) for x in tmp_list_pred_up_p])
                corrected_s_prediction = PredictionFilter.s_wave_correction(seismogram,
                                                                            tmp_list_pred_up_corrected,
                                                                            list_pred_preds)[:, 1]

                tmp_list_pred_up_s = {key for (key, value) in map_pred_s.items() if
                                    corrected_s_prediction[int(key / PredictionFilter.DELTA_X)] >=
                                    (float(i) / 100.0) and value[1] <= (float(j) / 100.0)}

                tmp_list_pred_down_s = {key for (key, value) in map_pred_s.items()} - tmp_list_pred_up_s

                prev_s_pred = corrected_s_prediction
                prev_map = tmp_list_pred_up_p

            answer[i, j, 1, 0] = len(tmp_list_pred_up_s & tmp_list_true_s)
            answer[i, j, 1, 1] = len(tmp_list_pred_up_s - tmp_list_true_s)
            answer[i, j, 1, 2] = len(tmp_list_pred_down_s & tmp_list_true_s)
            answer[i, j, 1, 3] = len(tmp_list_pred_down_s - tmp_list_true_s)

    print("exit")
    return answer

def get_matrix(dir_name: str):
    tmp_dir = dir_name if dir_name[-1] == '\\' else dir_name + '\\'
    true_dir = tmp_dir + 'true_40\\'
    pred_dir = tmp_dir + 'pred_40_0.5_ex\\'
    seis_dir = tmp_dir + 'records\\'
    seis_filter = 0.5, 'high'
    list_of_true_files = os_sorted(filter(os.path.isfile, glob.glob(true_dir + '*')))
    list_of_pred_files = os_sorted(filter(os.path.isfile, glob.glob(pred_dir + '*')))
    list_of_records = os_sorted(filter(os.path.isfile, glob.glob(seis_dir + '*')))
    list_of_map_true_p = []
    list_of_map_pred_p = []
    list_of_map_true_s = []
    list_of_map_pred_s = []
    list_of_lists_pred_preds = []
    list_of_seismograms = []

    for i in range(len(list_of_records)):
        st = obspy.read(list_of_records[i])
        sorted_list = sorted(st, key=lambda x: (x.stats.station, x.stats.channel))
        sorted_list[::3], sorted_list[1::3] = sorted_list[1::3], sorted_list[::3]
        seis = Seismogram(sorted_list[0:0 + 3], list_of_records[i])
        seis.apply_filter(2, seis_filter[0], seis_filter[1])
        list_of_seismograms.append(seis)

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
            flag = 0
            tmp_map_pred_p = {}
            tmp_map_pred_s = {}
            tmp_list_pred_preds = []
            lines = file.readlines()
            for line in lines:
                if line.rstrip('\n') == 'P':
                    continue
                elif line.rstrip('\n') == 'S':
                    flag = 1
                    continue
                elif line.rstrip('\n') == 'raw':
                    flag = 2
                    continue
                if flag == 0:
                    tmp_map_pred_p[int(line.split(' ')[0])] = [float(line.split(' ')[1]),
                                                               float((line.split(' ')[2]).rstrip('\n'))]
                elif flag == 1:
                    tmp_map_pred_s[int(line.split(' ')[0])] = [float(line.split(' ')[1]),
                                                               float((line.split(' ')[2]).rstrip('\n'))]
                elif flag == 2:
                    tmp_list_pred_preds.append((float(line.split(' ')[0]), float(line.split(' ')[1]), float((line.split(' ')[2]).rstrip('\n'))))
        list_of_map_pred_p.append(tmp_map_pred_p)
        list_of_map_pred_s.append(tmp_map_pred_s)
        list_of_lists_pred_preds.append(np.array(tmp_list_pred_preds))

    args_list_pre_p = list(zip(list_of_map_pred_p, list_of_map_true_p))
    with ProcessPoolExecutor(os.cpu_count()) as executor:
        results_pre_p = executor.map(compute_score_pre_p, args_list_pre_p)

    results_pre_p = np.array(list(results_pre_p))
    results_pre_p = results_pre_p.sum(axis=0)

    mcc_score_p = np.zeros((101, 101))
    for i in range(101):
        for j in range(101):
            if not (int(results_pre_p[i, j, 0] + results_pre_p[i, j, 1])
                    * int(results_pre_p[i, j, 0] + results_pre_p[i, j, 2])
                    * int(results_pre_p[i, j, 3] + results_pre_p[i, j, 1])
                    * int(results_pre_p[i, j, 3] + results_pre_p[i, j, 2]) == 0):
                mcc_score_p[i, j] = ((int(results_pre_p[i, j, 0]) * int(results_pre_p[i, j, 3])
                                     - int(results_pre_p[i, j, 1]) * int(results_pre_p[i, j, 2]))
                                     / math.sqrt(
                            int(results_pre_p[i, j, 0] + results_pre_p[i, j, 1])
                            * int(results_pre_p[i, j, 0] + results_pre_p[i, j, 2])
                            * int(results_pre_p[i, j, 3] + results_pre_p[i, j, 1])
                            * int(results_pre_p[i, j, 3] + results_pre_p[i, j, 2])))

    max_elem = np.where(mcc_score_p == mcc_score_p.max())
    max_elem = (max(max_elem[0]), min(max_elem[1]))
    print(max_elem)
    duplicated_max_elem = [max_elem] * len(list_of_map_pred_p)

    args_list_after_s =  list(zip(list_of_map_pred_p, list_of_map_true_p, list_of_map_pred_s, list_of_map_true_s, list_of_lists_pred_preds, list_of_seismograms, duplicated_max_elem))

    with ProcessPoolExecutor(os.cpu_count()) as executor:
        results_after_s = executor.map(compute_score_after_s, args_list_after_s)

    results_after_s = np.array(list(results_after_s))
    print(results_after_s.shape)
    results_after_s = results_after_s.sum(axis=0)

    mcc_score_s = np.zeros((101, 101))
    for i in range(101):
        for j in range(101):
            if not (int(results_after_s[i, j, 0] + results_after_s[i, j, 1])
                    * int(results_after_s[i, j, 0] + results_after_s[i, j, 2])
                    * int(results_after_s[i, j, 3] + results_after_s[i, j, 1])
                    * int(results_after_s[i, j, 3] + results_after_s[i, j, 2]) == 0):
                mcc_score_s[i, j] = ((int(results_after_s[i, j, 0]) * int(results_after_s[i, j, 3])
                                      - int(results_after_s[i, j, 1]) * int(results_after_s[i, j, 2]))
                                     / math.sqrt(
                            int(results_after_s[i, j, 0] + results_after_s[i, j, 1])
                            * int(results_after_s[i, j, 0] + results_after_s[i, j, 2])
                            * int(results_after_s[i, j, 3] + results_after_s[i, j, 1])
                            * int(results_after_s[i, j, 3] + results_after_s[i, j, 2])))

    # args_list = list(zip(list_of_map_pred_p, list_of_map_true_p, list_of_map_pred_s, list_of_map_true_s, list_of_lists_pred_preds, list_of_seismograms))
    #
    # with ProcessPoolExecutor(os.cpu_count()) as executor:
    #     results = executor.map(compute_score_angled, args_list)
    #
    # results = np.array(list(results))
    # print(results.shape)
    # results = results.sum(axis=0)
    # print(results.shape)
    #
    # mcc_score_p = np.zeros((101, 101))
    # mcc_score_s = np.zeros((101, 101))
    # for i in range(101):
    #     for j in range(101):
    #         if not (int(results[i, j, 0, 0] + results[i, j, 0, 1])
    #                 * int(results[i, j, 0, 0] + results[i, j, 0, 2])
    #                 * int(results[i, j, 0, 3] + results[i, j, 0, 1])
    #                 * int(results[i, j, 0, 3] + results[i, j, 0, 2]) == 0):
    #             mcc_score_p[i, j] = ((int(results[i, j, 0, 0]) * int(results[i, j, 0, 3])
    #                                  - int(results[i, j, 0, 1]) * int(results[i, j, 0, 2]))
    #                                  / math.sqrt(
    #                         int(results[i, j, 0, 0] + results[i, j, 0, 1])
    #                         * int(results[i, j, 0, 0] + results[i, j, 0, 2])
    #                         * int(results[i, j, 0, 3] + results[i, j, 0, 1])
    #                         * int(results[i, j, 0, 3] + results[i, j, 0, 2])))
    #
    #         if not (int(results[i, j, 1, 0] + results[i, j, 1, 1]) * int(results[i, j, 1, 0] + results[i, j, 1, 2]) * int(
    #                 results[i, j, 1, 3] + results[i, j, 1, 1]) * int(results[i, j, 1, 3] + results[i, j, 1, 2]) == 0):
    #             mcc_score_s[i, j] = (int(results[i, j, 1, 0]) * int(results[i, j, 1, 3]) - int(results[i, j, 1, 1]) * int(results[
    #                 i, j, 1, 2])) / math.sqrt(
    #                 int(results[i, j, 1, 0] + results[i, j, 1, 1]) * int(results[i, j, 1, 0] + results[i, j, 1, 2]) * int(
    #                             results[i, j, 1, 3] + results[i, j, 1, 1]) * int(
    #                             results[i, j, 1, 3] + results[i, j, 1, 2]))

    # args_list_p = [(i, j, list_of_map_pred_p, list_of_map_true_p, list_of_map_pred_s, list_of_map_true_s, list_of_lists_pred_preds, list_of_seismograms) for i in range(101) for j in range(101)]
    # # args_list_s = [(i, j, list_of_map_pred_s, list_of_map_true_s) for i in range(101) for j in range(101)]
    #
    # with ThreadPoolExecutor(os.cpu_count()) as executor:
    #     results = executor.map(compute_score, args_list_p)
    #     # results_s = executor.map(compute_score, args_list_s)
    #
    # f1_p = np.zeros((101, 101))
    # g_mean_p = np.zeros((101, 101))
    # p_4_p = np.zeros((101, 101))
    # mcc_score_p = np.zeros((101, 101))
    # for i, j, f1_score, g_mean, p_4, mcc_score, _, _, _, _ in results:
    #     f1_p[i, j] = f1_score
    #     g_mean_p[i, j] = g_mean
    #     p_4_p[i, j] = p_4
    #     mcc_score_p[i, j] = mcc_score
    #
    # f1_s = np.zeros((101, 101))
    # g_mean_s = np.zeros((101, 101))
    # p_4_s = np.zeros((101, 101))
    # mcc_score_s = np.zeros((101, 101))
    # for i, j, _, _, _, _, f1_score, g_mean, p_4, mcc_score in results:
    #     f1_s[i, j] = f1_score
    #     g_mean_s[i, j] = g_mean
    #     p_4_s[i, j] = p_4
    #     mcc_score_s[i, j] = mcc_score
    #
    #
    plt.imshow(mcc_score_p, cmap='coolwarm')
    plt.colorbar()
    plt.savefig('mcc_score_p.png', dpi=700)
    plt.clf()

    plt.imshow(mcc_score_s, cmap='coolwarm')
    plt.colorbar()
    plt.savefig('mcc_score_s.png', dpi=700)
    plt.clf()


if __name__ == '__main__':
    directory = input("enter path to folder: ")
    get_matrix(directory)
