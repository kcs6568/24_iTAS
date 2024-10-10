#!/usr/bin/python2.7
# adapted from: https://github.com/colincsl/TemporalConvolutionalNetworks/blob/master/code/metrics.py

import glob
import numpy as np
import argparse

from misc import *


def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content


def get_labels_start_end_time(frame_wise_labels, bg_class=["background"]):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i + 1)
    return labels, starts, ends


def levenstein(p, y, norm=False):
    m_row = len(p)    
    n_col = len(y)
    D = np.zeros([m_row+1, n_col+1], np.float_)
    for i in range(m_row+1):
        D[i, 0] = i
    for i in range(n_col+1):
        D[0, i] = i
    
    for j in range(1, n_col+1):
        for i in range(1, m_row+1):
            if y[j-1] == p[i-1]:
                D[i, j] = D[i-1, j-1]
            else:
                D[i, j] = min(D[i-1, j] + 1,
                              D[i, j-1] + 1,
                              D[i-1, j-1] + 1)
    
    if norm:
        score = (1 - D[-1, -1]/max(m_row, n_col)) * 100
    else:
        score = D[-1, -1]
    
    return score


def edit_score(recognized, ground_truth, norm=True, bg_class=["background"]):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


def f_score(recognized, ground_truth, overlap, bg_class=["background"]):
    p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
    y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0*intersection / union)*([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', default='train')
    parser.add_argument('--train_setup', default='T10_disjoint')

    parser.add_argument('--dataset', default="50salads")
    parser.add_argument('--split', default='1')
    parser.add_argument('--memory_size', default=60, type=int)

    parser.add_argument('--stages', default=4, type=int)

    parser.add_argument('--seed', default="default", type=str)

    parser.add_argument('--tas_lr', default=5e-4, type=float)
    parser.add_argument('--tca_lr', default=1e-3, type=float)

    parser.add_argument('--split_SIL', default=0, type=int)

    parser.add_argument('--case', default="none", type=str)

    parser.add_argument('--task', default=0, type=int)

    args = parser.parse_args()
    
    target_task = args.task
    
    if args.seed == "default":
        args.seed = 1538574472
    else:
        args.seed = int(args.seed)
    
    activity_info = open(f"./results/seed_{args.seed}/activity_order.txt", 'r').read().split("\n")[:-1]
    recog_path = f"./results/seed_{args.seed}/TASlr_{args.tas_lr}_TCAlr_{args.tca_lr}/{args.dataset}/mem_{args.memory_size}/split_{args.split}/task_{target_task}/"
    if args.case != 'none': recog_path += f"/{args.case}"
    
    ground_truth_path = f"{data_root_path}/"+args.dataset+"/groundTruth/"
    file_list = list(glob.glob(f"{recog_path}/tas/*"))
    list_of_videos = [tst.split("/")[-1] for tst in file_list]
    
    result_file = f"{recog_path}/results_tas.csv"
    recog_path = recog_path + "/tas/"    
    

    overlap = [.1, .25, .5]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

    correct = 0
    total = 0
    edit = 0

    for vid in list_of_videos:
        gt_file = ground_truth_path + vid + ".txt"
        gt_content = read_file(gt_file).split('\n')[0:-1]
        
        recog_file = recog_path + vid
        recog_content = read_file(recog_file).split('\n')[1].split()

        for i in range(len(gt_content)):
            total += 1
            if gt_content[i] == recog_content[i]:
                correct += 1
        
        edit += edit_score(recog_content, gt_content)

        for s in range(len(overlap)):
            tp1, fp1, fn1 = f_score(recog_content, gt_content, overlap[s])
            tp[s] += tp1
            fp[s] += fp1
            fn[s] += fn1
    
    all_results = {}
    print("Acc: %.4f" % (100*float(correct)/total))
    all_results.update({"Acc": round(100*float(correct)/total, 4)})
    
    print('Edit: %.4f' % ((1.0*edit)/len(list_of_videos)))
    all_results.update({"Edit": round((1.0*edit)/len(list_of_videos), 4)})
    
    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s]+fp[s])
        recall = tp[s] / float(tp[s]+fn[s])
    
        f1 = 2.0 * (precision*recall) / (precision+recall)

        f1 = np.nan_to_num(f1)*100
        print('F1@%0.2f: %.4f' % (overlap[s], f1))
        
        all_results.update({f"F1@{int(overlap[s]*100)}": round(f1, 4)})
    
    print(all_results)
    
    import csv, os
    if not os.path.exists(result_file):
        header = ["Acc", "Edit", "F1@10", "F1@25", "F1@50"]
    else:
        header = []
        
    with open(result_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        if len(header) != 0:
            writer.writerow(header)
        
        writer.writerow(list(all_results.values()))
    
if __name__ == '__main__':
    main()
