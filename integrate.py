import torch
import csv
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--train_setup', default='T10_disjoint')
parser.add_argument('--dataset', default="50salads")
parser.add_argument('--memory_size', default=60, type=int)
parser.add_argument('--seed', default="default", type=str)

parser.add_argument('--tas_lr', default=5e-4, type=float)
parser.add_argument('--tca_lr', default=1e-3, type=float)

parser.add_argument('--case', default="none", type=str)

args = parser.parse_args()
args.seed = 1538574472

num_tasks, scenario = int(args.train_setup.split("_")[0][1:]), args.train_setup.split("_")[1]

split_res = {}

csv_path = f"./models/seed_{args.seed}/"

min_task = []
for i in range(1, 5):
    split_res[i] = []
    csv_path_i = csv_path + f"TASlr_{args.tas_lr}"
    if args.case != 'none': csv_path_i += f"_{args.case}"
    csv_path_i += f"/{scenario}/T{num_tasks}/{args.dataset}/mem_{args.memory_size}/split_{i}"

    result_file = f"{csv_path_i}/results.csv"
    print(f"load result from {result_file}")
    
    with open(result_file, mode='r') as r_file:
        reader = csv.reader(r_file)
        data = pd.DataFrame(list(reader))
    
    for j in range(1, len(data)):
        split_res[i].append(data.loc[j].values[1:])
    min_task.append(len(split_res[i]))

min_count = min(min_task)

header = data.iloc[0].values
print("Header:", header)
results = []
for m in range(min_count):
    task_res = []
    for t in range(1, 5):
        task_res.append(split_res[t][m])
    task_mean = np.array(task_res, dtype=np.float32).mean(0)
    task_std = np.array(task_res, dtype=np.float32).std(0)
    info = f"[Task {m}] "
    for h in range(1, len(header[1:])+1):
        if header[h] == 'acc':
            task_mean[h-1] *= 100.
            task_std[h-1] *= 100.
        info += f"{header[h]}: {task_mean[h-1]:.1f} / {task_std[h-1]:.1f} \t"
    print(info)
    