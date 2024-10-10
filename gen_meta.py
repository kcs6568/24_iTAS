'''
    coffee
    orange juice
    chocolate milk
    tea
    bowl of cereals
    fried eggs
    pancakes
    fruit salad
    sandwich
    scrambled eggs 
'''

import os
import glob
import numpy as np
import torch

from misc import *

# dataset = "breakfast"

# activities = [
#     "coffee",
#     "orange juice",
#     "chocolate milk",
#     "tea",
#     "bowl of cereals",
#     "fried eggs",
#     "pancakes",
#     "fruit salad",
#     "sandwich",
#     "scrambled eggs "
# ]

# feat_path = "/root/data/breakfast/features"
# gt_path = "/root/data/breakfast/groundTruth"

# # for act in activities:

# file_act = []
# files = glob.glob("/root/data/breakfast/groundTruth/*")


# act_indices = {}
# for i, f in enumerate(files):
#     act = f.split("/")[-1].split("_")[-1].split(".")[0]
#     file_act.append(act)
    
#     if act not in act_indices:
#         act_indices.update({act: [i]})
#     else:
#         act_indices[act].append(i)

# activities, count_info = np.unique(np.array(file_act), return_counts=True)

# path_per_act = {act: [] for act in activities}
# for act in activities:
#     for idx in act_indices[act]:
#         path_per_act[act].append(files[idx])

# result_path = f"/root/src/iTAS/metadata/{dataset}/"
# os.makedirs(result_path, exist_ok=True)

# with open(result_path+"classname.txt", 'w') as f:
#     for a in activities:
#         f.write(a+"\n")

# action_counts = 0
# for act in activities:
#     all_gts = []
    
#     with open(result_path+act+".txt", 'w') as f:
#         for i, file in enumerate(path_per_act[act]):
#             f.write(file.split("/")[-1]+"\n")
#             gt_file = open(file, 'r')
#             gt = gt_file.read().split('\n')[:-1]
#             all_gts.extend(gt)
    
#     actions = np.unique(np.array(all_gts))
#     background_idx = np.where(actions == "SIL")
#     actions = np.delete(actions, background_idx)
    
#     with open(result_path+act+"_mapping.txt", 'w') as map_f:
#         map_f.write("0 SIL"+"\n")
        
#         for i, action in enumerate(actions):
#             map_f.write(f"{i+1} {action}"+"\n")



def generate_label(task_name):
    task_setting, dataset, task_num = task_name.split('-')
    mapping_file = f"./data/{dataset}/mapping.txt"

    file_ptr = open(mapping_file, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])

    eval(f'{task_setting}_label_mapping')(task_setting, dataset, task_num, actions_dict)
    generate_new_groundTruth(task_setting, dataset, task_num)


def disjoint_label_mapping(task_setting, dataset, task_num):
    
    task_sched = dataset + '-' + task_num
    mapping = []

    for i in range(len(activity_inc_tasks[dataset][f"{task_num}_task"])):
        file_list = []
        for act in activity_inc_tasks[dataset][f"{task_num}_task"][i]:
            file_list.extend(glob.glob(f'{data_root_path}/{dataset}/groundTruth/*{act}*'))
        
        label_mapping = []
        for filename in file_list:
            with open(filename, 'r') as f:
                tmp = f.readlines()
                tmp = list(set(tmp))
                label_mapping.extend(tmp)
        
        label_mapping = sorted(list(set(label_mapping)))
  
        label_mapping = [x[:-1] + f'_{i}' for x in label_mapping]
        print(label_mapping)

        mapping.extend(label_mapping)
    
    if not os.path.exists(f'./metadata/{dataset}/new_mapping/{task_setting}'):
            os.makedirs(f'./metadata/{dataset}/new_mapping/{task_setting}')
    
    with open(f'./metadata/{dataset}/new_mapping/{task_setting}/{task_num}-task_mapping.txt', 'w') as f:
        for i, lbl in enumerate(mapping):
            f.write(f'{i} {lbl}\n')


def blurry_label_mapping(task_setting, dataset, task_num, actions_dict):

    task_sched = dataset + '-' + task_num
    
    mappings = []
    for i in range(len(activity_inc_tasks[task_sched])):
        file_list = []
        for act in activity_inc_tasks[task_sched][i]:
            file_list.extend(glob.glob(f'data/breakfast/groundTruth/*{act}*'))
    
        label_mapping = []
        for filename in file_list:
            with open(filename, 'r') as f:
                tmp = f.readlines()
                tmp = list(set(tmp))
                label_mapping.extend(tmp)
            
        label_mapping = sorted(list(set(label_mapping)))
        # print(label_mapping)
        mapping = []
        for label in label_mapping:
            mapping.append(actions_dict[label[:-1]])
        
        mappings.append(mapping)
        # print(len(mapping), mapping)
    
    if not os.path.exists(f'./metadata/{dataset}/new_mapping/{task_setting}'):
            os.makedirs(f'./metadata/{dataset}/new_mapping/{task_setting}')
    
    with open(f'./metadata/{dataset}/{task_setting}/{task_num}-task_mapping.txt', 'w') as f:
        for i, lbl in enumerate(mappings):
            f.write(f'{lbl}\n')


def generate_new_groundTruth(task_setting, dataset, task_num):
    if not os.path.exists(f'metadata/{dataset}/groundTruth/{task_setting}/{task_num}-task/'):
        os.makedirs(f'metadata/{dataset}/groundTruth/{task_setting}/{task_num}-task/')
    
    for i in range(len(activity_inc_tasks[dataset][f"{task_num}_task"])):
        for act in activity_inc_tasks[dataset][f"{task_num}_task"][i]:
            file_list = (glob.glob(f'{data_root_path}/{dataset}/groundTruth/*{act}*'))

            for filename in file_list:
                with open(filename, 'r') as f:
                    tmp = f.readlines()
                
                savename = f'metadata/{dataset}/groundTruth/{task_setting}/{task_num}-task/{filename.split("/")[-1]}'
                with open(savename, 'w') as f:
                    for action in tmp:
                        f.write(f'{action[:-1]}_{i}\n')

generate_new_groundTruth("disjoint", "breakfast", "5")
disjoint_label_mapping("disjoint", "breakfast", "5")