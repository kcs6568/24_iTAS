#!/usr/bin/python2.7
import sys
import time
from datetime import timedelta
from loguru import logger
from copy import deepcopy
import torch
from model import Trainer
from batch_gen import BatchGenerator, BatchGenerator_Test, TCA_BatchGenerator
import os
import argparse
import random
import numpy as np

from misc import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--action', default='train')
parser.add_argument('--mode', default='segm', choices=['segm', 'gen', 'gen_sample', 'sample'])
parser.add_argument('--train_setup', default='T10_disjoint')

parser.add_argument('--dataset', default="50salads")
parser.add_argument('--split', default='1')
parser.add_argument('--memory_size', default=60, type=int)

parser.add_argument('--stages', default=4, type=int)

parser.add_argument('--seed', default="default", type=str)

parser.add_argument('--tas_lr', default=5e-4, type=float)

parser.add_argument('--tca_bs', default=256, type=int)
parser.add_argument('--tca_lr', default=1e-3, type=float)

parser.add_argument('--case', default="none", type=str)

parser.add_argument('--task', default=0, type=int)

parser.add_argument('--test_type', default='best')

args = parser.parse_args()

if args.seed == 'default':
    args.seed = 1538574472
else:
    args.seed = int(args.seed)

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.deterministic = True

num_stages = args.stages
num_layers = 10
num_f_maps = 64
features_dim = 2048
bz = 1
num_epochs = 50
lr = args.tas_lr
num_tasks, scenario = int(args.train_setup.split("_")[0][1:]), args.train_setup.split("_")[1]

if args.dataset == 'breakfast':
    class_per_task = 10 // num_tasks
    tca_epochs = 2500
    tca_lr = args.tca_lr
    num_activity = 10
    num_cross_valid = 4
    
elif args.data == 'yti':
    class_per_task = 5 // num_tasks
    tca_epochs = 250
    tca_lr = args.tca_lr
    num_activity = 5
    num_cross_valid = 5
    args.data = "YTI"
    
sample_rate = 1

target_task = args.task
activities = [activity_inc_tasks[args.dataset][f"{num_tasks}_task"][i] for i in range(len(activity_inc_tasks[args.dataset][f"{num_tasks}_task"]))]
print("Origin ordering:", activities)
target_activity = activities[target_task]
seen_activities = activities[:target_task]

old_model_dir = f"./models/seed_{args.seed}/TASlr_{lr}"
model_dir = f"./models/seed_{args.seed}/TASlr_{lr}"

replay_dir = f"./replay/seed_{args.seed}/TCAlr_{tca_lr}"
tca_model_dir = f"./generator/seed_{args.seed}/TCAlr_{tca_lr}"

if args.case != 'none':
    old_model_dir += f"_{args.case}"
    model_dir += f"_{args.case}"
    
old_model_dir += f"/{scenario}/T{num_tasks}/{args.dataset}/mem_{args.memory_size}/split_{args.split}/task_{target_task-1}/"
model_dir += f"/{scenario}/T{num_tasks}/{args.dataset}/mem_{args.memory_size}/split_{args.split}/"
tas_model_dir = model_dir + f"task_{target_task}/tas/"

replay_dir += f"/{scenario}/T{num_tasks}/{args.dataset}/split_{args.split}"
tca_model_dir += f"/{scenario}/T{num_tasks}/breakfast/split_{args.split}/task_{target_task}"

features_path = f"{data_root_path}/"+args.dataset+"/features/"
gt_path = f"./metadata/{args.dataset}/groundTruth/{scenario}/{num_tasks}-task/"
meta_path = f"./metadata/{args.dataset}/"

if args.action == 'train':
    if args.mode == 'segm':
        logger.add(f"{tas_model_dir}/" + "{time}.log")
        
        if not os.path.exists(tas_model_dir):
            os.makedirs(tas_model_dir)
            
    elif args.mode == 'gen':
        logger.add(f"{tca_model_dir}/" + "{time}.log")
        if not os.path.exists(tca_model_dir):
            os.makedirs(tca_model_dir)
    
elif args.action in ['predict', 'test']:
    logger.add(f"{tas_model_dir}/" + "{time}_Test.log")
    print(f"{tas_model_dir}/" + "{time}_Test.log")
    
elif args.mode == 'sample':
    replay_dir += f"/task_{target_task}"
    if not os.path.exists(replay_dir):
        os.makedirs(replay_dir)    

mapping_file = f"./metadata/{args.dataset}/new_mapping/{scenario}/{num_tasks}-task_mapping.txt"
file_ptr = open(mapping_file, 'r')
actions = file_ptr.read().split('\n')[:-1]
file_ptr.close()
action_dict = dict()
for a in actions:
    action_dict[a.split()[1]] = int(a.split()[0])
num_classes = len(action_dict)

logger.add(sys.stdout, colorize=True, format="{message}")
logger.info(f"!! Running Activity: task {target_task} / {target_activity}")
logger.info(f"!! Setup: {args.train_setup}")
logger.info("!!! All action-label information")
logger.info(action_dict)
logger.info("---"*40)

seen_activities = activities[:target_task]
logger.info(f"!!! Seen classes: {seen_activities} / Current task: {target_task})")

vid_list_file = f"{data_root_path}/"+args.dataset+"/splits/train.split"+args.split+".bundle"
vid_list_file_tst = f"{data_root_path}/"+args.dataset+"/splits/test.split"+args.split+".bundle"

if args.mode == 'segm':
    tca_param = None
else:
    tca_param = {
        "x_dim": 2048,
        "a_dim": num_classes,
        "c_dim": 1,
        "h_dim": 1024,
        "z_dim": 256
    }

start = time.time()
trainer = Trainer(num_stages, num_layers, num_f_maps, features_dim, num_classes, tca_param)
if args.mode == 'segm':
    if args.action == "train":
        batch_gen = BatchGenerator(num_classes, action_dict, 
                                gt_path, features_path, sample_rate, 
                                target_task, target_activity)
        batch_gen.read_data(vid_list_file)
        
        batch_gen_tst = BatchGenerator_Test(num_classes, action_dict, 
                                gt_path, features_path, sample_rate, 
                                target_task, target_activity)
        batch_gen_tst.read_data(vid_list_file_tst, seen_activities)
        
        
        if target_task > 0:
            trainer.model.load_state_dict(torch.load(old_model_dir + "tas/best.model", map_location='cpu'))
            logger.info(f"load previously trained model. (path: {old_model_dir})")
            batch_gen.read_replay_data(replay_dir, args.memory_size, activities)
            logger.info(f"load pre-generated features before {target_task} tasks from {replay_dir}.")
        
        train_results, train_results_segm = trainer.train_TAS(tas_model_dir, batch_gen, batch_gen_tst,
                          num_epochs=num_epochs, batch_size=bz, learning_rate=lr, 
                          action_dict=action_dict, device=device, logger=logger)
        
        csv_path = f"{model_dir}/results.csv"
        if not os.path.exists(csv_path):
            header = ["task"] + list(train_results.keys())
        else:
            header = []
        
        import csv
        with open(csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            if len(header) != 0:
                writer.writerow(header)
            
            writer.writerow([target_task] + list(train_results.values()))
            
        csv_path2 = f"{model_dir}/results_segm.csv"
        if not os.path.exists(csv_path2):
            header = ["task"] + list(train_results_segm.keys())
        else:
            header = []
        
        with open(csv_path2, mode='a', newline='') as file:
            writer = csv.writer(file)
            
            if len(header) != 0:
                writer.writerow(header)
            
            writer.writerow([target_task] + list(train_results_segm.values()))
            
    
    # if args.action == "predict":
    #     batch_gen_tst = BatchGenerator_Test(num_classes, action_dict, 
    #                             gt_path, features_path, sample_rate, 
    #                             target_task, target_activity)
    #     batch_gen_tst.read_data(vid_list_file_tst, seen_activities)
    #     trainer.predict(batch_gen_tst, tas_model_dir, tas_results_dir, features_path, action_dict, device, sample_rate)
        
    
    elif args.action == "test":
        load_type = args.test_type
        logger.info(f"Test for task {target_task} from model ({tas_model_dir}{load_type}.model)")
        trainer.model.load_state_dict(torch.load(tas_model_dir + f"{load_type}.model", map_location='cpu'))
        trainer.model.to(device)
        
        batch_gen_tst = BatchGenerator_Test(num_classes, action_dict, 
                                gt_path, features_path, sample_rate, 
                                target_task, target_activity)
        batch_gen_tst.read_data(vid_list_file_tst, seen_activities)
        
        results = trainer.test(batch_gen_tst, action_dict, device)
        
        
    elif args.action == "test_all":
        model_dir = "/".join(model_dir.split("/")[:-2])
        load_type = args.test_type
        
        print(f"test all splits from the {load_type} model type")
        print("model loaded from", model_dir)
        all_results = []
        
        for s in range(1, num_cross_valid):
            model_dir_s = model_dir + f"/split_{s}/task_{target_task}/tas"
            vid_list_file_tst = f"{data_root_path}/"+args.dataset+"/splits/test.split"+str(s)+".bundle"
            
            batch_gen_tst = BatchGenerator_Test(num_classes, action_dict, 
                                        gt_path, features_path, sample_rate, 
                                        target_task, target_activity, replay_dir)
            batch_gen_tst.read_data(vid_list_file_tst, seen_activities)
            print("Test size:", batch_gen_tst.get_size)
        
            print(f"Test the split {s} for task {target_task} from model ({model_dir_s}/{load_type}.model)")
            trainer.model.load_state_dict(torch.load(model_dir_s + f"/{load_type}.model", map_location='cpu'))
            trainer.model.to(device)
            
            results = trainer.test(batch_gen_tst, action_dict, device)
            
            all_results.append(list(results.values())[:-1])
            print(f"split {s}\t{list(results.values())}\n")
            
            batch_gen_tst.reset()
            
        print(np.mean(np.array(all_results), axis=0))
        print(f"All Sum: {np.array(all_results).sum()} / Segm. Metric Sum: {np.array(all_results[1:]).sum()}")
            


elif 'gen' in args.mode:
    batch_gen2 = TCA_BatchGenerator(num_classes, action_dict, 
                                    gt_path, features_path, sample_rate, 
                                    target_task, target_activity, args.dataset)
    
    batch_gen2.read_data(vid_list_file)
    trainer.train_TCA(tca_model_dir, batch_gen2, num_epochs=tca_epochs, batch_size=4096, learning_rate=tca_lr, device=device, logger=logger)


elif args.mode == 'sample':
    batch_gen2 = TCA_BatchGenerator(num_classes, action_dict, 
                                    gt_path, features_path, sample_rate, 
                                    target_task, target_activity, args.dataset)
    batch_gen2.read_data(vid_list_file, mode=args.mode)
    batch_gen2.sample(trainer.TCA_model, action_dict, tca_param['z_dim'], 
                      tca_model_dir, replay_dir, device)
end = time.time()
logger.info(f"Running Time: {timedelta(seconds=int(end - start))}")