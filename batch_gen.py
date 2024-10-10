#!/usr/bin/python2.7

import os
import glob
import random
import numpy as np

import torch
from einops import rearrange


class BatchGenerator(object):
    def __init__(self, num_classes, actions_dict, 
                 gt_path, features_path, sample_rate,
                 task, activity):
        self.list_of_examples = list()
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate
        self.cur_task = task
        self.activity = activity
        self.class_per_task = len(self.activity)
        self.exemplar_count = 0
        self.vid2task = {}
        self.cnt = 1
        
        
    @property
    def get_size(self):
        return len(self.list_of_examples)
    
        
    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False
    
    def read_data(self, vid_list_file):
        file_ptr = open(vid_list_file, 'r').read().split('\n')[:-1]
        self.list_of_examples = []
        for file in file_ptr:
            for act in self.activity:
                if act in file:
                    self.list_of_examples.append(file)
        
        self.cur_task_len = len(self.list_of_examples)
        random.shuffle(self.list_of_examples)
        
    
    def read_replay_data(self, replay_dir, memory_size, activity_order):
        list_of_replay = []
        mSize_per_class = memory_size // (self.class_per_task * self.cur_task)
        
        for t in range(self.cur_task):
            for act in activity_order[t]:
                feat_list = glob.glob(replay_dir + f"/task_{t}/*{act}*")    
                random.shuffle(feat_list)
                sampled_replay = random.sample(feat_list, mSize_per_class)
                list_of_replay.extend(sampled_replay)
                self.exemplar_count += len(sampled_replay)
                
        
        self.list_of_examples.extend(list_of_replay)
        random.shuffle(self.list_of_examples)
        

    def next_batch(self, batch_size):
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size
            
        batch_input = []
        batch_target = []
        for vid in batch:
            if 'replay' in vid:
                features = np.load(vid)
                file_ptr = open(self.gt_path + vid.split('/')[-1][:-4] + '.txt', 'r') # load gt for generated features
                
            else:
                features = np.load(self.features_path + vid.split('.')[0] + '.npy')
                file_ptr = open(self.gt_path + vid, 'r')
            
            content = file_ptr.read().split('\n')[:-1]
            classes = np.zeros(min(np.shape(features)[1], len(content)))
            for i in range(len(classes)):
                classes[i] = self.actions_dict[content[i]] # {action: label}
            batch_input.append(features[:, ::self.sample_rate])
            batch_target.append(classes[::self.sample_rate])
        
        '''
        mask:
            (1) batch size가 1보다 크고,
            (2) batch간 time length가 맞지 않아 zero padding을 해야 하는 상황에서,
             --> 올바른 output을 내기 위해 사용
             
             If I set the batch size larger than 1, the padding will happen and the mask will select the relevant outputs for evaluation.
        '''
        
        length_of_sequences = list(map(len, batch_target))
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)
        mask = torch.zeros(len(batch_input), self.num_classes, max(length_of_sequences), dtype=torch.float)
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
            mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])
        
        return batch_input_tensor, batch_target_tensor, mask
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

class BatchGenerator_Test(object):
    def __init__(self, num_classes, actions_dict, 
                 gt_path, features_path, sample_rate,
                 task, activity):
        self.list_of_examples = list()
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate
        self.cur_task = task
        self.activity = activity
        self.vid2task = {}
        
        
    @property
    def get_size(self):
        return len(self.list_of_examples)

        
    def reset(self):
        self.index = 0


    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False
    
    
    def read_data(self, vid_list_file, seen_class):
        file_ptr = open(vid_list_file, 'r').read().split('\n')[:-1]
        for file in file_ptr:
            for t, act in enumerate(self.activity): # old class + new class
                if act in file:
                    self.list_of_examples.append(file)
                    self.vid2task[file] = t
        
        if len(seen_class) > 0:
            for file in file_ptr:
                for classes in seen_class:
                    for act in classes:
                        if act in file:
                            self.list_of_examples.append(file)
    
    
    def next_batch(self, batch_size):
        batch = self.list_of_examples[self.index:self.index + batch_size]
        self.index += batch_size
        
        batch_input = []
        batch_target = []
        for vid in batch:
            features = np.load(self.features_path + vid.split('.')[0] + '.npy')
            file_ptr = open(self.gt_path + vid, 'r')
            content = file_ptr.read().split('\n')[:-1]
            classes = np.zeros(min(np.shape(features)[1], len(content)))
            for i in range(len(classes)):
                classes[i] = self.actions_dict[content[i]] # {action: label}
            batch_input.append(features[:, ::self.sample_rate])
            batch_target.append(classes[::self.sample_rate])
        
        length_of_sequences = list(map(len, batch_target))
        batch_input_tensor = torch.zeros(len(batch_input), np.shape(batch_input[0])[0], max(length_of_sequences), dtype=torch.float)
        batch_target_tensor = torch.ones(len(batch_input), max(length_of_sequences), dtype=torch.long)*(-100)
        
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, :np.shape(batch_input[i])[1]] = torch.from_numpy(batch_input[i])
            batch_target_tensor[i, :np.shape(batch_target[i])[0]] = torch.from_numpy(batch_target[i])
        
        return batch_input_tensor, batch_target_tensor


    
    
    
    
    
    
    
    
    
    
    
    
    
class TCA_BatchGenerator(object):
    def __init__(self, num_classes, 
                actions_dict, 
                 gt_path, features_path, sample_rate,
                 task, activity, dataset):
        self.list_of_examples = list()
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate
        self.cur_task = task
        self.activity = activity
        self.seq_info_path = f"./metadata/{dataset}/seq_info"
        
        
    def reset(self):
        self.index = 0
        self.shuffle()
        

    def has_next(self):
        # if self.index < len(self.list_of_examples):
        if self.index < self.batch_input.shape[0]:
            return True
        return False
    
    def len(self):
        return self.batch_input.shape[0]
    

    def read_data(self, vid_list_file, mode='gen'):
        file_ptr = open(vid_list_file, 'r').read().split('\n')[:-1]
        
        self.list_of_examples = []
        for file in file_ptr:
            for act in self.activity:
                if act in file:
                    self.list_of_examples.append(file)
        
        if mode == 'gen':
            random.shuffle(self.list_of_examples)
            self.preprocess()
            self.shuffle()
    
    
    def shuffle(self):
        idx = torch.randperm(self.batch_input.size(0))
        self.batch_input = self.batch_input[idx]
        self.batch_target = self.batch_target[idx]
        self.batch_onehoe = self.batch_onehoe[idx]
        self.batch_coherence = self.batch_coherence[idx]
        
    
    def preprocess(self):
        batch_input = []
        batch_target = []
        batch_co_values = []
        batch_onehot = []
        
        for vid in self.list_of_examples:
            features = np.load(self.features_path + vid.split('.')[0] + '.npy')
            file_ptr = open(self.gt_path + vid, 'r')
            content = file_ptr.read().split('\n')[:-1]
            classes = np.zeros(min(np.shape(features)[1], len(content)))
            
            for i in range(len(classes)):
                classes[i] = self.actions_dict[content[i]] # {action: label}
                
            batch_input.append(features[:, ::self.sample_rate])
            batch_target.append(classes[::self.sample_rate])
            
            # coherent_values = np.zeros(min(np.shape(features)[1], len(content)), dtype=np.float32)
            coherence = []
            onehot_labels = np.zeros((min(np.shape(features)[1], len(content)), self.num_classes), dtype=np.float32)
            seq_info = open(f"{self.seq_info_path}/{vid}", 'r').read().split('\n')[:-1]
            
            for i, info in enumerate(seq_info):
                action, start, duration, end = info.split(",")
                start, end, duration = int(start), int(end), int(duration)
                action = f"{action}_{self.cur_task}"
                
                duration = end-start
                if i == len(seq_info)-2:
                    duration += 1   
                
                coherence = np.concatenate([coherence, np.linspace(0, 1, duration, dtype=np.float32)], dtype=np.float32)
                onehot_labels[start:end, self.actions_dict[content[start]]] = 1
            
            # batch_co_values.append(coherent_values)    
            batch_co_values.append(coherence)
            batch_onehot.append(onehot_labels)
        
        self.batch_input = torch.from_numpy(np.concatenate(batch_input, axis=1)).transpose(1, 0)
        self.batch_target = torch.from_numpy(np.concatenate(batch_target, axis=0)).type(torch.float32)
        self.batch_onehoe = torch.from_numpy(np.concatenate(batch_onehot, axis=0))
        self.batch_coherence = torch.from_numpy(np.concatenate(batch_co_values, axis=0, dtype=np.float32)).unsqueeze(-1)
        
    
    def next_batch(self, batch_size):
        batch_input = self.batch_input[self.index:self.index + batch_size]
        batch_onehot = self.batch_onehoe[self.index:self.index + batch_size]
        batch_coherence = self.batch_coherence[self.index:self.index + batch_size]
        self.index += batch_size
        
        
        return batch_input, batch_onehot, batch_coherence


    def sample(self, model, action_dict, latent_size, gen_model_dir, save_dir, device):
        print(f"!!! Target task: {self.cur_task}")
        print(f"!!! Save dir: {save_dir}")
        
        os.makedirs(save_dir, exist_ok=True)
        
        model.eval()
        with torch.no_grad():
            model.to(device)
            
            cum_count = 0
            all_vid = {act: [] for act in self.activity}
            for vid in self.list_of_examples:
                for act in self.activity:
                    if act in vid:
                        all_vid[act].append(vid)
            
            for act, vid_list in all_vid.items():
                count = 0
                model_weight =  gen_model_dir + f"/best.model"
                model.load_state_dict(torch.load(model_weight, map_location='cpu'))
                print(f"\n!!! Activity: {act} \t the best model weight is loaded form path: {model_weight}")
                
                for vid in vid_list:
                    seq_info = open(f"{self.seq_info_path}/{vid}", 'r').read().split('\n')[:-1]
                            
                    replay_x = []
                    for i, info in enumerate(seq_info): # segment-level generation
                        action, start, duration, end = info.split(",")
                        start, end, duration = int(start), int(end), int(duration)
                        action = f"{action}_{self.cur_task}"
                        
                        if i == len(seq_info) - 1:
                            duration += 1
                        
                        coherent_values = np.linspace(0, 1, duration, dtype=np.float32)
                        coherent_values = torch.from_numpy(np.array([coherent_values]))
                        
                        assert coherent_values.min() == 0
                        assert coherent_values.max() == 1
                        assert len(coherent_values[0]) == duration, f"Coherence array: {len(coherent_values[0])}"
                        
                        class_number = action_dict[action]
                        onehot_labels = torch.zeros((1, self.num_classes), dtype=torch.float32)
                        onehot_labels[0, class_number] = 1
                        
                        latent_vectors = torch.randn(1, latent_size, dtype=torch.float32)
                        for d in range(duration):
                            gen_x = model.dec(
                                latent_vectors.to(device),
                                onehot_labels.to(device), 
                                coherent_values[:, d:d+1].to(device)
                            )
                            replay_x.append(gen_x.cpu().numpy())
                            
                    replay_x = np.concatenate(replay_x)
                    replay_x = np.transpose(replay_x)
                    
                    file_name = vid.split(".")[0]
                    
                    np.save(f'{save_dir}' + f'/{file_name}.npy', replay_x)
                    count += 1
                    cum_count += 1
                    print(f"T{self.cur_task}-{act}-{count} ({cum_count}) \t Pseudo features for activity {act} were generated ({vid} {replay_x.shape}).")
                    
            