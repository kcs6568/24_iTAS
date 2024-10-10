import sys
import copy
import numpy as np
# from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from eval import f_score, edit_score


class cVAE(nn.Module):
    def __init__(self, x_dim, a_dim, c_dim, h_dim, z_dim):
        super(cVAE, self).__init__()
        self.enc_fc1 = nn.Linear(x_dim + a_dim + c_dim, h_dim, dtype=torch.float32)
        self.enc_fc2 = nn.Linear(h_dim, z_dim, dtype=torch.float32)
        self.enc_mu = nn.Linear(z_dim, z_dim, dtype=torch.float32)
        self.enc_var = nn.Linear(z_dim, z_dim, dtype=torch.float32)
        self.dec_fc1 = nn.Linear(z_dim + a_dim + c_dim, h_dim, dtype=torch.float32)
        self.dec_fc2 = nn.Linear(h_dim, x_dim, dtype=torch.float32)
        
        
    def enc(self, x, a=None, c=None):
        if a is not None:
            x = torch.cat([x,a], -1)
        if c is not None:
            x = torch.cat([x,c], -1)
        
        h1 = F.relu(self.enc_fc1(x))
        h2 = F.relu(self.enc_fc2(h1))
        
        mu = self.enc_mu(h2)
        logvar = self.enc_var(h2)
        
        return mu, logvar
        

    def dec(self, z, a=None, c=None):
        if a is not None:
            z = torch.cat([z,a], -1)
        if c is not None:
            z = torch.cat([z,c], -1)
        
        h = F.relu(self.dec_fc1(z))
        return self.dec_fc2(h)
        
    
    def sampling(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add(mu) # return z sample
    
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    
    def forward(self, x, a=None, c=None):
        mu, logvar = self.enc(x, a, c)
        z = self.reparameterize(mu, logvar)
        out = self.dec(z, a, c)
        return out, mu, logvar


class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MultiStageModel, self).__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel(num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages-1)])

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


class Trainer:
    def __init__(self, num_blocks, num_layers, num_f_maps, dim, num_classes, tca_param):
        self.model = MultiStageModel(num_blocks, num_layers, num_f_maps, dim, num_classes)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes
        
        if tca_param is not None:
            self.TCA_model = cVAE(**tca_param)
            print(self.TCA_model)
        

    def train_TAS(self, save_dir, batch_gen, batch_gen_tst, num_epochs, batch_size, learning_rate, 
                  action_dict, device, logger):
        self.model.train()
        self.model.to(device)
        best_all_score = 0
        best_all_count = 0
        
        
        best_segm_score = 0
        best_segm_count = 0
        
        best_all_info = {}
        best_segm_info = {}
        
        # optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        
        logger.info("---------------- Start time stamping ----------------")
        logger.info(f"Data size --> [Train]: old - {batch_gen.exemplar_count} / new - {batch_gen.cur_task_len} \t [Test]: {batch_gen_tst.get_size}")
        for epoch in range(num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            while batch_gen.has_next():
                batch_input, batch_target, mask = batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                
                optimizer.zero_grad()
                predictions = self.model(batch_input, mask)

                loss = 0
                for p in predictions:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    loss += 0.15*torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, :, 1:])

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
                total += torch.sum(mask[:, 0, :]).item()
                
            batch_gen.reset()
            logger.info(f"[Task {batch_gen.cur_task}][epoch {epoch+1}/{num_epochs}][Train] loss: {(epoch_loss / len(batch_gen.list_of_examples)):.4f} \t accuracy: {(float(correct)/total):.4f}")
            perf, perf_wo_acc = self.test(batch_gen_tst, action_dict, device)
            test_info = f"[Task {batch_gen.cur_task}][epoch {epoch+1}/{num_epochs}][Test ] Acc: {perf['acc']:.4f} \t Edit: {perf['edit']:.4f} \t F1@[10, 25, 50]: {perf['f1@0.1']:.4f} {perf['f1@0.25']:.4f} {perf['f1@0.5']:.4f} \t sum: {perf['sum']:.4f} \t sum w/o acc: {perf_wo_acc['sum']:.4f}"
            
            torch.save(self.model.state_dict(), save_dir + "last.model")
            torch.save(optimizer.state_dict(), save_dir + "last.opt")
            
            if perf['sum'] >= best_all_score:
                torch.save(self.model.state_dict(), save_dir + "best.model")
                torch.save(optimizer.state_dict(), save_dir + "best.opt")
                best_all_score = perf['sum']
                best_all_info = perf
                best_all_count += 1
                test_info += f"  * ({best_all_count})"
            
            if perf_wo_acc['sum'] >= best_segm_score:
                torch.save(self.model.state_dict(), save_dir + "best_segm.model")
                torch.save(optimizer.state_dict(), save_dir + "best_segm.opt")
                best_segm_score = perf_wo_acc['sum']
                best_segm_info = perf_wo_acc
                best_segm_count += 1
                test_info += f" / @ ({best_segm_count})"
            
            logger.info(test_info)
        
        logger.info(f"Best score w/  acc: sum={best_all_score}\n\t{best_all_info}\n")
        logger.info(f"Best score w/o acc: sum={best_segm_score}\n\t{best_segm_info}")
        
        return best_all_info, best_segm_info
    
    
    def test(self, batch_gen_tst, actions_dict, device):
        self.model.eval()
        perf = dict()
        
        overlap = [.1, .25, .5]
        tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)

        correct = 0
        total = 0
        edit = 0        
        
        with torch.no_grad():
            while batch_gen_tst.has_next():
                batch_input, batch_target = batch_gen_tst.next_batch(1)
                batch_input, batch_target = batch_input.to(device), batch_target.to(device)
                
                p = self.model(batch_input, torch.ones(batch_input.size(), device=device))
                _, predicted = torch.max(p.data[-1], 1)
                
                correct += ((predicted == batch_target).float().squeeze(1)).sum().item()
                
                total += batch_target.shape[-1]
                
                predicted = predicted.squeeze()
                batch_target = batch_target.squeeze()
                predicted_list = []
                batch_target_list = []
                
                for i in range(len(predicted)):
                    predicted_list = np.concatenate((predicted_list, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]]*batch_gen_tst.sample_rate))
                    batch_target_list = np.concatenate((batch_target_list, [list(actions_dict.keys())[list(actions_dict.values()).index(batch_target[i].item())]]*batch_gen_tst.sample_rate))
                
                bg_class = np.array([k for k in batch_target_list if "SIL" in k])
                bg_class = list(np.unique(bg_class))
                
                edit += edit_score(predicted_list, batch_target_list)

                for s in range(len(overlap)):
                    tp1, fp1, fn1 = f_score(predicted_list, batch_target_list, overlap[s])
                    tp[s] += tp1
                    fp[s] += fp1
                    fn[s] += fn1
        
        perf['acc'] = float(correct) / total         
        perf['edit'] = ((1.0*edit)/len(batch_gen_tst.list_of_examples))
        for s in range(len(overlap)):
            precision = tp[s] / float(tp[s]+fp[s])
            recall = tp[s] / float(tp[s]+fn[s])

            f1 = 2.0 * (precision*recall) / (precision+recall)
            f1 = np.nan_to_num(f1)*100
            perf[f'f1@{overlap[s]}'] = f1

        perf_wo_acc = copy.deepcopy(perf)
        perf_wo_acc.pop('acc')
        
        perf['sum'] = perf['acc'] + perf['edit'] + perf['f1@0.1'] + perf['f1@0.25'] + perf['f1@0.5']
        perf_wo_acc['sum'] = perf['edit'] + perf['f1@0.1'] + perf['f1@0.25'] + perf['f1@0.5']
        
        
        self.model.train()
        batch_gen_tst.reset()

        return perf, perf_wo_acc
    
    
    def predict(self, batch_gen_tst, model_dir, results_dir, features_path, actions_dict, device, sample_rate):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/best.model"))
            
            for vid in batch_gen_tst.list_of_examples:
                #print vid
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                predictions = self.model(input_x)
                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze()
                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [list(actions_dict.keys())[list(actions_dict.values()).index(predicted[i].item())]]*sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()
                
            
    def train_TCA(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, device, logger):
        self.TCA_model.train()
        self.TCA_model.to(device)
        
        self.vae_mse = nn.MSELoss(reduction='sum')
        
        def vae_loss(pred, inputs, mu, logvar):
            recon_loss = self.vae_mse(pred, inputs)
            KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            return recon_loss / inputs.size(0), KLD / inputs.size(0)
            
        
        logger.info(f"Target Activity: {batch_gen.activity}")
        logger.info(f"Training Info.:\n\t Sample length: {batch_gen.batch_input.shape[0]} \n\t batch size: {batch_size} --> Iteration: {batch_gen.batch_input.shape[0] // batch_size}")
        
        best_loss = 100.
        optimizer = optim.Adam(self.TCA_model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            epoch_loss = 0
            batch_iters = 0
            
            while batch_gen.has_next():
                batch_input, onehot_labels, coherent_values = batch_gen.next_batch(batch_size)
                batch_input, onehot_labels, coherent_values = batch_input.to(device), onehot_labels.to(device), coherent_values.to(device)
                
                optimizer.zero_grad()
                outputs, mu, logvar = self.TCA_model(batch_input, onehot_labels, coherent_values)
                
                recon_loss, reg_loss = vae_loss(outputs, batch_input, mu, logvar)
                loss = recon_loss + reg_loss
                
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                batch_iters += 1
            
            batch_gen.reset()
            
            avg_loss = epoch_loss / batch_iters
            log_info = f"[cVAE][Epoch {epoch+1}/{num_epochs}] total_loss: {avg_loss:.4f} \t recon_loss: {recon_loss:.4f} \t reg_loss: {reg_loss:.4f}"
            if best_loss > avg_loss:
                torch.save(self.TCA_model.state_dict(), save_dir + "/best.model")
                torch.save(optimizer.state_dict(), save_dir + "/best.model.opt")
                best_loss = avg_loss    
                log_info += "  *"
                
            torch.save(self.TCA_model.state_dict(), save_dir + "/last.model")
            torch.save(optimizer.state_dict(), save_dir + "/last.model.opt")
            
            logger.info(log_info)
            
            # break
    