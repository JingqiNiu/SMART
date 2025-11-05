import time
import os
import numpy as np
import torch
import copy
from .process_define import DEFINE_PROCESS
from .metrics_define import DEFINE_SCORE

def calc_channel(pred_channels):
    task_num = len(pred_channels)
    pred_channels = np.cumsum([0] + pred_channels)
    slice_list = []
    for idx in range(task_num):
        ss = slice(pred_channels[idx] ,pred_channels[idx+1])
        slice_list.append ( ss)
    return slice_list

class Post_Process(object):
    def __init__(self, config ,global_rank,test_out_dir):
        self.cfg = copy.deepcopy(config)
        self.global_rank = global_rank
        self.test_out_dir = test_out_dir

        self.metric_define =  [DEFINE_SCORE[key] for key in config['metric_define']]
        self.process_define =  [DEFINE_PROCESS[key] for key in config['task_type']]
        self.slice_list=  calc_channel(config['channels'])
        self.label_names = config['label_name']
        default_weight = np.ones(len(self.process_define))
        self.score_weight = config.get('score_weight',default_weight)
        self.single_task = len(self.metric_define) == 1
        
    def on_epoch(self, labels ,results,paths ):
        device = 'cpu'
        if torch.is_tensor(results):
            device = results.device
            results = results.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
        if results.ndim == 1:
            results = results[:,None]
        if labels.ndim == 1:
            labels = labels[:,None]
            
        print('======================================================')
        print(time.asctime(time.localtime(time.time())))
        idx = 0
        final_score_all = []
        for process_fun,metric_fun ,ch ,name in zip(self.process_define,self.metric_define,self.slice_list,self.label_names):
            if self.single_task:
                test_dir = self.test_out_dir
            else:
                print(f'########## task_names: {name} ###############')
                test_dir = os.path.join(self.test_out_dir,name)
            result = results[:,ch]
            if result.ndim == 2 and result.shape[1] ==1:
                result =  np.squeeze(result,axis=1)
            
            score = process_fun(metric_fun,result,labels[:,idx],paths,self.global_rank,test_dir,**self.cfg)
            final_score_all.append(score)
            idx += 1
        final_score = np.array(final_score_all) * self.score_weight
        final_score = final_score.mean()
        print(f'final_score_all_task:{final_score_all}')        
        print(f'final_score:{final_score}')        
        print('======================================================')          
        return torch.tensor(final_score, device=device)

