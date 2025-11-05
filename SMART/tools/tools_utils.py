
import os
import os.path as osp
import sys
import numpy as np
import yaml
import pandas as pd
import glob

sys.path.append('/'.join(__file__.split('/')[:-2]) )
from data_lib import get_data_infor, basic_read_csv
from tools.pd_helper import copy_img_by_col
from data_lib.process_csv import merge_multiple_dataset
from data_lib.parse_data_infor import data_str_list_to_dict

def open_file_of_exp(dir_path, suffix=['yaml', 'csv', 'txt']):
    """
    Read the files in the folder

    Args:
        dir_path (str): Path to the folder
        suffix (list, optional): The suffix name of the file that can be opened. Defaults to ['yaml' ,'csv' ,'txt'].

    Returns:
        [dict]: Contain files
    """
    res = {}
    if not osp.exists(dir_path):
        return {}
    files_names = os.listdir(dir_path)
    for file_name in files_names:
        abs_name = osp.join(dir_path, file_name)
        if not os.path.isfile(abs_name):
            continue
        f_open = open(abs_name, 'r')
        if file_name.endswith('.yaml') and ('yaml' in suffix):
            res[file_name] = yaml.load(f_open, Loader=yaml.FullLoader)

        elif file_name.endswith('.csv') and ('csv' in suffix):
            res[file_name] = pd.read_csv(abs_name)

        elif file_name.endswith('.txt') and ('txt' in suffix):
            res[file_name] = f_open.read()
        f_open.close()
    return res


def data_distribution(data_names, label_name, dataset_define=None):
    """
    The data distribution of datasets

    Args:
        data_names ( str,list,dict): [description]
        label_name (str): [description]
        dataset_define (dict): [description]

    Returns:
        [pd.DataFrame]: 
    """
    table = {}
    data_names = data_str_list_to_dict(data_names)

    for data_name in data_names:
        data_infor = get_data_infor(data_name, label_name, dataset_define)
        csv = basic_read_csv(data_infor[0])
        count = csv['label'].value_counts(dropna=False).sort_index()
        table[data_name] = pd.Series(count)
        
    table = pd.DataFrame(table).fillna(0).astype(np.int).T
    table['sum'] = table.sum(axis=1)
    sum_row = table.sum(axis=0)
    sum_row.name = 'sum'
    table = table.append(sum_row)
    return table


def get_csv_from_data_name(data_names, label_name):
    """
    Get the CSV of all data sets based on the dataset name

    Args:
        data_names ( str,list,dict): [description]
        label_name (str): [description]

    Returns:
        [pd.DataFrame]: [description]
    """
    csv_infor = get_data_infor(data_names, label_name)
    all_csv = []
    for item in csv_infor:
        csv = pd.read_csv(item['csv_path'])
        csv.rename({item['path_col']: 'path', item['label_col']: 'label'}, inplace=True, axis=1)
        csv['source'] = osp.basename(item['csv_path'])
        all_csv.append(csv)
    all_csv = pd.concat(all_csv, axis=0, ignore_index=True)

    return all_csv


def get_best_score_in_trail(trail_dir):

    pattern = os.path.join(trail_dir, 'checkpoints/epoch=*')
    all_ck = glob.glob(pattern)
    all_ck = pd.DataFrame(all_ck, columns=['ck_name'])
    if all_ck.shape[0] == 0:
        return None ,None ,all_ck
    all_ck['final_score'] = all_ck.ck_name.str.extract(
        r'.*final_score=(.*?)_val.*.ckpt').astype('float')
    all_ck.sort_values(by='final_score', ascending=False, inplace=True)
    best_final_score = all_ck['final_score'].iloc[0]
    best_ck_path =  all_ck['ck_name'].iloc[0]

    return best_final_score,best_ck_path, all_ck

def vis_dataset(data_name, scale, label_name, out_dir):
    """
    The dataset is sampled and copied to the target folder for visualization of the dataset

    Args:
        data_name (str): The name of a single data set
        scale ( int,float,list): Sampling the dataset. Four ways
        label_name (str): [description]
        out_dir (str): [description]
    """
    data_name = {data_name: scale}
    csv_infor = get_data_infor(data_name, label_name)

    csv = merge_multiple_dataset(csv_infor)
    csv.columns = ['path', label_name]
    copy_img_by_col(csv, label_name, "", out_dir)
