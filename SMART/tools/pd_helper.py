#!/usr/bin/env python
# coding: utf-8

import glob2
import pandas as pd
import numpy as np
import yaml
import os
import sys
from shutil import copyfile


def copy_img(paths, in_dir, out_dir):
    """
    Copy the files in the path to the destination folder.
    Args:
        paths (list): Relative paths
        in_dir (str):  The root directory of the files
        out_dir (str): Destination folder
    """
    print('file count', len(paths))
    os.makedirs(out_dir, exist_ok=True)
    err_count = 0
    for path in paths:
        in_name = os.path.join(in_dir, path)
        base_name = os.path.basename(in_name)
        out_name = os.path.join(out_dir, base_name)
        try:
            copyfile(in_name, out_name)
        except:
            err_count += 1
            print(f'err {err_count}: ', in_name)


def copy_img_by_col(df, col, in_dir, out_dir, path_col='path', sample_num=None):
    """
    Create different folders based on the value of a column in df. Copy the image to the folder.
    Args:
        df (pd.DataFrame): [description]
        col (str): [description]
        in_dir (str): [description]
        out_dir (str): [description]
        path_col (str, optional): The column name of the image in CSV. Defaults to 'path'.
        sample_num( int): the number of samples
    """
    if isinstance(col, str):
        for val in df[col].unique():
            print('process :'f'{col}_{val}')
            sub_df = df[df[col] == val]
            path_se = sub_df[path_col]
            if sample_num and (sample_num <= path_se.shape[0]):
                path_se = path_se.sample(n=sample_num)
            sub_paths = path_se.to_list()
            sub_out_dir = os.path.join(out_dir, f'{col}_{val}')
            copy_img(sub_paths, in_dir, sub_out_dir)
    elif isinstance(col, list):
        def _fun(df):
            dir_name = '_{}_'.join(col + ['']).format(*df.name)
            dir_name = dir_name[:-1] + '/'
            print('process :', dir_name)

            path_se = df[path_col]
            if sample_num and (sample_num <= path_se.shape[0]):
                path_se = path_se.sample(n=sample_num)
            sub_paths = path_se.to_list()

            sub_out_dir = os.path.join(out_dir, dir_name)
            copy_img(sub_paths, in_dir, sub_out_dir)
        df.groupby(col).apply(_fun)


def merge_pred_with_gt(pred_name, gt_csv, on=[]):
    """
    Combining the predicted CSV with the GTs CSV,

    Args:
        pred_name (str): single predicted CSV
        gt_csv (pd.DataFrame)): the GTs CSV
        on (list, optional): The key to which the two CSVs are connected. Defaults to [ ].

    Returns:
        [pd.DataFrame]: The merged CSV
    """
    def _get_pred_name_in_disk(row):
        start = 'gt={}_pred={}_'.format(row['gts'], row['preds'])
        out_name = start + row['base_name']
        return out_name

    pred_csv = pd.read_csv(pred_name)
    if isinstance(gt_csv , str):
        gt_csv = pd.read_csv(gt_csv)

    pred_csv['base_name'] = pred_csv['paths'].apply(os.path.basename)
    pred_csv['pred_name'] = pred_csv.apply(_get_pred_name_in_disk, axis=1)
    gt_csv['base_name'] = gt_csv['path'].apply(os.path.basename)
 
    if on:
        merge = pd.merge(pred_csv, gt_csv,left_on=on[0], right_on=on[1], how='left')
    else:
        merge = pd.merge(pred_csv, gt_csv, left_on='base_name',right_on='base_name', how='left')
    return merge
