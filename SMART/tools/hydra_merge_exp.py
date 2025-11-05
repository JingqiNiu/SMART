import os
import os.path as osp
import pandas as pd
import argparse
from tools_utils import open_file_of_exp,get_best_score_in_trail



def merge_exp( exp_root_dir):
    out_dict = {'final_score': {} }
    out_dict_csv =  osp.join(exp_root_dir,'merge_final_score.csv')
    result_txt = osp.join(exp_root_dir,'merge_test_result.txt')

    for item in os.listdir(exp_root_dir):

        exp_dir = os.path.join(exp_root_dir,item)
        if not osp.isdir(exp_dir):
            continue

        test_result_dir = osp.join(exp_dir, 'test_result/')
        train_result_dir = osp.join(exp_dir, 'evaluate_train_result/')

        # Read files
        test_result_dict = open_file_of_exp(test_result_dir)
        train_result_dict = open_file_of_exp(train_result_dir)
        final_score ,_,_ = get_best_score_in_trail(exp_dir)
        
        out_dict['final_score'][item] = final_score
        cls_report = test_result_dict.get('cls_report.txt',"")
        cls_report +=  test_result_dict.get('reg_report.txt',"")
        with open(result_txt, 'a') as result_txt_f:
            result_txt_f.write(item + '\n')
            result_txt_f.write(cls_report)
    out_csv= pd.DataFrame(out_dict)
    out_csv.sort_values('final_score', axis=0, inplace=True ,ascending=False)
    out_csv.to_csv(out_dict_csv)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='hydra merge exp.')
    parser.add_argument('root_dir',  type=str, help='the path to exp directory')
    args = parser.parse_args()

    merge_exp( args.root_dir)