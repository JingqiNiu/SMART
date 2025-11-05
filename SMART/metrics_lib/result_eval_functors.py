import os
import shutil
import csv

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report,roc_curve,auc, roc_auc_score
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd 
import  seaborn  as sns
matplotlib.use('Agg')
def generate_binary_confusion(input_confusion, result_txt):
    
    '''
    Function to generate binary report in a multi-class (classes >= 3) classification problem
    Parameters:
        input_confusion -> numpy.ndarray: the input confusion matrix with n*n
        result_txt -> str: the report txt file to write the output
    '''

    h, w = input_confusion.shape
    assert h==w
    
    with open(result_txt, 'a') as f:
        for i in range(h):
            tn = np.sum(input_confusion[:i+1,:i+1])
            fp = np.sum(input_confusion[:i+1,i+1:])
            tp = np.sum(input_confusion[i+1:,i+1:])
            fn = np.sum(input_confusion[i+1:,:i+1])
        
            confusion_matrix = np.array([[tn, fp],[fn, tp]])
            
            precision = tp/(tp+fp)
            accuracy = (tp+tn)/(tp+fp+tn+fn)
            specificity = tn/(tn+fp)
            recall = tp/(tp+fn)
            F1 = 2*(precision*recall)/(precision+recall)
            youden = (tp / (tp + fn)) + (tn / (tn + fp)) - 1
            f.write('If the cls : {} is Positive \n.'.format(i))
            f.write('Confusion Matrix: \n{}\n.'.format(confusion_matrix))
            f.write('Youden {:.3f},Precision {:.3f}, accuracy, {:.3f}, specificity, {:.3f}, recall {:.3f}, F1 {:.3f}.\n\n\n'.format(youden,precision,accuracy,specificity,recall,F1))
        for i in range(input_confusion.shape[0]):
            # 计算该类别的TN和FP
            TN = input_confusion.sum() - input_confusion[i, :].sum() - input_confusion[:, i].sum() + input_confusion[i, i]
            FP = input_confusion[:, i].sum() - input_confusion[i, i]

            # 计算该类别的specificity，并添加到列表中
            specificity = TN / (TN + FP)
            print(f'For cls : {i} , specificity is {specificity}')
def get_cls_report(result_dic, result_txt, binary = False):
    
    '''
    Function to generate the classification report
    Parameters:
        gts -> int list: the list of ground truth labels
        preds -> int list: the list of predicted labels
        result_txt -> str: the report txt file to write the output
    Returns:
        confuse -> numpy.ndarray: the confusion matrix
        cls_report -> str: the whole classification report
    '''
    all_preds = result_dic['preds']
    all_gts = result_dic.get('gts', [None]*len(all_preds))
    
    all_probs = result_dic['probs']
    confuse = confusion_matrix(all_gts, all_preds)
    a = result_dic['gts']
    # if binary == True:
    #     print(f'the  gts is {a.shape} all_probs shape {all_probs.shape}')
    #     auroc = roc_auc_score(result_dic['gts'], all_probs[:,1])
    # else:
    #     for i in range(all_probs.shape[1]):
    #         auroc = roc_auc_score((result_dic['gts'] ==i).astype(int), all_probs[:,i])
    #         print(f'For class {i} the auroc is {auroc}')
        
    cls_report = classification_report(all_gts, all_preds,digits=3)
    
    with open(result_txt, 'a') as f:
        # if binary == True:
        #     f.write(f'Teh AUROC is: {auroc} \n')
        f.write('Confusion Matrix: \n')
        f.write(str(confuse))
        f.write('\n')
        f.write(cls_report)
        f.write('\n\n')
        
    return confuse, cls_report

def draw_roc(gts, probs, write_path):
    
    '''
    Function to draw ROC curve
    Parameters:
        gts -> int list: the list of ground truth labels
        probs -> float list: the list of predicted probabilities
        write_path -> str: path to write the ROC curve
    '''

    fpr,tpr,_ = roc_curve(gts, probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr,tpr,label='AUC = %0.3f' % roc_auc)
    plt.plot(np.array([0.0,1.0]),np.array([0.0,1.0]),color='black',linestyle='dashed')
    plt.title('ROC curve')# of '+disease_type)
    plt.legend(loc='lower right')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')

    plt.savefig(os.path.join(write_path, 'roc.jpg'),dpi=800)
    
def print_failure_cases(gts, preds, img_paths, write_path):

    '''
    Function to print the failure cases
    Parameters:
        gts -> int list: the list of ground truth labels
        preds -> int list: the list of predicted labels
        img_paths -> str list: the list of all image paths
        write_path -> str: the path to write the failure cases
    '''
    
    # create the failure case directories
    failure_case_dir = os.path.join(write_path, 'failure_cases')
    
    if not os.path.isdir(failure_case_dir):
        os.mkdir(failure_case_dir)

    # copy and rename failure cases based on the predicted labels    
    for item in zip(gts, preds, img_paths):
        if item[0] != item[1]:
            prefix = 'gt={}_pred={}_'.format(item[0], item[1])
            write_img_name = prefix + os.path.basename(item[2])
            shutil.copy(item[2], failure_case_dir)
            os.rename(os.path.join(failure_case_dir, os.path.basename(item[2])), os.path.join(failure_case_dir, write_img_name))




def eval_model_cls(result_dic, write_path ,need_failure = True, binary = True):
    
    '''
    Function to evaluate the classification model results
    Parameters:
        result_dic -> dict: the result dictionary returned by the deep classification model
        write_path -> str: the path to write the results
    '''
    os.makedirs(write_path,exist_ok=True)
    all_preds = result_dic['preds']
    all_gts = result_dic.get('gts', [None]*len(all_preds))
    
    all_probs = result_dic['probs']
    all_paths = result_dic['paths']
    
    if None not in all_gts:
        result_txt = os.path.join(write_path, 'cls_report' + '.txt')
        confuse, cls_report = get_cls_report(result_dic, result_txt, binary = binary)
        generate_binary_confusion(confuse, result_txt)
        if need_failure:
            print_failure_cases(all_gts, all_preds, all_paths, write_path)
        if confuse.shape == (2,2):
            draw_roc(all_gts, all_probs[:,1], write_path)   
    write_csv(all_preds, all_gts, all_probs, all_paths,write_path )

def write_csv( all_preds, all_gts, all_probs, all_paths ,write_path):
    os.makedirs( write_path, exist_ok=True)

    all_csv = pd.concat([pd.Series(all_paths) ,pd.Series(all_preds) ,pd.Series(all_gts)] , axis=1)
    all_csv.columns = ['paths', 'preds',  'gts']
    class_num = all_probs.shape[1]
    for i in range(class_num):
        all_csv[f'prob{i}']=  all_probs[:,i].reshape(-1,1)
    csv_name = os.path.join(write_path, 'report_table.csv')
    all_csv.to_csv( csv_name , index =False)
    

def eval_model_cls_multitask(gtss, predss):
    
    pass

def eval_model_seg(result_dic, write_path):
    
    '''
    Function to evaluate the segmentation model
    Parameters:
        result_dic -> dict: the result dictionary returned by the deep classification model
        write_path -> str: the path to write the results
    '''

    if 'score' in result_dic:
        all_scores = result_dic['score']
        all_paths = result_dic['paths']

        report_txt = os.path.join(write_path, 'report.txt')
        report_csv = os.path.join(write_path, 'image_dice_scores.csv')

        with open(report_txt, 'w') as f, open(report_csv, 'w') as g:

            for i in range(all_scores.shape[1]):
                f.write('Average dice of class {}: {}\n'.format(i, np.average(all_scores[:,i])))

            writer = csv.writer(g)
            writer.writerow(['path', 'score'])

            for path, score in zip(all_paths, all_scores):
                writer.writerow([path, score])

def eval_model_seg_multitask(result_dic, write_path):

    pass

def eval_model_reg(result_dic,write_path,need_failure=1.0,threshold =0.5):
    os.makedirs(write_path,exist_ok=True)

    ## 生成CSV
    csv = result_dic['csv']
    csv_name = os.path.join(write_path , f'report_table.csv' )
    csv.to_csv(csv_name,index=False)

    # 生成文本
    result_txt = os.path.join(write_path , f'reg_report.txt' )
    with open(result_txt, 'w') as f:
        txt_content = result_txt + '\n' + result_dic['txt']
        f.write(txt_content)

    # 生成failure case
    failure_dir = os.path.join(write_path , 'failure_cases/')
    os.makedirs(failure_dir,exist_ok=True)
    failure_csv = csv[csv['abs_diff'] > threshold]
    if need_failure > 0:
        try:
            if isinstance(need_failure,float):
                failure_csv = failure_csv.sample(frac=need_failure)
                if failure_csv.shape[0] > 5000:
                    failure_csv = failure_csv.sample(n = 5000)
                    print('Too many failure case, only sample 5000.')
            else:
                failure_csv = failure_csv.sample(n=need_failure)
        except Exception as err:
            print(err)
    for i , row in failure_csv.iterrows():
        out_name = os.path.basename(row['paths'])
        score_str = f"{row['diff']:.2f}_{row['preds']:.2f}_{row['gts']:.2f}"
        out_name = f"{failure_dir}{score_str}#{out_name}"
        shutil.copyfile(row['paths'],out_name)
    
    # 生成分布图
    data = csv[['preds','gts']]
    fig,axes= plt.subplots(2,1,figsize=(10,10))
    group = [ axes[0],axes[1] ]
    data.plot.hist( bins=100,subplots=True,sharex =False,ax=group)
    plt.savefig(os.path.join(write_path , f'plot_hist.jpg' )) # 分布频率图

    fig = plt.figure()
    ax = sns.kdeplot(data=data,cut=0,fill=False,common_norm=True)
    ax.get_figure().savefig(os.path.join(write_path , f'plot_density.jpg' )) #  # 分布图

    fig = plt.figure()
    ax = csv['diff'].plot.hist( bins=100)
    ax.get_figure().savefig(os.path.join(write_path , f'plot_diff_hist.jpg' )) # 差距的直方图

    fig = plt.figure()
    ax = sns.scatterplot(csv['preds'], csv['gts'],alpha=0.5,marker='+')
    ax.get_figure().savefig(os.path.join(write_path, f'plot_pred_scatter.jpg') ) #散点图
    
    fig = plt.figure()
    ax = sns.scatterplot(csv['preds'].rank(), csv['gts'].rank(),alpha=0.5,marker='+')
    ax.get_figure().savefig(os.path.join(write_path, f'plot_rank_scatter.jpg') )  #rank