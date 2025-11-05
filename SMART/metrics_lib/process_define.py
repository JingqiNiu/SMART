from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import torch 
import os
import numpy as np
import pandas as pd
from .result_eval_functors import eval_model_cls,eval_model_reg
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
def np_soft_max(x ,axis =1):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    s = x_exp / x_sum
    return s

def np_sigmoid(x):
  
    z = np.exp(-x)
    sig = 1 / (1 + z)

    return sig


def bin_all_metric(metric_fun,results, labels ,paths, global_rank,test_out_dir,**kwargs):
    need_failure = kwargs.get('need_failure', True)
    labels = labels.astype(int)
    probs = np_soft_max( results ,axis=1)
    preds = np.argmax( probs , axis =1)
    cm = confusion_matrix(labels, preds)
    cr = classification_report(y_true=labels, y_pred=preds)
    print(f'confusion_matrix:\n {cm} \n')
    print(cr)
    if len(paths) > 0 and global_rank == 0:  # Only  gpu 0 :
        res_dict = {'preds': preds,
                    'gts': labels,
                    'probs': probs,
                    'paths': np.array(paths)}
        eval_model_cls(res_dict, test_out_dir,need_failure)
    score = metric_fun(labels,preds, probs[:,1], test_out_dir=test_out_dir)
    return score
def cal_multicls_specificity(matrix):
    n_classes = len(matrix)
    specificity = {}
    for j in range(n_classes):
        numerator = matrix[:j, :j].sum() + matrix[j+1:, :j].sum() + matrix[:j, j+1:].sum() + matrix[j+1:, j+1:].sum()
        denominator = matrix[:j, :].sum() + matrix[j+1:, :].sum()
        specificity[j] = numerator / denominator
def micro_averaged_auroc(y_true_binary, probs):
    n_classes = probs.shape[1]
    aurocs = []
    class_weights = np.array([np.sum(y_true_binary == i) for i in range(n_classes)]) / len(y_true_binary)
    fpr = dict()
    tpr = dict()
    for i in range(n_classes):
        label = np.where(y_true_binary == i, 1, 0)
        fpr[i], tpr[i], _ = roc_curve(label, probs[:, i])
        roc_auc = auc(fpr[i], tpr[i])
        aurocs.append(roc_auc)
    print(len(aurocs))
    print(class_weights)
    micro_auroc = np.average(aurocs, weights=class_weights)
    
    return micro_auroc
def cls_softmax_default(metric_fun,results, labels ,paths, global_rank,test_out_dir,**kwargs):
    """ For multi-classification task"""
    need_failure = kwargs.get('need_failure', True)
    labels = labels.astype(int)
    probs = np_soft_max( results ,axis=1)
    preds = np.argmax( probs , axis =1)
    cm = confusion_matrix(labels, preds)
    cr = classification_report(y_true=labels, y_pred=preds)
    multi_spe = cal_multicls_specificity(cm)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    if len(paths) > 0 and global_rank == 0:  # Only  gpu 0 :
        res_dict = {'preds': preds,
                    'gts': labels,
                    'probs': probs,
                    'paths': np.array(paths)}
        eval_model_cls(res_dict, test_out_dir,need_failure)
    # score_old = metric_fun(labels,preds)
    print(probs.shape[1])
    for i in range(probs.shape[1]):
        # 将当前类别设置为正例，其他类别设置为负例
        y_true_binary = np.where(labels == i, 1, 0)
        # 计算 ROC 曲线
        fpr[i], tpr[i], _ = roc_curve(y_true_binary, probs[:, i])
        # 计算 AUC
        roc_auc[i] = auc(fpr[i], tpr[i])
    # total_fpr = np.concatenate(list(fpr.values()))
    # total_tpr = np.concatenate(list(tpr.values()))
    # micro_auc = auc(total_fpr, total_tpr)

    # 计算宏平均
    micro_auc = micro_averaged_auroc(labels, probs)
    print('the cal micro_auc is',micro_auc)
    macro_auc = np.mean(list(roc_auc.values()))
    print('np.unique(y_true_binary)',np.unique(labels))
    label_one_cls = label_binarize(labels, classes=np.unique(labels))
    print(f'label_one_cls shape {label_one_cls.shape}, probs {probs.shape}')
    if len(np.unique(labels)) == 2:
         probs = probs[:, 2]
    sklearn_macro_auc = roc_auc_score(label_one_cls, probs, average = 'macro', multi_class = 'ovr')
    print('@@@@@@#######',label_one_cls.shape, probs.shape)
    sklearn_weighted_auc = roc_auc_score(label_one_cls, probs, average = 'weighted', multi_class = 'ovr')
    print(f'sklearn_macro_auc is {sklearn_macro_auc}, sklearn_weighted_auc is {sklearn_weighted_auc}')
    # print(f'Micro-Averaged AUC: {micro_auc:.2f}')
    plt.figure(figsize=(8, 6))
    for i in range(3):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('One-vs-All ROC Curve')
    plt.legend(loc='lower right')
    os.makedirs(test_out_dir, exist_ok=True)
    plt.savefig(os.path.join(test_out_dir, 'roc.jpg'),dpi=800)
    test_out_dir = os.path.join(test_out_dir,"all_metric.txt")

    # 将print输出重定向到文件
    with open(test_out_dir, "w") as f:
        # 输出 confusion matrix 和 classification report
        f.write(f'confusion_matrix:\n {cm} \n')
        f.write('classfication report is' + cr + '\n')
        f.write(f'multi_spe:\n {multi_spe} \n')
        # 输出 Macro-Averaged AUC 和 roc_auc
        f.write(f'Macro-Averaged AUC: {macro_auc:.2f}\n')
        f.write('the roc_auc is' + str(roc_auc) + '\n')
    print(f'confusion_matrix:\n {cm} \n')
    print('classfication report is',cr)
    print(f'Macro-Averaged AUC: {macro_auc:.2f}')
    print('the roc_auc is',roc_auc)
    score =  (sklearn_macro_auc + sklearn_weighted_auc)/2
    return score

def cls_sigmoid_default(metric_fun,results, labels ,paths, global_rank,test_out_dir,**kwargs):
    need_failure = kwargs.get('need_failure', True)
    labels = labels.astype(int)
    probs = np_soft_max( results ,axis=1)
    preds = np.argmax( probs , axis =1)
    cm = confusion_matrix(labels, preds)
    cr = classification_report(y_true=labels, y_pred=preds)
    print(f'confusion_matrix:\n {cm} \n')
    print(cr)
    if len(paths) > 0 and global_rank == 0:  # Only  gpu 0 :
        res_dict = {'preds': preds,
                    'gts': labels,
                    'probs': probs,
                    'paths': np.array(paths)}
        eval_model_cls(res_dict, test_out_dir,need_failure)
    score = metric_fun(labels, preds, probs)
    return score

def reg_default(metric_fun,results, labels ,paths, global_rank,test_out_dir,**kwargs):
    """ For regression task"""
    need_failure = kwargs.get('need_failure', 0.5)
    failure_threshold = kwargs.get('failure_threshold', 0.5)

    csv = np.concatenate([results[:,None],labels[:,None]],axis=1)
    csv = pd.DataFrame(csv,columns=['preds','gts'])
    csv['diff'] =  csv['gts'] - csv['preds']
    csv['abs_diff'] = csv['diff'].abs()

    # 输出日志
    mean_score = csv['diff'].abs().mean()
    diff_str = csv['diff'].describe() 
    diff_abs_str = csv['abs_diff'].describe()
    pred_str =  csv['preds'].describe()

    txt = f'The average gap: {mean_score}\n\n'
    txt += f'Prediction distribution:\n{pred_str}\n\n'
    txt += f'Absolute gap distribution:\n{diff_abs_str}\n\n' 
    txt += f'Gap distribution:\n{diff_str}\n\n' 
    print(txt)

    if len(paths) > 0 and global_rank == 0:  # Only  gpu 0 :
        csv.insert(0,'paths',np.array(paths))
        res_dict = {'csv': csv,'txt':txt }
        eval_model_reg(res_dict,test_out_dir,need_failure,failure_threshold)
    score = metric_fun(labels,results)
    return score

def reg_int_label(metric_fun,results, labels ,paths, global_rank,test_out_dir,**kwargs):
    """ For regression task,and the label is int"""
    need_failure = kwargs.get('need_failure', True)
    preds = np.round(results).astype(int)
    labels = labels.astype(int)
    final_score = metric_fun(labels, preds)
    cm = confusion_matrix(labels, preds)
    cr = classification_report(y_true=labels, y_pred=preds)
    print(f'confusion_matrix:\n {cm} \n')
    print(cr)
    num_sample = results.shape[0]
    if len(paths) > 0 and global_rank == 0:  # Only  gpu 0 :
        res_dict = {'preds': preds,
                    'gts': labels,
                    'probs': results.reshape(num_sample,-1),
                    'paths': np.array(paths)}
        eval_model_cls(res_dict, test_out_dir,need_failure) 
    return final_score

def empty_method(metric_define,results, labels ,paths, global_rank,test_out_dir):
    return 0
    
DEFINE_PROCESS = dict()
DEFINE_PROCESS ['reg_int_label'] = reg_int_label
DEFINE_PROCESS['bin_all_metric'] = bin_all_metric
DEFINE_PROCESS['cls_softmax_default'] = cls_softmax_default
DEFINE_PROCESS['cls_sigmoid_default'] = cls_sigmoid_default
DEFINE_PROCESS['reg_default'] = reg_default
DEFINE_PROCESS['empty_method'] = empty_method




