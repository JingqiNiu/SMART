from sklearn.metrics import confusion_matrix, cohen_kappa_score, roc_curve, auc
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, precision_score
from sklearn.metrics import mean_absolute_error
import inspect
import numpy as np
from .stats_fun_v2 import cal_ci95
import os
def get_binary_spec_recall(input_confusion, show=False):
    h, w = input_confusion.shape
    spec_recall = []
    assert h == w
    for i in range(h-1):
        tn = np.sum(input_confusion[:i+1, :i+1])
        fp = np.sum(input_confusion[:i+1, i+1:])
        tp = np.sum(input_confusion[i+1:, i+1:])
        fn = np.sum(input_confusion[i+1:, :i+1])

        confusion_matrix = np.array([[tn, fp], [fn, tp]])

        precision = tp/(tp+fp)
        accuracy = (tp+tn)/(tp+fp+tn+fn)
        specificity = tn/(tn+fp)
        recall = tp/(tp+fn)
        F1 = 2*(precision*recall)/(precision+recall)
        if show:
            l = [str(p) for p in range(h)]
            l.insert(i+1, '/')
            l = ''.join(l)
            print('Cutoff: ', l)
            print('Confusion Matrix: \n{}'.format(confusion_matrix))
            print('Precision {:.3f}, accuracy, {:.3f}, specificity, {:.3f}, recall {:.3f}, F1 {:.3f}.\n'.format(
                precision, accuracy, specificity, recall, F1))
        spec_recall.append([specificity, recall])
    return spec_recall

def calculate_specificity(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    return specificity
def bin_class_score(labels, preds_arg, preds_logit,*args, **kwargs):
    # print(f'labels unique {np.unique(labels)}, preds unique {np.unique(preds)}')
    # print(f'the labels shape {labels.shape}, labels {labels}, preds shape {preds.shape}, {preds}')
    test_out_dir = kwargs.get('test_out_dir', None)
    final_score = recall_score(labels, preds_arg, average='macro')

    auc_score = roc_auc_score(labels, preds_logit)
    one_hot_auroc = roc_auc_score(labels, preds_arg)
    acc = accuracy_score(labels , preds_arg)
    cm = confusion_matrix(labels, preds_arg)
    prec = precision_score(labels, preds_arg)
    rec = recall_score(labels, preds_arg)
    f1 = f1_score(labels, preds_arg)
    specificity = recall_score(labels, preds_arg, pos_label=0)
    fpr, tpr, thresholds = roc_curve(labels, preds_logit)
    ci95 = cal_ci95(labels, preds_logit)
    print(f'ci 95 is {ci95}')
    # 找到最佳阈值
    os.makedirs(test_out_dir ,exist_ok=True)
    test_out_dir = os.path.join(test_out_dir,"all_metric.txt")
    # 将print输出重定向到文件




    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # 计算Youden指数
    youden = tpr[optimal_idx] - fpr[optimal_idx]
    with open(test_out_dir, "w") as f:
        # 输出 confusion matrix 和 classification report
        f.write(f'confusion_matrix:\n {cm} \n')
        out_txt = f"youden {youden:.3f} thresholds {optimal_threshold:.3f}, Precision: , {round(prec, 3)}, Recall:(Sensitivity) {round(rec, 3)}, specificity {round(specificity, 3)} ,Recall macro, {round(final_score, 3)} ,F1-score: {round(f1, 3)}, AUROC {round(auc_score, 3)} , Accuary, {acc}, specificity {specificity}"
        f.write('youden' + str(out_txt) + '\n')
    print(f"youden {youden:.3f}, thresholds {optimal_threshold:.3f}", "Precision: ", round(prec, 3), "Recall:(Sensitivity) ", round(rec, 3), "specificity", round(specificity, 3) ,"Recall macro", round(final_score, 3),  "F1-score: ", round(f1, 3), "AUROC", round(auc_score, 3) , "Accuary", acc, "specificity",specificity)
    return auc_score #final_score

def cal_multicls_specificity(matrix):
    n_classes = len(matrix)
    specificity = {}
    for j in range(n_classes):
        numerator = matrix[:j, :j].sum() + matrix[j+1:, :j].sum() + matrix[:j, j+1:].sum() + matrix[j+1:, j+1:].sum()
        denominator = matrix[:j, :].sum() + matrix[j+1:, :].sum()
        specificity[j] = numerator / denominator
def cal_binary_metric(cm):
    # TP是第二行第二列的值
    TP = cm[1, 1]
    # TN是第一行第一列的值
    TN = cm[0, 0]
    # FP是第一行第二列的值
    FP = cm[0, 1]
    # FN是第二行第一列的值
    FN = cm[1, 0]


    # youden = (TP / (TP + FN)) + (TN / (TN + FP)) - 1
    youden = (TP / (TP + FN)) + (TN / (TN + FP)) - 1
    print(youden)

    # recall = TP / (TP + FN)
    recall = TP / (TP + FN)
    print(recall)

    # specificity = TN / (TN + FP)
    specificity = TN / (TN + FP)
    return youden,recall,specificity,   {'youden':youden, 'recall':recall, 'specificity':specificity}
def default_multi_class(labels, preds):
    import pandas as pd
    print('the unique labels is',np.unique(labels))
    cm = confusion_matrix(labels, preds)
    # multi_cm = pd.DataFrame(cm, columns=['pre_0', 'pre_1', 'pre_2', 'pre_3'], index=['label_0', 'label_1', 'label_2', 'label_3'])
    multi_spe = cal_multicls_specificity(cm)
    print('the spe for each cls is',multi_spe)
    sum_lower_right = cm[1:, 1:].sum()
 
    # 第一行右边三列求和
    sum_right = cm[0, 1:].sum()

    # 第一列下面三行求和
    sum_bottom = cm[1:, 0].sum()

    # 构造2x2输出矩阵
    binary_cm = np.array([[cm[0,0], sum_right],
                            [sum_bottom, sum_lower_right]])
    binary_cm_csv = pd.DataFrame(binary_cm, columns=[ 'pre_0', 'pre_1'], index=['label_0', 'label_1'])
    youden,recall,specificity,dict_ = cal_binary_metric(binary_cm)
    print(dict_)
    binary_cm_csv.loc[0, 'youden'] = youden
    binary_cm_csv.loc[0, 'recall'] = recall
    binary_cm_csv.loc[0, 'specificity'] = specificity
    print('the binary cm is', binary_cm)
    f1 = f1_score(labels, preds, average='macro')
    kappa = cohen_kappa_score(labels, preds,  weights='quadratic')
    final_score = (f1 + kappa) / 2.0
    print('the f1 score is',f1)
    print('the kappa score is',kappa)
    return final_score

def kappa_score(labels, preds):
    preds = np.round(preds).astype(int)
    kappa = cohen_kappa_score(labels, preds,  weights='quadratic')
    return kappa

def dr_metric_score(labels, preds):
    cm = confusion_matrix(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    spc_recal = get_binary_spec_recall(cm)
    kappa = cohen_kappa_score(labels, preds,  weights='quadratic')
    cut_score = (spc_recal[0][0] + spc_recal[1][0] +
                 spc_recal[1][1] + spc_recal[3][1]) / 4.0
    final_score = (f1 + kappa + cut_score) / 3.0
    return final_score
    
def mae(labels, preds):
    return -mean_absolute_error(preds , labels)

DEFINE_SCORE = dict()
DEFINE_SCORE['default'] = default_multi_class
DEFINE_SCORE['default_bin'] = bin_class_score
DEFINE_SCORE['dr'] = dr_metric_score
DEFINE_SCORE['mae'] = mae
DEFINE_SCORE['kappa'] = kappa_score
