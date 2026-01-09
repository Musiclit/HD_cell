'''
Functions for linear regression, rarely used
'''
import pandas as pd
from statsmodels.stats.multitest import multipletests


def correct_pval(stat_df, pval_names, corp_names, method):
    pvals = stat_df[pval_names]
    _, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method=method)
    stat_df[corp_names] = pvals_corrected
    

def print_sensitivity_specificity(confusion_matrix):
    recall1, _, _ = cal_sensitivity_specificity(confusion_matrix, 'central')
    recall2, _, _ = cal_sensitivity_specificity(confusion_matrix, 'left')
    recall3, _, _ = cal_sensitivity_specificity(confusion_matrix, 'right')
    print(f'BALANCED ACCURACY: {(recall1[2] + recall2[2] + recall3[2]) / 3:.3f}')


def cal_sensitivity_specificity(confusion_matrix, ring):    
    cell_types = {'central': 'EPG', 'left': 'PEN_L', 'right': 'PEN_R'}
    ctype = cell_types[ring]

    recallN = confusion_matrix.loc[ring, ctype]
    recallD = confusion_matrix.loc[:, ctype].sum()
    recall = recallN / recallD

    specificityN = confusion_matrix.sum().sum() - confusion_matrix.loc[ring].sum() - confusion_matrix.loc[:,ctype].sum() + confusion_matrix.loc[ring,ctype]
    specificityD = confusion_matrix.sum().sum() - confusion_matrix.loc[:, ctype].sum()
    specificity = specificityN / specificityD

    print(f'{ring:15} Recall: {recallN:3d}/{recallD:3d} = {recall:.3f};   Specificity: {specificityN:3d}/{specificityD:3d} = {specificity:.3f};   Accuracy: {(recall + specificity) / 2:.3f}')
    return [recallN, recallD, recall], [specificityN, specificityD, specificity], (recall + specificity) / 2
