import json
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def get_labels_and_frequencies(path):
    label_freqs = Counter()
    data_labels = [json.loads(line)["label"] for line in open(path)]
    if type(data_labels) == list:
        for label_row in data_labels:
            if label_row == '':
                label_row = ["'Others'"]
            else:
                label_row=label_row.split(', ')  
                
            label_freqs.update(label_row)
    else: pass
    return list(label_freqs.keys()), label_freqs

def get_label_accuracy(hypothesis, reference):
    df_hyp = pd.read_csv('/home/jhmoon/NegBio/chexpert-labeler/'+hypothesis)
    df_ref = pd.read_csv('/home/jhmoon/NegBio/chexpert-labeler/'+reference)
    df_hyp_pos1 = (df_hyp == 1).astype(int)
    del df_hyp_pos1["Reports"]
    df_hyp_pos1 = np.array(df_hyp_pos1)
    
    df_ref_pos1 = (df_ref == 1).astype(int)
    del df_ref_pos1["Reports"]
    df_ref_pos1 = np.array(df_ref_pos1)
    df_hyp_0 = (df_hyp == 0).astype(int)
    del df_hyp_0["Reports"]
    df_hyp_0 = np.array(df_hyp_0)
    df_ref_0 = (df_ref == 0).astype(int)
    del df_ref_0["Reports"]
    df_ref_0 = np.array(df_ref_0)
    df_hyp_neg1 = (df_hyp == -1).astype(int)
    del df_hyp_neg1["Reports"]
    df_hyp_neg1 = np.array(df_hyp_neg1)
    df_ref_neg1 = (df_ref == -1).astype(int)
    del df_ref_neg1["Reports"]
    df_ref_neg1 = np.array(df_ref_neg1)
    df_hyp_all = df_hyp_pos1 + df_hyp_0 + df_hyp_neg1
    df_ref_all = df_ref_pos1 + df_ref_0 + df_ref_neg1

    # Accuarcy
    accuracy_pos1 = (df_ref_pos1 == df_hyp_pos1).sum() / df_ref_pos1.size
    accuracy_0 = (df_ref_0 == df_hyp_0).sum() / df_ref_0.size
    accuracy_neg1 = (df_ref_neg1 == df_hyp_neg1).sum() / df_ref_neg1.size
    accuracy_all = (df_ref_all == df_hyp_all).sum() / df_ref_all.size

    # Precision
    precision_pos1 = precision_score(df_ref_pos1, df_hyp_pos1, average="micro")
    precision_0 = precision_score(df_ref_0, df_hyp_0, average="micro")
    precision_neg1 = precision_score(df_ref_neg1, df_hyp_neg1, average="micro")
    precision_all = precision_score(df_ref_all, df_hyp_all, average="micro")

    # Recall
    recall_pos1 = recall_score(df_ref_pos1, df_hyp_pos1, average="micro")
    recall_0 = recall_score(df_ref_0, df_hyp_0, average="micro")
    recall_neg1 = recall_score(df_ref_neg1, df_hyp_neg1, average="micro")
    recall_all = recall_score(df_ref_all, df_hyp_all, average="micro")

    # F1
    f1_pos1 = f1_score(df_ref_pos1, df_hyp_pos1, average="micro")
    f1_0 = f1_score(df_ref_0, df_hyp_0, average="micro")
    f1_neg1 = f1_score(df_ref_neg1, df_hyp_neg1, average="micro")
    f1_all = f1_score(df_ref_all, df_hyp_all, average="micro")

    return (accuracy_pos1, precision_pos1, recall_pos1, f1_pos1), (accuracy_0, precision_0, recall_0, f1_0), (accuracy_neg1, precision_neg1, recall_neg1, f1_neg1), (accuracy_all, precision_all, recall_all, f1_all)
    
if __name__ == '__main__':
    metric_pos1, metric_0, metric_neg1, metric_all = get_label_accuracy(hypothesis = 'small_sc_50ep_4baem.csv', reference = 'base_sc_30ep_4beam_gt.csv')
    print("(micro) accuracy, precision, recall, f1 for all : {}, {}, {}, {}".format(round(metric_all[0], 4), round(metric_all[1], 4), round(metric_all[2], 4), round(metric_all[3], 4)))
    print("(micro) accuracy, precision, recall, f1 for postive: {}, {}, {}, {}".format(round(metric_pos1[0], 4), round(metric_pos1[1], 4), round(metric_pos1[2], 4), round(metric_pos1[3], 4)))
    print("(micro) accuracy, precision, recall, f1 for negative: {}, {}, {}, {}".format(round(metric_0[0], 4), round(metric_0[1], 4), round(metric_0[2], 4), round(metric_0[3], 4)))
    print("(micro) accuracy, precision, recall, f1 for ambi: {}, {}, {}, {}".format(round(metric_neg1[0], 4), round(metric_neg1[1], 4), round(metric_neg1[2], 4), round(metric_neg1[3], 4)))