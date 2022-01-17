import os
import json
from nltk.translate.bleu_score import sentence_bleu
import nltk
import numpy as np
from tqdm import tqdm
import csv
from sklearn.metrics import confusion_matrix
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def language_eval_bleu(model_recover_path, eval_model, preds):
    lst_bleu_1gram = []
    lst_bleu_2gram = []
    lst_bleu_3gram = []
    lst_bleu_4gram = []
    lst_cumulative_4gram = []

    with open(model_recover_path.split('.')[0]+str(eval_model)+'_gt.csv', 'w', newline='') as gt:
        with open(model_recover_path.split('.')[0]+str(eval_model)+'.csv', 'w', newline='') as gen:
            list_of_list_of_references = []
            list_of_list_of_hypotheses = []

            for i, preds in tqdm(enumerate(preds), total=len(preds)):
                for key, value in preds.items():
                    if key == 'gt_caption':
                        reference = value

                    elif key == 'gen_caption':
                        candidate = value

                        gt_writer = csv.writer(gt)
                        gen_writer = csv.writer(gen)

                        gt_writer.writerow([str(reference)])
                        gen_writer.writerow([str(candidate)])

                reference = reference.split(' ')
                hypothesis = candidate.split(' ')

                references = [reference] # list of references for 1 sentence. #[[refe]]
                list_of_list_of_references.append(references)
                list_of_list_of_hypotheses.append(hypothesis)
                

            bleu_1gram = nltk.translate.bleu_score.corpus_bleu(list_of_list_of_references, list_of_list_of_hypotheses, weights=(1, 0, 0, 0))
            bleu_2gram = nltk.translate.bleu_score.corpus_bleu(list_of_list_of_references, list_of_list_of_hypotheses, weights=(0.5, 0.5, 0, 0))
            bleu_3gram = nltk.translate.bleu_score.corpus_bleu(list_of_list_of_references, list_of_list_of_hypotheses, weights=(0.33, 0.33, 0.33, 0))
            bleu_4gram = nltk.translate.bleu_score.corpus_bleu(list_of_list_of_references, list_of_list_of_hypotheses, weights=(0.25, 0.25, 0.25, 0.25))
            
            print(f'1-Gram BLEU: {bleu_1gram:.2f}')
            print(f'2-Gram BLEU: {bleu_2gram:.2f}')
            print(f'3-Gram BLEU: {bleu_3gram:.2f}')
            print(f'4-Gram BLEU: {bleu_4gram:.2f}')

            
    gt.close()
    gen.close()

    return bleu_1gram, bleu_2gram, bleu_3gram, bleu_4gram


def get_label_accuracy_v1(hypothesis, reference):
    df_hyp = pd.read_csv(hypothesis)
    df_ref = pd.read_csv(reference)

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


    # Accuarcy
    accuracy_pos1 = (df_ref_pos1 == df_hyp_pos1).sum() / df_ref_pos1.size
    accuracy_0 = (df_ref_0 == df_hyp_0).sum() / df_ref_0.size
    accuracy_neg1 = (df_ref_neg1 == df_hyp_neg1).sum() / df_ref_neg1.size

    # Precision
    precision_pos1 = precision_score(df_ref_pos1, df_hyp_pos1, average="micro")
    precision_0 = precision_score(df_ref_0, df_hyp_0, average="micro")
    precision_neg1 = precision_score(df_ref_neg1, df_hyp_neg1, average="micro")


    # Recall
    recall_pos1 = recall_score(df_ref_pos1, df_hyp_pos1, average="micro")
    recall_0 = recall_score(df_ref_0, df_hyp_0, average="micro")
    recall_neg1 = recall_score(df_ref_neg1, df_hyp_neg1, average="micro")

    # F1
    f1_pos1 = f1_score(df_ref_pos1, df_hyp_pos1, average="micro")
    f1_0 = f1_score(df_ref_0, df_hyp_0, average="micro")
    f1_neg1 = f1_score(df_ref_neg1, df_hyp_neg1, average="micro")

    return (accuracy_pos1, precision_pos1, recall_pos1, f1_pos1), (accuracy_0, precision_0, recall_0, f1_0), (accuracy_neg1, precision_neg1, recall_neg1, f1_neg1)



def get_label_accuracy_v2(target, reference):
    df_tgt = pd.read_csv(target)
    df_ref = pd.read_csv(reference)    
    positive_tgt = df_tgt.isin([1.0])
    negative_tgt = df_tgt.isin([0.0])
    ambi_tgt = df_tgt.isin([-1.0])

    positive_ref = df_ref.isin([1.0])
    negative_ref = df_ref.isin([0.0])
    ambi_ref = df_ref.isin([-1.0])

    all_result = (df_tgt == df_ref)

    acc_list = []
    pos_precision = []
    neg_precision = []
    amb_precision = []
    
    pos_recall = []
    neg_recall = []
    amb_recall = []

    all_precision_lt = []
    all_recall_lt = []

    for row in range(len(df_tgt)):
        if len(positive_ref.loc[row].unique()) != 1:
            positive_precision = precision_score(positive_ref.loc[row],positive_tgt.loc[row], average="binary", pos_label=True)
            positive_recall = recall_score(positive_ref.loc[row],positive_tgt.loc[row], average="binary", pos_label=True)
            pos_precision.append(positive_precision)
            pos_recall.append(positive_recall)

        if len(negative_ref.loc[row].unique()) != 1:
            negative_precision = precision_score(negative_ref.loc[row],negative_tgt.loc[row], average="binary", pos_label=True)
            negative_recall = recall_score(negative_ref.loc[row],negative_tgt.loc[row], average="binary", pos_label=True)
            neg_precision.append(negative_precision)
            neg_recall.append(negative_recall)


        if len(ambi_ref.loc[row].unique()) != 1:
            ambi_precision = precision_score(ambi_ref.loc[row],ambi_tgt.loc[row], average="binary", pos_label=True)
            ambi_recall = recall_score(ambi_ref.loc[row],ambi_tgt.loc[row], average="binary", pos_label=True)
            amb_precision.append(ambi_precision)
            amb_recall.append(ambi_recall)

        acc_for_every_class = accuracy_score(df_ref.iloc[row,1:].fillna(0).values, df_tgt.iloc[row,1:].fillna(0).values)
        
        all_precision = precision_score(df_ref.iloc[row,1:].fillna(0).values, df_tgt.iloc[row,1:].fillna(0).values, average='macro')
        all_recall = recall_score(df_ref.iloc[row,1:].fillna(0).values, df_tgt.iloc[row,1:].fillna(0).values, average='macro')

        acc_list.append(acc_for_every_class)
        all_precision_lt.append(all_precision)
        all_recall_lt.append(all_recall)

    acc_array = np.mean(acc_list)
    pos_precision = np.mean(pos_precision)
    pos_recall = np.mean(pos_recall)
    neg_precision = np.mean(neg_precision)
    neg_recall = np.mean(neg_recall)
    amb_precision = np.mean(amb_precision)
    amb_recall = np.mean(amb_recall)
    all_precision_lt = np.mean(all_precision_lt)
    all_recall_lt = np.mean(all_recall_lt)
    return acc_array, pos_precision, pos_recall, neg_precision, neg_recall, amb_precision, amb_recall, all_precision_lt, all_recall_lt


if __name__ == "__main__":

    metric_pos1, metric_0, metric_neg1 = get_label_accuracy_v1(hypothesis = '/home/jhmoon/NegBio/chexpert-labeler/small_sc_50ep_4baem.csv', reference = '/home/jhmoon/NegBio/chexpert-labeler/base_sc_30ep_4beam_gt.csv')
    print("(micro) accuracy, precision, recall, f1 for pos1: {}, {}, {}, {}".format(round(metric_pos1[0], 4), round(metric_pos1[1], 4), round(metric_pos1[2], 4), round(metric_pos1[3], 4)))
    print("(micro) accuracy, precision, recall, f1 for zero: {}, {}, {}, {}".format(round(metric_0[0], 4), round(metric_0[1], 4), round(metric_0[2], 4), round(metric_0[3], 4)))
    print("(micro) accuracy, precision, recall, f1 for neg1: {}, {}, {}, {}".format(round(metric_neg1[0], 4), round(metric_neg1[1], 4), round(metric_neg1[2], 4), round(metric_neg1[3], 4)))


    # # target, reference
    acc_array, pos_precision, pos_recall, neg_precision, neg_recall, amb_precision, amb_recall, all_precision_lt, all_recall_lt \
     = get_label_accuracy_v2('/home/jhmoon/NegBio/chexpert-labeler/small_sc_50ep_4baem.csv', '/home/jhmoon/NegBio/chexpert-labeler/base_sc_30ep_4beam_gt.csv')

    print("accuracy",round(acc_array,3))
    print("all_precision_lt",round(all_precision_lt,3))
    print("all_recall_lt",round(all_recall_lt,3))
    print("pos_precision",round(pos_precision,3))
    print("pos_recall",round(pos_recall,3))
    print("neg_precision",round(neg_precision,3))
    print("neg_recall",round(neg_recall,3))
    print("amb_precision",round(amb_precision,3))
    print("amb_recall",round(amb_recall,3))
    
    # print(b)

    
    