from math import nan
import os
import datetime
import json

"""Define argument parser class."""
import argparse
import glob
from pathlib import Path
import csv
from nltk.translate.bleu_score import sentence_bleu
import nltk
import numpy as np
import pandas as pd

from tqdm import tqdm
from chexpert_labeler.loader import Loader
from chexpert_labeler.stages import Extractor, Classifier, Aggregator
from chexpert_labeler.constants.constants import *

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, multilabel_confusion_matrix

extractor = Extractor('downstream_task/report_generation_and_vqa/chexpert_labeler/phrases/mention',
                        'downstream_task/report_generation_and_vqa/chexpert_labeler/phrases/unmention')
classifier = Classifier('downstream_task/report_generation_and_vqa/chexpert_labeler/patterns/pre_negation_uncertainty.txt',
                        'downstream_task/report_generation_and_vqa/chexpert_labeler/patterns/negation.txt',
                        'downstream_task/report_generation_and_vqa/chexpert_labeler/patterns/post_negation_uncertainty.txt')
aggregator = Aggregator(CATEGORIES)


def label(hypo_path, ref_path, output_path):
    """Label the provided report(s)."""
    # Load reports in place.
    hypo_loader = Loader(hypo_path, extract_impression=False)
    hypo_loader.load()

    # Extract observation mentions in place.
    extractor.extract(hypo_loader.collection)
    # Classify mentions in place.
    classifier.classify(hypo_loader.collection)
    # Aggregate mentions to obtain one set of labels for each report.
    hypo_labels = aggregator.aggregate(hypo_loader.collection)
    
    """Write labeled reports to specified path."""
    reports = hypo_loader.reports

    hypo_labeled_reports = pd.DataFrame({REPORTS: reports})
    for index, category in enumerate(CATEGORIES):
        hypo_labeled_reports[category] = hypo_labels[:, index]

    hypo_labeled_reports = hypo_labeled_reports[[REPORTS] + CATEGORIES][1:]#.fillna(0)
    hypo_labeled_reports.to_csv(output_path+"hypo_label.csv", index=False)  #column name del.

    ###########
    # Load reports in place.
    ref_loader = Loader(ref_path, extract_impression=False)
    ref_loader.load()

    # Extract observation mentions in place.
    extractor.extract(ref_loader.collection)
    # Classify mentions in place.
    classifier.classify(ref_loader.collection)
    # Aggregate mentions to obtain one set of labels for each report.
    ref_labels = aggregator.aggregate(ref_loader.collection)
    
    """Write labeled reports to specified path."""
    reports = ref_loader.reports

    ref_labeled_reports = pd.DataFrame({REPORTS: reports})
    for index, category in enumerate(CATEGORIES):
        ref_labeled_reports[category] = ref_labels[:, index]

    ref_labeled_reports = ref_labeled_reports[[REPORTS] + CATEGORIES][1:]#.fillna(0)
    ref_labeled_reports.to_csv(output_path+"ref_label.csv", index=False)  #column name del.
    return hypo_labeled_reports, ref_labeled_reports


def get_label_accuracy(df_hyp, df_ref):
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

    # mcm = multilabel_confusion_matrix(y_true, y_pred)
    # tn = mcm[:, 0, 0]
    # tp = mcm[:, 1, 1]
    # fn = mcm[:, 1, 0]
    # fp = mcm[:, 0, 1]

    # recall = tp / (tp + fn)
    # precision = tp / (tp + fp)

    # recall_lt = [x for x in recall if str(x) != 'nan']
    # precision_lt = [x for x in precision if str(x) != 'nan']
    
    # print("recall_lt", recall_lt)
    # print("precision_lt", precision_lt)
    # input("STOP")
    
    # all label에 대해 acc 계산을 위한 것
    df_all_matching = (df_hyp.fillna(4) == df_ref.fillna(4)).astype(int)
    del df_all_matching["Reports"]
    df_all_matching = np.array(df_all_matching)

    ## all용 precision, recall, f1 계산을 위한 df_ref_all, df_hyp_all
    df_ref_all = np.array(df_ref_pos1 + df_ref_0 + df_ref_neg1)
    df_hyp_all = np.array(df_hyp_pos1 + df_hyp_0 + df_hyp_neg1)
    df_all_matching_exclude_TN = (df_hyp == df_ref).astype(int)
    del df_all_matching_exclude_TN["Reports"]
    df_all_matching_exclude_TN = np.array(df_all_matching_exclude_TN)

    # print("df_all_matching_exclude_TN", df_all_matching_exclude_TN)

    # Accuarcy
    accuracy_pos1 = (df_ref_pos1 == df_hyp_pos1).sum() / df_ref_pos1.size
    accuracy_0 = (df_ref_0 == df_hyp_0).sum() / df_ref_0.size
    accuracy_neg1 = (df_ref_neg1 == df_hyp_neg1).sum() / df_ref_neg1.size
    accuracy_all = df_all_matching.sum() / df_all_matching.size

    # Precision
    precision_pos1 = precision_score(df_ref_pos1, df_hyp_pos1, average="micro", zero_division=0)
    precision_0 = precision_score(df_ref_0, df_hyp_0, average="micro", zero_division=0)
    precision_neg1 = precision_score(df_ref_neg1, df_hyp_neg1, average="micro", zero_division=0)
    precision_all = df_all_matching_exclude_TN.sum() / df_hyp_all.sum()

    # Recall
    recall_pos1 = recall_score(df_ref_pos1, df_hyp_pos1, average="micro", zero_division=0)
    recall_0 = recall_score(df_ref_0, df_hyp_0, average="micro", zero_division=0)
    recall_neg1 = recall_score(df_ref_neg1, df_hyp_neg1, average="micro", zero_division=0)
    recall_all = df_all_matching_exclude_TN.sum() / df_ref_all.sum()

    # F1
    f1_pos1 = f1_score(df_ref_pos1, df_hyp_pos1, average="micro", zero_division=0)
    f1_0 = f1_score(df_ref_0, df_hyp_0, average="micro", zero_division=0)
    f1_neg1 = f1_score(df_ref_neg1, df_hyp_neg1, average="micro", zero_division=0)
    f1_all = 2 / (1/precision_all + 1/recall_all)

    # AUROC
    auroc_pos1 = roc_auc_score(df_ref_pos1, df_hyp_pos1, average="micro")
    auroc_0 = roc_auc_score(df_ref_0, df_hyp_0, average="micro")
    auroc_neg1 = roc_auc_score(df_ref_neg1, df_hyp_neg1, average="micro")
    auroc_all = roc_auc_score(df_ref_all, df_hyp_all, average="micro")
    
    accuracy_all_list = []
    precision_all_list = []
    recall_all_list = []
    f1_all_list = []
    for i in range(df_all_matching.shape[1]):
        acc = df_all_matching[:,i].sum() / df_all_matching[:,i].size
        pcn = df_all_matching_exclude_TN[:,i].sum() / df_hyp_all[:,i].sum()
        rcl = df_all_matching_exclude_TN[:,i].sum() / df_ref_all[:,i].sum()
        f1 = 2 / (1/pcn + 1/rcl)
        accuracy_all_list.append(acc)
        precision_all_list.append(pcn)
        recall_all_list.append(rcl)
        f1_all_list.append(f1)
    

    accuracy_all_list = [x for x in accuracy_all_list if str(x) != 'nan']
    precision_all_list = [x for x in precision_all_list if str(x) != 'nan']
    recall_all_list = [x for x in recall_all_list if str(x) != 'nan']
    f1_all_list = [x for x in f1_all_list if str(x) != 'nan']

    return  (sum(accuracy_all_list)/len(accuracy_all_list), sum(precision_all_list)/len(precision_all_list), sum(recall_all_list)/len(recall_all_list), sum(f1_all_list)/len(f1_all_list))

def language_eval_bleu(model_recover_path, eval_model, preds):
    lst_bleu_1gram, lst_bleu_2gram, lst_bleu_3gram, lst_bleu_4gram, lst_cumulative_4gram = [], [], [], [], []

    reference_path = model_recover_path.split('.')[0]+str(eval_model)+'_gt.csv'
    hypothesis_path = model_recover_path.split('.')[0]+str(eval_model)+'.csv'
    
    with open(reference_path, 'w', newline='') as gt:
        with open(hypothesis_path, 'w', newline='') as gen:
            list_of_list_of_references, list_of_list_of_hypotheses = [], []

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

    labeled_hypothesis, labeled_reference = label(hypothesis_path, reference_path, output_path=model_recover_path.split('.')[0])
    metric_pos1 = get_label_accuracy(labeled_hypothesis, labeled_reference)

    return bleu_1gram, bleu_2gram, bleu_3gram, bleu_4gram, metric_pos1
