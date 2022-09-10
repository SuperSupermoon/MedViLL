"""
Entry-point script to label radiology reports.

# python label.py --reports_path reports.py --output_path report_labeled.py
"""
from math import nan
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import datetime

"""Define argument parser class."""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from helpers import *

from chexpert_labeler.loader import Loader
from chexpert_labeler.stages import Extractor, Classifier, Aggregator
from chexpert_labeler.constants.constants import *

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score


def label_usecolumn(args, usecolumn, output_path):
    """Label the provided report(s)."""

    loader = Loader(args.reports_path, args.extract_impression, usecolumn)

    extractor = Extractor(args.mention_phrases_dir,
                          args.unmention_phrases_dir,
                          verbose=args.verbose)
    classifier = Classifier(usecolumn, 
                            args.pre_negation_uncertainty_path,
                            args.negation_path,
                            args.post_negation_uncertainty_path,
                            verbose=args.verbose)
    aggregator = Aggregator(CATEGORIES,
                            verbose=args.verbose)

    # Load reports in place.
    loader.load()

    # Extract observation mentions in place.
    extractor.extract(loader.collection)
    # Classify mentions in place.
    classifier.classify(loader.collection)
    # Aggregate mentions to obtain one set of labels for each report.
    labels = aggregator.aggregate(loader.collection)
    
    
    """Write labeled reports to specified path."""
    reports = loader.reports

    verbose= args.verbose
    labeled_reports = pd.DataFrame({REPORTS: reports})
    for index, category in enumerate(CATEGORIES):
        labeled_reports[category] = labels[:, index]
    if verbose:
        print(f"Writing reports and labels to {output_path}.")
        
    labeled_reports = labeled_reports[[REPORTS] + CATEGORIES][1:]#.fillna(0)
    labeled_reports.to_csv(output_path, index=False)  #column name del.
    
    return labeled_reports


def label(args, output_path):
    """Label the provided report(s)."""

    loader = Loader(args.reports_path, args.extract_impression)

    extractor = Extractor(args.mention_phrases_dir,
                          args.unmention_phrases_dir,
                          verbose=args.verbose)
    classifier = Classifier(args.pre_negation_uncertainty_path,
                            args.negation_path,
                            args.post_negation_uncertainty_path,
                            verbose=args.verbose)
    aggregator = Aggregator(CATEGORIES,
                            verbose=args.verbose)

    # Load reports in place.
    loader.load()

    # Extract observation mentions in place.
    extractor.extract(loader.collection)
    # Classify mentions in place.
    classifier.classify(loader.collection)
    # Aggregate mentions to obtain one set of labels for each report.
    labels = aggregator.aggregate(loader.collection)
    
    
    """Write labeled reports to specified path."""
    reports = loader.reports

    verbose= args.verbose
    labeled_reports = pd.DataFrame({REPORTS: reports})
    for index, category in enumerate(CATEGORIES):
        labeled_reports[category] = labels[:, index]
    if verbose:
        print(f"Writing reports and labels to {output_path}.")
        
    labeled_reports = labeled_reports[[REPORTS] + CATEGORIES][1:]#.fillna(0)
    labeled_reports.to_csv(output_path, index=False)  #column name del.
    
    return labeled_reports


def get_label_accuracy_v4(hypothesis, reference):
    df_hyp = pd.read_csv(hypothesis)
    df_ref = pd.read_csv(reference)
    # positive label에 대해 계산
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
    '''
    precision fall_out fpr 정밀도 
    recall sensitivity, tpr 재현율
    auc(fpr, tpr)
    '''    #There is way to compute weighted AUC. compare it with TieNet results.
    auroc_pos1 = roc_auc_score(df_ref_pos1, df_hyp_pos1, average="micro")
    auroc_0 = roc_auc_score(df_ref_0, df_hyp_0, average="micro")
    auroc_neg1 = roc_auc_score(df_ref_neg1, df_hyp_neg1, average="micro")
    auroc_all = roc_auc_score(df_ref_all, df_hyp_all, average="micro")
    
    
    
    # all에서 클래스별로 acc, precision, recall, f1구하기
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
    

    return  (accuracy_pos1, precision_pos1, recall_pos1, f1_pos1, auroc_pos1), \
            (accuracy_0, precision_0, recall_0, f1_0, auroc_0), \
            (accuracy_neg1, precision_neg1, recall_neg1, f1_neg1, auroc_neg1), \
            (accuracy_all, precision_all, recall_all, f1_all, auroc_all), \
            accuracy_all_list, precision_all_list, recall_all_list, f1_all_list




if __name__ == "__main__":
    start = datetime.datetime.now()
    print("\n", start, "\n")
    
    """Initialize argument parser."""
    # Input report parameters.
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--extract_impression', action='store_true', help='Extract the impression section of the report.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print progress to stdout.')
    # Phrases & Rules
    chex_base = "/home/dylee/__workspace/scale-up-transformer/chexpert_labeler/"
    parser.add_argument('--mention_phrases_dir', default=chex_base+'phrases/mention', help='Directory containing mention phrases for each observation.')
    parser.add_argument('--unmention_phrases_dir', default=chex_base+'phrases/unmention', help='Directory containing unmention phrases for each observation.')
    parser.add_argument('--pre_negation_uncertainty_path', default=chex_base+'patterns/pre_negation_uncertainty.txt', help='Path to pre-negation uncertainty rules.')
    parser.add_argument('--negation_path', default=chex_base+'patterns/negation.txt', help='Path to negation rules.')
    parser.add_argument('--post_negation_uncertainty_path', default=chex_base+'patterns/post_negation_uncertainty.txt', help='Path to post-negation uncertainty rules.')
    
    ### Output parameters.
    ### ! ###
    parser.add_argument('--do_label_gen', default=False, type=str2bool, help='If false, only config evaluation matric computation.')
    parser.add_argument('--do_label_gt', default=True, type=str2bool, help='If false, only config evaluation matric computation.')
    args = parser.parse_args()
    args.mention_phrases_dir = Path(args.mention_phrases_dir)
    args.unmention_phrases_dir = Path(args.unmention_phrases_dir)

    
    
    ######### ! #########
    # base_path = "/home/edlab/dylee/scaleup_transformer/i2t_Performers/mlm_20220116/"
    # base_path = "metadata/1to4/3of3_p/"
    base_path = "metadata/20220313/test/"
    
    # base_path = "metadata/1of1_256_4/clmmlm/"
    # args.reports_path = glob.glob(base_path+"test_*.csv")[-1]
    
    ##!## Chexpert label for the generated reports.
    args.reports_path = glob.glob(base_path+"GEN_*.csv")[-1]
    hypothesis_path = Path(base_path+"chexpert_labeled_reports_hypothesis.csv")
    if args.do_label_gen:
        # usecolumn = 0
        # labeled_hypothesis = label(args, usecolumn=usecolumn, output_path=hypothesis_path)
        labeled_hypothesis = label(args, output_path=hypothesis_path)

        print("Generated text labeling DONE!")
        print("\n",   args.reports_path)
        print("\nThe labeling hypothesis took for ", datetime.datetime.now() -start, "\n\n")        
        exit()
    else: pass
        # labeled_hypothesis = glob.glob(base_path+"labeled_GEN_*.csv")[-1]
        ##!## Chexpert label for the prime reports.
    args.reports_path = glob.glob(base_path+"GT_*.csv")[-1]
 
    if args.do_label_gt:
        # usecolumn = 1
        # labeled_reference = label(args, usecolumn, output_path=reference_path)
        labeled_reference = label(args, output_path=reference_path)
        
        print("Ground truth text labeling DONE!")
        print("\n",   args.reports_path)
        print("\nThe labeling reference took for ", datetime.datetime.now() -start, "\n\n")
        
    else: pass
        # labeled_reference = glob.glob(base_path+"labeled_GT_*.csv")[-1]
        
        
    
    ######### ! #########
    metric_pos1, metric_0, metric_neg1, metric_all, accuracy_all_list, precision_all_list, recall_all_list, f1_all_list = get_label_accuracy_v4(hypothesis = hypothesis_path, reference = reference_path)
    print(metric_all)
    print("(micro) accuracy, precision, recall, f1,  for ALL : {}, {}, {}, {}".format(round(metric_all[0], 3), round(metric_all[1], 3), round(metric_all[2], 3), round(metric_all[3], 3)))#, round(metric_all[4], 3)))
    print("(micro) accuracy, precision, recall, f1,  for pos1: {}, {}, {}, {}".format(round(metric_pos1[0], 3), round(metric_pos1[1], 3), round(metric_pos1[2], 3), round(metric_pos1[3], 3)))#, round(metric_pos1[4], 3)))
    print("(micro) accuracy, precision, recall, f1,  for zero: {}, {}, {}, {}".format(round(metric_0[0], 3), round(metric_0[1], 3), round(metric_0[2], 3), round(metric_0[3], 3)))#, round(metric_pos1[4], 3)))
    print("(micro) accuracy, precision, recall, f1,  for neg1: {}, {}, {}, {}".format(round(metric_neg1[0], 3), round(metric_neg1[1], 3), round(metric_neg1[2], 3), round(metric_neg1[3], 3)))#, round(metric_pos1[4], 3)))
    # print("(micro) auroc_multi_ovr", auroc_multi_ovr)
    
    ######
    acc_array, pos_precision, pos_recall, neg_precision, neg_recall, amb_precision, amb_recall, all_precision_lt, all_recall_lt = get_label_accuracy_v3(hypothesis_path, reference_path)
    print("\naccuracy",round(acc_array,3))
    print("\nall_precision_lt",round(all_precision_lt,3))
    print("all_recall_lt",round(all_recall_lt,3))
    print("\npos_precision",round(pos_precision,3))
    print("pos_recall",round(pos_recall,3))
    print("\nneg_precision",round(neg_precision,3))
    print("neg_recall",round(neg_recall,3))
    print("\namb_precision",round(amb_precision,3))
    print("amb_recall",round(amb_recall,3))


# wandb.init(project='ScaleUpTransformers_'+args.exp_name, dir=args.infer_save,  config=args)
# wandb.log 추가
