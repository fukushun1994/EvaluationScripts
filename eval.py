#coding: utf-8

import sys
import collections
import numpy as np
import json

'''
    output format
    {
        'label_list': [],
        'classification': {
            'by_chunk': {
                'each_chunk': [
                    {
                        'TP': ,
                        'FP': ,
                        'FN': ,
                        'TN': ,
                        'precision': ,
                        'recall': ,
                        'F-measure': ,
                        'exact_match': ,
                        'partial_match':
                    }
                ],
                'total': {
                    'TP': ,
                    'FP': ,
                    'FN': ,
                    'TN': ,
                    'macro_precision': ,
                    'macro_recall': ,
                    'macro_F-measure': ,
                    'micro_precision': ,
                    'micro_recall': ,
                    'micro_F-measure': ,
                    'exact_match': ,
                    'partial_match':
                }
            },
            'by_label': [
                {
                    'TP': ,
                    'FP': ,
                    'FN': ,
                    'TN': ,
                    'precision': ,
                    'recall': ,
                    'F-measure':
                }
            ]
        }
        'evidence': [
            {
                'each_chunk': [
                    {
                        'TPs': ,
                        'FPs': ,
                        'FNs': ,
                        'precision': ,
                        'recall': ,
                        'F-measure': ,
                        'BLEU': ,
                        'ROUGE': ,
                        'exact_match': ,
                        'partial_match':
                    }
                ],
                'total': {
                    'TPs': ,
                    'FPs': ,
                    'FNs': ,
                    'macro_precision': ,
                    'macro_recall': ,
                    'macro_F-measure': ,
                    'macro_BLEU': ,
                    'macro_ROUGE': ,
                    'micro_precision': ,
                    'micro_recall': ,
                    'micro_F-measure': ,
                    'micro_BLEU': ,
                    'micro_ROUGE': ,
                    'exact_match': ,
                    'partial_match':
                }
            }
        ]
    }
'''

# calculate precision, recall and F-measure from TP, FP and FN
def calc_PRF(TP, FP, FN):
    P = TP/(TP+FP) if TP+FP > 0 else 1.0
    R = TP/(TP+FN) if TP+FN > 0 else 1.0
    F = 2 * P * R / (P + R) if P+R > 0 else 0.0
    return P, R, F

# evaluate one chunk
def eval_classification_by_chunk(chunk):
    TP, FP, FN, TN = 0, 0, 0, 0
    for y_pred, y_answer in zip(chunk['y_pred'], chunk['y_answer']):
        if y_pred == 1:
            if y_answer == 1:
                TP += 1
            else:
                FP += 1
        else:
            if y_answer == 1:
                FN += 1
            else:
                TN += 1
    P, R, F = calc_PRF(TP, FP, FN)
    exact_match = 1 if F == 1.0 else 0
    partial_match = 1 if TP > 0 or F == 1.0 else 0
    return {
        'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
        'precision': P, 'recall': R, 'F-measure': F,
        'exact_match': exact_match, 'partial_match': partial_match
    }
# evaluate for one label
def eval_classification_by_label(results, label):
    TP, FP, FN, TN = 0, 0, 0, 0
    for chunk in results:
        pred = chunk['y_pred']
        ans = chunk['y_answer']
        if pred[label] == 1:
            if ans[label] == 1:
                TP += 1
            else:
                FP += 1
        else:
            if ans[label] == 1:
                FN += 1
            else:
                TN += 1
    P, R, F = calc_PRF(TP, FP, FN)
    return {
        'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN,
        'precision': P, 'recall': R, 'F-measure': F
    }

# create index ngrams from 0-1 array
# e.g. create_ngrams([1, 1, 1, 0, 1], 4) => [ [(0,), (1,), (2,), (4,)], [(0,1), (1,2)], [(0,1,2)], [] ]
def create_ngrams(array, N):
    res = [[] for i in range(N)]
    for i in range(len(array)):
        n = 0
        while n < N and i+n < len(array) and array[i+n] == 1:
            res[n].append(tuple(range(i, i+n+1)))
            n += 1
    return res

def calc_BLEU(N, TPs, FPs):
    BLEU = 0
    for n in range(N):
        if TPs[n]+FPs[n] == 0:
            BLEU += np.log(1/2)
        else:
            if n == 0:
                BLEU += np.log(TPs[n]/(TPs[n]+FPs[n]))
            else:
                BLEU += np.log((TPs[n]+1)/(TPs[n]+FPs[n]+1))
    BLEU = np.exp(BLEU/N)
    return BLEU

def calc_ROUGE(N, TPs, FNs):
    ROUGE = 0
    for n in range(N):
        if TPs[n]+FNs[n] == 0:
            ROUGE += np.log(1/2)
        else:
            if n == 0:
                ROUGE += np.log(TPs[n]/(TPs[n]+FNs[n]))
            else:
                ROUGE += np.log((TPs[n]+1)/(TPs[n]+FNs[n]+1))
    ROUGE = np.exp(ROUGE/N)
    return ROUGE

def evaluate_evidence(chunk, label, N):
    # evaluate only chunks which are classified correctly
    if chunk['y_pred'][label] != chunk['y_answer'][label] or chunk['y_answer'][label] != 1:
        return None
    pred = chunk['z_pred'][label]
    ans = chunk['z_answer'][label]

    system_ngrams = create_ngrams(pred, N)
    reference_ngrams = create_ngrams(ans, N)

    TPs = [0 for i in range(N)]
    FPs = [0 for i in range(N)]
    FNs = [0 for i in range(N)]
    for n in range(N):
        for ngram in system_ngrams[n]:
            if ngram in reference_ngrams[n]:
                TPs[n] += 1
        FPs[n] = len(system_ngrams[n]) - TPs[n]
        FNs[n] = len(reference_ngrams[n]) - TPs[n]
    P, R, F = calc_PRF(TPs[0], FPs[0], FNs[0])
    BLEU = calc_BLEU(N, TPs, FPs)
    ROUGE = calc_ROUGE(N, TPs, FNs)
    exact_match = 1 if F == 1.0 else 0
    partial_match = 1 if TPs[0] > 0 or F == 1.0 else 0
    return {
        'TPs': TPs, 'FPs': FPs, 'FNs': FNs,
        'precision': P, 'recall': R, 'F-measure': F,
        'BLEU': BLEU, 'ROUGE': ROUGE,
        'exact_match': exact_match, 'partial_match': partial_match
    }

if __name__ == '__main__':
    result_file = sys.argv[1]   # input file
    out_file = sys.argv[2]  # output file
    with open(result_file, 'r') as fp:
    	data = json.load(fp)

    label_list = data['label_list']
    results = data['results']

    label_num = len(label_list)
    chunk_num = len(results)

    # evaluation of classification
    classification = {'by_chunk': {}, 'by_label': []}
    ## by_chunk
    total_TP, total_FP, total_FN, total_TN = 0, 0, 0, 0
    macro_P, macro_R, macro_F = 0, 0, 0
    total_exact_match, total_partial_match = 0, 0
    each_chunk = []
    for chunk in results:
        evaluation = eval_classification_by_chunk(chunk)
        each_chunk.append(evaluation)
        # add each value to total value
        total_TP += evaluation['TP']
        total_FP += evaluation['FP']
        total_FN += evaluation['FN']
        total_TN += evaluation['TN']
        macro_P += evaluation['precision']
        macro_R += evaluation['recall']
        macro_F += evaluation['F-measure']
        total_exact_match += evaluation['exact_match']
        total_partial_match += evaluation['partial_match']

    micro_P, micro_R, micro_F = calc_PRF(total_TP, total_FP, total_FN)
    classification['by_chunk'] = {
        'each_chunk': each_chunk,
        'total': {
            'TP': total_TP, 'FP': total_FP, 'FN': total_FN, 'TN': total_TN,
            'macro_precision': macro_P / chunk_num,
            'macro_recall': macro_R / chunk_num,
            'macro_F-measure': macro_F / chunk_num,
            'micro_precision': micro_P,
            'micro_recall': micro_R,
            'micro_F-measure': micro_F,
            'exact_match': total_exact_match / chunk_num,
            'partial_match': total_partial_match / chunk_num
        }
    }
    ## by_label
    for label in range(label_num):
        evaluation = eval_classification_by_label(results, label)
        classification['by_label'].append(evaluation)

    # evaluation of evidence span identification
    evidence = []
    for label in range(label_num):
        N = 4
        total_TPs = [0 for i in range(N)]
        total_FPs = [0 for i in range(N)]
        total_FNs = [0 for i in range(N)]
        total_exact_match = 0
        total_partial_match = 0
        macro_P, macro_R, macro_F = 0, 0, 0
        macro_BLEU, macro_ROUGE = 0, 0
        each_chunk = []
        evidence_chunk_num = 0
        for chunk in results:
            evaluation = evaluate_evidence(chunk, label, N)
            if evaluation is not None:
                evidence_chunk_num += 1
                for n in range(N):
                    total_TPs[n] += evaluation['TPs'][n]
                    total_FPs[n] += evaluation['FPs'][n]
                    total_FNs[n] += evaluation['FNs'][n]
                total_exact_match += evaluation['exact_match']
                total_partial_match += evaluation['partial_match']
                macro_P += evaluation['precision']
                macro_R += evaluation['recall']
                macro_F += evaluation['F-measure']
                macro_BLEU += evaluation['BLEU']
                macro_ROUGE += evaluation['ROUGE']
            each_chunk.append(evaluation)
        micro_P, micro_R, micro_F = calc_PRF(total_TPs[0], total_FPs[0], total_FNs[0])
        micro_BLEU = calc_BLEU(N, total_TPs, total_FPs)
        micro_ROUGE = calc_ROUGE(N, total_TPs, total_FNs)

        if evidence_chunk_num > 0:
            total = {
                'TPs': total_TPs, 'FPs': total_FPs, 'FNs': total_FNs,
                'macro_precision': macro_P/evidence_chunk_num,
                'macro_recall': macro_R/evidence_chunk_num,
                'macro_F-measure': macro_F/evidence_chunk_num,
                'macro_BLEU': macro_BLEU/evidence_chunk_num,
                'macro_ROUGE': macro_ROUGE/evidence_chunk_num,
                'micro_precision': micro_P,
                'micro_recall': micro_R,
                'micro_F-measure': micro_F,
                'micro_BLEU': micro_BLEU,
                'micro_ROUGE': micro_ROUGE,
                'exact_match': total_exact_match/evidence_chunk_num,
                'partial_match': total_partial_match/evidence_chunk_num
            }
        else:
            total = None
        evidence.append({
            'each_chunk': each_chunk,
            'total': total
        })

    evaluation = {
        'label_list': label_list,
        'classification': classification,
        'evidence': evidence
    }

    with open(out_file, 'w') as fp:
        json.dump(evaluation, fp)
