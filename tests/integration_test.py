#coding: utf-8

import json
import sys

eps = 1e-8

def compare_key_values(a, b, keys):
    for key in keys:
        if type(a[key]) == int or type(a[key]) == float:
            assert abs(a[key] - b[key]) < eps, '{}: abs({} - {}) > {}'.format(key, a[key], b[key], eps)
        else:
            assert a[key] == b[key], '{}: {} != {}'.format(key, a[key], b[key])


output_file = sys.argv[1]
answer_file = sys.argv[2]

with open(output_file, 'r') as fp:
    output = json.load(fp)
with open(answer_file, 'r') as fp:
    answer = json.load(fp)

assert output['label_list'] == answer['label_list']
label_list = output['label_list']

# comparison for classification
out_by_chunk = output['classification']['by_chunk']
ans_by_chunk = answer['classification']['by_chunk']

for out_chunk, ans_chunk in zip(out_by_chunk['each_chunk'], ans_by_chunk['each_chunk']):
    keys = ['TP', 'FP', 'FN', 'TN', 'precision', 'recall', 'F-measure', 'exact_match', 'partial_match']
    compare_key_values(out_chunk, ans_chunk, keys)

keys = ['TP', 'FP', 'FN', 'TN', 'macro_precision', 'macro_recall', 'macro_F-measure', 'micro_precision', 'micro_recall', 'micro_F-measure', 'exact_match', 'partial_match']
compare_key_values(out_by_chunk['total'], ans_by_chunk['total'], keys)

out_by_label = output['classification']['by_label']
ans_by_label = answer['classification']['by_label']
for label in range(len(label_list)):
    keys = ['TP', 'FP', 'FN', 'TN', 'precision', 'recall', 'F-measure']
    compare_key_values(out_by_label[label], ans_by_label[label], keys)

# comparison for evidence
out_evidence = output['evidence']
ans_evidence = answer['evidence']
for out_label, ans_label in zip(out_evidence, ans_evidence):
    for out_chunk, ans_chunk in zip(out_label['each_chunk'], ans_label['each_chunk']):
        assert (out_chunk is not None and ans_chunk is not None) or (out_chunk is None and ans_chunk is None)
        if not out_chunk:
            continue
        keys = ['TPs', 'FPs', 'FNs', 'precision', 'recall', 'F-measure', 'BLEU', 'ROUGE', 'exact_match', 'partial_match']
        compare_key_values(out_chunk, ans_chunk, keys)
    assert (out_label['total'] is not None and ans_label['total'] is not None) or (out_label['total'] is None and ans_label['total'] is None)
    if out_label['total'] is None:
        continue
    keys = ['TPs', 'FPs', 'FNs', 'macro_precision', 'macro_recall', 'macro_F-measure', 'macro_BLEU', 'macro_ROUGE', 'micro_precision', 'micro_recall', 'micro_F-measure', 'micro_BLEU', 'micro_ROUGE','exact_match', 'partial_match']
    compare_key_values(out_label['total'], ans_label['total'], keys)
