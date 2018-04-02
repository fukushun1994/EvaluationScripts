#coding: utf-8

import sys
import pickle
import collections
import numpy as np

result_file = sys.argv[1]
laebl_list_file = sys.argv[2]
with open(result_file, 'rb') as fp:
	data = pickle.load(fp)
with open(laebl_list_file, 'rb') as fp:
	laebl_list = pickle.load(fp)

macro_P, macro_R, macro_F = 0, 0, 0
total_TP, total_FP, total_FN = 0, 0, 0
exact = 0
null_exact, null_cnt = 0, 0
each_cnt = {}
for label in laebl_list:
	each_cnt[label] = {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0}

for d in data:
	TP, FP, FN = 0, 0, 0
	for p in d['pred']:
		if p in d['answer']:
			TP += 1
			each_cnt[p]['TP'] += 1
		else:
			FP += 1
			each_cnt[p]['FP'] += 1
	for a in d['answer']:
		if a not in d['pred']:
			FN += 1
			each_cnt[a]['FN'] += 1
	total_TP += TP
	total_FP += FP
	total_FN += FN
	if TP+FN == 0:	# 正解が空の場合
		if FP == 0:
			null_exact += 1
		null_cnt += 1
	else:
		P = TP / (TP + FP) if TP + FP > 0 else 0
		R = TP / (TP + FN)
		F = 2 * P * R / (P + R) if P + R > 0 else 0
		macro_P += P
		macro_R += R
		macro_F += F
		if FP + FN == 0:
			exact += 1
macro_P /= len(data)-null_cnt
macro_R /= len(data)-null_cnt
macro_F /= len(data)-null_cnt
exact /= len(data)-null_cnt
null_exact /= null_cnt

micro_P = total_TP / (total_TP + total_FP)
micro_R = total_TP / (total_TP + total_FN)
micor_F = 2 * micro_P * micro_R / (micro_P + micro_R)

print('マクロ平均(P, R, F)')
print('{}\t{}\t{}'.format(macro_P, macro_R, macro_F))
print('完全一致率（正解が空でない, 空）')
print('{}\t{}'.format(exact, null_exact))
print('マイクロ平均(P, R, F)')
print('{}\t{}\t{}'.format(micro_P, micro_R, micor_F))

print('\n各属性ごとの評価')
for label in laebl_list:
	TP = each_cnt[label]['TP']
	FP = each_cnt[label]['FP']
	FN = each_cnt[label]['FN']
	P = TP / (TP + FP) if TP + FP > 0 else float('nan')
	R = TP / (TP + FN) if TP + FN > 0 else float('nan')
	F = 2 * P * R / (P + R) if P + R > 0 else float('nan')
	print('{}\t{}\t{}\t{}'.format(label, P, R, F))

# 根拠抽出の評価
macro_P = collections.defaultdict(int)
macro_R = collections.defaultdict(int)
macro_F = collections.defaultdict(int)
total_correct = collections.defaultdict(int)
total_pred_cnt = collections.defaultdict(int)
cnt = collections.defaultdict(int)
N = 4
bleu = collections.defaultdict(float)
rouge = collections.defaultdict(float)

for d in data:
	for label in d['pred']:
		if label not in d['answer']:
			continue
		cnt[label] += 1
		pred_cnt = 0
		TP = 0
		for pz in d['rationale'][label]:
			pred_cnt += 1
			for interval in d['rationale_answer'][label]:
				if pz >= interval[0] and pz < interval[1]:
					TP += 1
		rationale_cnt = 0
		for interval in d['rationale_answer'][label]:
			rationale_cnt += interval[1] - interval[0]
		P = TP / pred_cnt if pred_cnt > 0 else 0
		R = TP / rationale_cnt if rationale_cnt > 0 else 0
		F = 2 * P * R / (P + R) if P+R > 0 else 0
		macro_P[label] += P
		macro_R[label] += R
		macro_F[label] += F
		#total_correct[label] += TP
		#total_pred_cnt[label] += pred_cnt

		system_ngrams = [[] for i in range(N)]
		reference_ngrams = [[] for i in range(N)]
		pred_z = d['rationale'][label]
		answer_z = d['rationale_answer'][label]
		for i in range(len(pred_z)):
			system_ngrams[0].append((pred_z[i],))
			for n in range(1, N):
				if i + n >= len(pred_z):
					break
				if pred_z[i+n] != pred_z[i+n-1]+1:
					break
				system_ngrams[n].append(tuple(pred_z[i:i+n+1]))
		for interval in answer_z:
			for i in range(interval[0], interval[1]):
				for n in range(min(interval[1] - i, N)):
					reference_ngrams[n].append(tuple(range(i, i+n+1)))

		tmp_bleu = 0.0
		tmp_rouge = 0.0
		for n in range(N):
			TP = 0
			for ngram in system_ngrams[n]:
				if ngram in reference_ngrams[n]:
					TP += 1
			if len(system_ngrams[n]) == 0:
				tmp_bleu += np.log(1/2)
			else:
				if n == 0:
					tmp_bleu += np.log(TP / len(system_ngrams[n]))
				else:
					tmp_bleu += np.log((TP+1) / (len(system_ngrams[n])+1))
			if len(reference_ngrams[n]) == 0:
				tmp_rouge += np.log(1/2)
			else:
				if n == 0:
					tmp_rouge += np.log(TP / len(reference_ngrams[n]))
				else:
					tmp_rouge += np.log((TP+1) / (len(reference_ngrams[n])+1))
		bleu[label] += np.exp(tmp_bleu / N)
		rouge[label] += np.exp(tmp_rouge / N)

print('\n根拠抽出評価')
print('\tP\tR\tF')
for label in laebl_list:
	if cnt[label] == 0:
		print('{}\tnan\tnan\tnan'.format(label))
	else:
		print('{}\t{}\t{}\t{}'.format(label, macro_P[label] / cnt[label], macro_R[label] / cnt[label], macro_F[label] / cnt[label]))

print('\n\tBLEU\tROUGE')
for label in laebl_list:
	print('{}\t{}\t{}'.format(label, bleu[label]/cnt[label] if cnt[label] > 0 else float('nan'), rouge[label]/cnt[label] if cnt[label] > 0 else float('nan')))



'''
print('\nマイクロ平均')
for label in laebl_list:
	if total_pred_cnt[label] == 0:
		print('{}\ttotal_pred_cnt is 0'.format(label))
	else:
		print('{}\t{}\t({}/{})'.format(label, total_correct[label] / total_pred_cnt[label], total_correct[label], total_pred_cnt[label]))
'''
