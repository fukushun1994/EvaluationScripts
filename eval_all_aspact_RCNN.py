#coding: utf-8

import sys, json

output_dir = sys.argv[1]
sparsity = sys.argv[2]
coherent = sys.argv[3]

each_segment = []

with open('{}/{}_{}_0.json'.format(output_dir, sparsity, coherent), 'r') as fp:
	for l in fp:
		each_segment.append({'TP': 0, 'FP':0, 'FN':0})


for i in range(38):
	with open('{}/{}_{}_{}.json'.format(output_dir, sparsity, coherent, i), 'r') as fp:
		lines = fp.readlines()
		for j, l in enumerate(lines):
			data = json.loads(l)
			y = data['y']
			pred_y = [1.0 if v > 0.5 else 0.0 for v in data['p_r']]
			if y[i] == 1.0:
				if pred_y[i] == 1.0:
					each_segment[j]['TP'] += 1
				else:
					each_segment[j]['FN'] += 1
			else:
				if pred_y[i] == 1.0:
					each_segment[j]['FP'] += 1

total_P, total_R, total_F, exact, cnt = 0, 0, 0, 0, 0
null_exact, null_cnt = 0, 0
for s in each_segment:
	TP, FP, FN = s['TP'], s['FP'], s['FN']
	if TP + FN > 0:
		P = TP / (TP + FP) if TP + FP > 0 else 0
		R = TP / (TP + FN) if TP + FN > 0 else 0
		F = 2 * P * R / (P + R) if P + R > 0 else 0
		total_P += P
		total_R += R
		total_F += F
		if FP + FN == 0:
			exact += 1
		cnt += 1
	else:
		if FP + FN == 0:
			null_exact += 1
		null_cnt += 1
print('P\tR\tF\texact')
print('{}\t{}\t{}\t{}'.format(total_P/cnt, total_R/cnt, total_F/cnt, exact/cnt))
print('null_exact')
print(null_exact/null_cnt)
