#coding: utf-8

import sys, json
import numpy as np

result_file = sys.argv[1]
aspect = int(sys.argv[2])
out_file_name = sys.argv[3]

decoder = json.JSONDecoder()

evaluation_result = {}

total_P, total_R, total_F, total_Acc, exact = 0.0, 0.0, 0.0, 0.0, 0.0
null_total_Acc, null_exact = 0, 0
cnt, null_cnt = 0, 0
total_selected = 0
total_selected_portion = 0.0
each_cnt = [{'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0} for i in range(38)]
total_correct = 0
total_P_z, total_R_z, total_F_z = 0.0, 0.0, 0.0
select_cnt = 0

with open(result_file, 'r') as fp:
	lines = fp.readlines()
	for l in lines:
		data = decoder.decode(l)
		y = data['y']
		pred_y = [1.0 if v > 0.5 else 0.0 for v in data['p_r']]
		TP, FP, FN, TN = 0, 0, 0, 0
		for i in range(len(y)):
			if pred_y[i] == 1.0:
				if y[i] == 1.0:
					TP += 1
					each_cnt[i]['TP'] += 1
				else:
					FP += 1
					each_cnt[i]['FP'] += 1
			else:
				if y[i] == 1.0:
					FN += 1
					each_cnt[i]['FN'] += 1
				else:
					TN += 1
					each_cnt[i]['TN'] += 1
		if max(y) == 0:
			null_cnt += 1
			null_total_Acc += (TP+TN)/(TP+FP+FN+TN)
			null_exact += (FP+FN==0)
		else:
			cnt += 1
			P = TP/(TP+FP) if TP+FP > 0 else 0.0
			R = TP/(TP+FN) if TP+FN > 0 else 0.0
			F = 2*P*R/(P+R) if P+R > 0 else 0.0
			Acc = (TP+TN)/(TP+FP+FN+TN)
			total_P += P
			total_R += R
			total_F += F
			total_Acc += Acc
			exact += (FP+FN == 0)


		# 根拠抽出の評価
		if aspect != -1 and (y[aspect] != 1.0 or pred_y[aspect] != 1.0):
			continue
		select_cnt += 1
		text = data['text'].split(' ')
		word_cnt = 0
		selected = 0
		for i in range(len(text)):
			if text[i] != '<padding>':
				word_cnt += 1
				if data['z'][i] == 1:
					selected += 1
		total_selected += selected
		total_selected_portion += selected / word_cnt
		TP = 0
		idx = 0
		for i in range(len(data['z'])):
			if text[i] == '<padding>':
				continue
			if data['z'][i] == 1:
				for interval in data['z_answer']:
					if idx >= interval[0] and idx < interval[1]:
						TP += 1
						break
			idx += 1
		rationale_cnt = 0
		for interval in data['z_answer']:
			rationale_cnt += interval[1] - interval[0]
		P = TP / selected if selected > 0 else 0
		R = TP / rationale_cnt if rationale_cnt > 0 else 0
		F = 2 * P * R / (P + R) if P+R > 0 else 0
		total_P_z += P
		total_R_z += R
		total_F_z += F

print('Precision\tRecall\tF-measure\tAccuracy\texact')
print('{}\t{}\t{}\t{}\t{}'.format(total_P/cnt, total_R/cnt, total_F/cnt, total_Acc/cnt, exact/cnt))
print('Accuracy\texact')
print('{}\t{}'.format(null_total_Acc/null_cnt, null_exact/null_cnt))
print('selected')
print(total_selected_portion / select_cnt if select_cnt > 0 else float('nan'))

print('\n根拠抽出評価')
print('P\tR\tF')
print('{}\t{}\t{}\n'.format(total_P_z / select_cnt if select_cnt > 0 else float('nan'), total_R_z / select_cnt if select_cnt > 0 else float('nan'), total_F_z / select_cnt if select_cnt > 0 else float('nan')))

evaluation_result['whole_P'] = total_P/cnt
evaluation_result['whole_R'] = total_R/cnt
evaluation_result['whole_F'] = total_F/cnt
evaluation_result['whole_exact'] = exact/cnt
evaluation_result['whole_null_exact'] = null_exact/null_cnt
evaluation_result['selected_portion'] = total_selected_portion / select_cnt if select_cnt > 0 else float('nan')
evaluation_result['selection_P'] = total_P_z / select_cnt if select_cnt > 0 else float('nan')
evaluation_result['selection_R'] = total_R_z / select_cnt if select_cnt > 0 else float('nan')
evaluation_result['selection_F'] = total_F_z / select_cnt if select_cnt > 0 else float('nan')

label_list = [
		"沿線",
		"駅徒歩",
		"駅利便性",
		"エリア",
		"周辺環境",
		"土地特徴",
		"目的地からの時間",
		"物件種別",
		"間取りタイプ",
		"賃料",
		"価格",
		"築後年数",
		"専有面積",
		"構造",
		"部屋の位置",
		"部屋の広さ",
		"日当たり・採光",
		"入居条件",
		"入居時期",
		"物件のターゲット",
		"状況",
		"室内設備",
		"冷暖房",
		"バス・トイレ",
		"キッチン",
		"テレビ・通信",
		"収納",
		"建物設備",
		"セキュリティ",
		"規模",
		"リフォーム・リノベーション",
		"階数",
		"表示情報",
		"お得条件",
		"所有権",
		"証明書類",
		"瑕疵保証",
		"資金面の優遇制度"
	]

if aspect == -1:
	evaluation_result['aspect_P'] = []
	evaluation_result['aspect_R'] = []
	evaluation_result['aspect_F'] = []
NaN = float('nan')
for i in range(38):
	if i == aspect or aspect == -1:
		TP, FP, FN = each_cnt[i]['TP'], each_cnt[i]['FP'], each_cnt[i]['FN']
		P = TP / (TP + FP) if TP+FP > 0 else NaN
		R = TP / (TP + FN) if TP+FN > 0 else NaN
		F = 2 * P * R / (P + R) if P+R > 0 else NaN
		print('{}\t{:.4}\t{:.4}\t{:.4}'.format(label_list[i], P, R, F))
		if aspect != -1:
			evaluation_result['aspect_P'] = P
			evaluation_result['aspect_R'] = R
			evaluation_result['aspect_F'] = F
		else:
			evaluation_result['aspect_P'].append(P)
			evaluation_result['aspect_R'].append(R)
			evaluation_result['aspect_F'].append(F)

#BLEU, ROUGE
N = 4
bleu = 0.0
rouge = 0.0
cnt = 0
with open(result_file, 'r') as fp:
	for l in fp:
		data = json.loads(l)
		y = data['y']
		pred_y = [1.0 if v > 0.5 else 0.0 for v in data['p_r']]
		if aspect != -1 and (y[aspect] != 1.0 or pred_y[aspect] != 1.0):
			continue
		cnt += 1
		system_ngrams = [[] for i in range(N)]
		reference_ngrams = [[] for i in range(N)]
		text = data['text'].split(' ')
		idx = 0
		for i in range(len(data['z'])):
			if text[i] == '<padding>':
				continue
			if data['z'][i] == 1:
				for n in range(N):
					if i + n >= len(data['z']):
						break
					if data['z'][i+n] != 1:
						break
					system_ngrams[n].append(tuple(range(idx, idx + n + 1)))
			idx += 1
		for interval in data['z_answer']:
			for i in range(interval[0], interval[1]):
				for n in range(min(interval[1] - i, N)):
					reference_ngrams[n].append(tuple(range(i, i+n+1)))

		tmp_bleu = 0.0
		tmp_rouge = 0.0
		TP = 0
		for n in range(N):
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
		bleu += np.exp(tmp_bleu / N)
		rouge += np.exp(tmp_rouge / N)

print('BLEU\tROUGE')
print('{}\t{}'.format(bleu / cnt if cnt > 0 else float('nan'), rouge / cnt if cnt > 0 else float('nan')))

evaluation_result['BLEU'] = bleu / cnt if cnt > 0 else float('nan')
evaluation_result['ROUGE'] = rouge / cnt if cnt > 0 else float('nan')

with open(out_file_name, 'w') as fp:
	json.dump(evaluation_result, fp)