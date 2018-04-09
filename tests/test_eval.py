import unittest
import eval

class TestEval(unittest.TestCase):
    def setUp(self):
        self.test_results = [
            {
                'y_pred':   [0, 1, 0, 1, 1],
                'y_answer': [0, 0, 1, 1, 0]
            },
            {
                'y_pred':   [0, 1, 0, 1, 0],
                'y_answer': [0, 0, 1, 1, 0]
            },
            {
                'y_pred':   [0, 1, 0, 1, 1],
                'y_answer': [0, 0, 1, 1, 1]
            },
            {
                'y_pred':   [0, 1, 0, 1, 0],
                'y_answer': [0, 0, 1, 1, 1]
            },
            {
                'y_pred':   [0, 1, 0, 1, 1],
                'y_answer': [0, 0, 1, 1, 1]
            }
        ]
        self.test_chunk = {
            'y_pred': [1, 1, 1, 1, 1, 0, 1],
            'y_answer': [1, 1, 1, 1, 1, 1, 0],
            'z_pred': [
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 0, 0, 1],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0]
            ],
            'z_answer': [
                [0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0]
            ]
        }
    # calc_PRF(TP, FP, FN)
    def test_calc_PRF1(self):
        self.assertEqual(eval.calc_PRF(0, 0, 0), (1, 1, 1))
    def test_calc_PRF2(self):
        self.assertEqual(eval.calc_PRF(10, 10, 10), (0.5, 0.5, 0.5))
    def test_calc_PRF3(self):
        self.assertEqual(eval.calc_PRF(10, 0, 0), (1, 1, 1))
    def test_calc_PRF4(self):
        self.assertEqual(eval.calc_PRF(0, 10, 0), (0, 1, 0))
    def test_calc_PRF5(self):
        self.assertEqual(eval.calc_PRF(0, 0, 10), (1, 0, 0))
    def test_calc_PRF6(self):
        self.assertEqual(eval.calc_PRF(0, 10, 10), (0, 0, 0))
    def test_calc_PRF7(self):
        self.assertEqual(eval.calc_PRF(10, 20, 40), (1/3, 1/5, 1/4))

    # eval_classification_by_chunk(chunk)
    def test_eval_classification_by_chunk1(self):
        chunk = {
            'y_pred':   [0, 0, 0, 0, 0],
            'y_answer': [0, 0, 0, 0, 0]
        }
        expect = {
            'TP': 0, 'FP': 0, 'FN': 0, 'TN': 5,
            'precision': 1, 'recall': 1, 'F-measure': 1,
            'exact_match': 1, 'partial_match': 1
        }
        self.assertEqual(eval.eval_classification_by_chunk(chunk), expect)
    def test_eval_classification_by_chunk2(self):
        chunk = {
            'y_pred':   [1, 1, 1, 1, 1],
            'y_answer': [0, 0, 0, 0, 0]
        }
        expect = {
            'TP': 0, 'FP': 5, 'FN': 0, 'TN': 0,
            'precision': 0, 'recall': 1, 'F-measure': 0,
            'exact_match': 0, 'partial_match': 0
        }
        self.assertEqual(eval.eval_classification_by_chunk(chunk), expect)
    def test_eval_classification_by_chunk3(self):
        chunk = {
            'y_pred':   [0, 0, 0, 0, 0],
            'y_answer': [1, 1, 1, 1, 1]
        }
        expect = {
            'TP': 0, 'FP': 0, 'FN': 5, 'TN': 0,
            'precision': 1, 'recall': 0, 'F-measure': 0,
            'exact_match': 0, 'partial_match': 0
        }
        self.assertEqual(eval.eval_classification_by_chunk(chunk), expect)
    def test_eval_classification_by_chunk4(self):
        chunk = {
            'y_pred':   [1, 1, 1, 1, 1],
            'y_answer': [1, 1, 1, 1, 1]
        }
        expect = {
            'TP': 5, 'FP': 0, 'FN': 0, 'TN': 0,
            'precision': 1, 'recall': 1, 'F-measure': 1,
            'exact_match': 1, 'partial_match': 1
        }
        self.assertEqual(eval.eval_classification_by_chunk(chunk), expect)
    def test_eval_classification_by_chunk5(self):
        chunk = {
            'y_pred':   [1, 0, 1, 0, 1],
            'y_answer': [0, 0, 1, 1, 1]
        }
        expect = {
            'TP': 2, 'FP': 1, 'FN': 1, 'TN': 1,
            'precision': 2/3, 'recall': 2/3, 'F-measure': 2/3,
            'exact_match': 0, 'partial_match': 1
        }
        self.assertEqual(eval.eval_classification_by_chunk(chunk), expect)

    # eval_classification_by_label(results, label)
    def test_eval_classification_by_label1(self):
        expect = {
            'TP': 0, 'FP': 0, 'FN': 0, 'TN': 5,
            'precision': 1, 'recall': 1, 'F-measure': 1
        }
        self.assertEqual(eval.eval_classification_by_label(self.test_results, 0), expect)
    def test_eval_classification_by_label2(self):
        expect = {
            'TP': 0, 'FP': 5, 'FN': 0, 'TN': 0,
            'precision': 0, 'recall': 1, 'F-measure': 0
        }
        self.assertEqual(eval.eval_classification_by_label(self.test_results, 1), expect)
    def test_eval_classification_by_label3(self):
        expect = {
            'TP': 0, 'FP': 0, 'FN': 5, 'TN': 0,
            'precision': 1, 'recall': 0, 'F-measure': 0
        }
        self.assertEqual(eval.eval_classification_by_label(self.test_results, 2), expect)
    def test_eval_classification_by_label4(self):
        expect = {
            'TP': 5, 'FP': 0, 'FN': 0, 'TN': 0,
            'precision': 1, 'recall': 1, 'F-measure': 1
        }
        self.assertEqual(eval.eval_classification_by_label(self.test_results, 3), expect)
    def test_eval_classification_by_label5(self):
        expect = {
            'TP': 2, 'FP': 1, 'FN': 1, 'TN': 1,
            'precision': 2/3, 'recall': 2/3, 'F-measure': 2/3
        }
        self.assertEqual(eval.eval_classification_by_label(self.test_results, 4), expect)

    # create_ngrams(array, N)
    def test_create_ngrams1(self):
        array = [1, 1, 1, 1, 1, 1]
        N = 4
        expect = [
            [(0,), (1,), (2,), (3,), (4,), (5,)],
            [(0,1), (1,2), (2,3), (3,4), (4,5)],
            [(0,1,2), (1,2,3), (2,3,4), (3,4,5)],
            [(0,1,2,3), (1,2,3,4), (2,3,4,5)]
        ]
        self.assertEqual(eval.create_ngrams(array, N), expect)
    def test_create_ngrams2(self):
        array = [1, 1, 0, 1, 1, 1]
        N = 4
        expect = [
            [(0,), (1,), (3,), (4,), (5,)],
            [(0,1), (3,4), (4,5)],
            [(3,4,5)],
            []
        ]
        self.assertEqual(eval.create_ngrams(array, N), expect)
    def test_create_ngrams3(self):
        array = [0, 0, 0, 0, 0, 0]
        N = 4
        expect = [[], [], [], []]
        self.assertEqual(eval.create_ngrams(array, N), expect)

    # calc_BLEU(N, TPs, FPs)
    def test_calc_BLEU1(self):
        N = 4
        TPs = [0, 0, 0, 0]
        FPs = [0, 0, 0, 0]
        self.assertEqual(eval.calc_BLEU(N, TPs, FPs), 0.5)
    def test_calc_BLEU2(self):
        N = 4
        TPs = [1, 1, 1, 1]
        FPs = [0, 0, 0, 0]
        self.assertEqual(eval.calc_BLEU(N, TPs, FPs), 1)
    def test_calc_BLEU3(self):
        N = 4
        TPs = [0, 0, 0, 0]
        FPs = [1, 1, 1, 1]
        self.assertEqual(eval.calc_BLEU(N, TPs, FPs), 0)
    def test_calc_BLEU4(self):
        N = 4
        TPs = [1, 1, 1, 1]
        FPs = [1, 1, 1, 1]
        self.assertAlmostEqual(eval.calc_BLEU(N, TPs, FPs), (4/27)**(1/4))
    def test_calc_BLEU5(self):
        N = 4
        TPs = [30, 20, 0, 0]
        FPs = [20, 10, 1, 0]
        self.assertAlmostEqual(eval.calc_BLEU(N, TPs, FPs), (3*21/(5*31*2*2))**(1/4))

    # calc_ROUGE(N, TPs, FPs)
    def test_calc_ROUGE1(self):
        N = 4
        TPs = [0, 0, 0, 0]
        FNs = [0, 0, 0, 0]
        self.assertEqual(eval.calc_ROUGE(N, TPs, FNs), 0.5)
    def test_calc_ROUGE2(self):
        N = 4
        TPs = [1, 1, 1, 1]
        FNs = [0, 0, 0, 0]
        self.assertEqual(eval.calc_ROUGE(N, TPs, FNs), 1)
    def test_calc_ROUGE3(self):
        N = 4
        TPs = [0, 0, 0, 0]
        FNs = [1, 1, 1, 1]
        self.assertEqual(eval.calc_ROUGE(N, TPs, FNs), 0)
    def test_calc_ROUGE4(self):
        N = 4
        TPs = [1, 1, 1, 1]
        FNs = [1, 1, 1, 1]
        self.assertAlmostEqual(eval.calc_ROUGE(N, TPs, FNs), (4/27)**(1/4))
    def test_calc_ROUGE5(self):
        N = 4
        TPs = [30, 20, 0, 0]
        FNs = [20, 10, 1, 0]
        self.assertAlmostEqual(eval.calc_ROUGE(N, TPs, FNs), (3*21/(5*31*2*2))**(1/4))

    # evaluate_evidence(chunk, label, N)
    def test_evaluate_evidence1(self):
        N = 4
        expect = {
            'TPs': [0, 0, 0, 0], 'FPs': [0, 0, 0, 0], 'FNs': [0, 0, 0, 0],
            'precision': 1, 'recall': 1, 'F-measure': 1,
            'BLEU': 0.5, 'ROUGE': 0.5,
            'exact_match': 1, 'partial_match': 1
        }
        self.assertEqual(eval.evaluate_evidence(self.test_chunk, 0, N), expect)
    def test_evaluate_evidence2(self):
        N = 4
        expect = {
            'TPs': [0, 0, 0, 0], 'FPs': [0, 0, 0, 0], 'FNs': [6, 5, 4, 3],
            'precision': 1, 'recall': 0, 'F-measure': 0,
            'BLEU': 0.5, 'ROUGE': 0,
            'exact_match': 0, 'partial_match': 0
        }
        self.assertEqual(eval.evaluate_evidence(self.test_chunk, 1, N), expect)
    def test_evaluate_evidence3(self):
        N = 4
        expect = {
            'TPs': [0, 0, 0, 0], 'FPs': [6, 5, 4, 3], 'FNs': [0, 0, 0, 0],
            'precision': 0, 'recall': 1, 'F-measure': 0,
            'BLEU': 0, 'ROUGE': 0.5,
            'exact_match': 0, 'partial_match': 0
        }
        self.assertEqual(eval.evaluate_evidence(self.test_chunk, 2, N), expect)
    def test_evaluate_evidence4(self):
        N = 4
        expect = {
            'TPs': [6, 5, 4, 3], 'FPs': [0, 0, 0, 0], 'FNs': [0, 0, 0, 0],
            'precision': 1, 'recall': 1, 'F-measure': 1,
            'BLEU': 1, 'ROUGE': 1,
            'exact_match': 1, 'partial_match': 1
        }
        self.assertEqual(eval.evaluate_evidence(self.test_chunk, 3, N), expect)
    def test_evaluate_evidence5(self):
        # pred: [1, 1, 1, 0, 0, 1]
        # and:  [0, 1, 1, 1, 1, 1]
        N = 4
        expect = {
            'TPs': [3, 1, 0, 0], 'FPs': [1, 1, 1, 0], 'FNs': [2, 3, 3, 2],
            'precision': 3/4, 'recall': 3/5, 'F-measure': 2/3,
            'BLEU': (1/8)**(1/4), 'ROUGE': (1/50)**(1/4),
            'exact_match': 0, 'partial_match': 1
        }
        result = eval.evaluate_evidence(self.test_chunk, 4, N)
        float_keys = ['precision', 'recall', 'F-measure', 'BLEU', 'ROUGE']
        for key in float_keys:
            self.assertAlmostEqual(result[key], expect[key])
            expect.pop(key)
            result.pop(key)
        self.assertEqual(result, expect)
    def test_evaluate_evidence6(self):
        N = 4
        self.assertIsNone(eval.evaluate_evidence(self.test_chunk, 5, N))
    def test_evaluate_evidence7(self):
        N = 4
        self.assertIsNone(eval.evaluate_evidence(self.test_chunk, 6, N))

if __name__ == '__main__':
    unittest.main()
