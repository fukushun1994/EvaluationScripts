# EvaluationScripts
## Input Format
The input file must be dumped by `pickle`.

The input data is a dictionary which contains the results of classification and identification of the evidence spans.
The result for each segment is a dictionary object wihch contains `word_list`,`y_pred`, `y_answer`, `z_pred`, `z_answer`, `z_pred_text` and `z_answer_text` (see below).

```python
{
  'labels': ['label1, label2, ...'], # labels 
  'results':
    # results for each segment
    [
      # segment1
      {
        'word_list': ['word1', 'word2', ...], # words in the segment
        'y_pred': [0, 1, 1, 0, ...], # prediction of classification (#DB fields elements)
        'y_answer': [0, 1, 0, 0, ...], # correct answer of classification (#DB fields elements)
        'z_pred':
          # predictions of evidence spans for each DB field (#DB fields elements)
          [
            [0, 1, 1, 0, ...], # prediction for DB field 0 (#words elements)
            [1, 0, 0, 1, ...], # prediction for DB field 1 (#words elements)
            ...
          ],
        'z_answer':
          # correct answer of evidence spans for each DB field (#DB fields elements)
          [
            [0, 1, 1, 1, ...], # correct answer for DB field 0 (#words elements)
            [1, 1, 0, 1, ...], # correct answer for DB field 1 (#words elements)
            ...
          ],
        'z_pred_text':
          # space separated text of z_pred for each DB fieald (#DB fields elements)
          [
            '__ word2 word3 __ ...',
            'word1 __ __ word4 ...',
            ...
          ],
        'z_answer_text':
          # space separated text of z_answer for each DB fieald (#DB fields elements)
          [
            '__ word2 word3 word4 ...',
            'word1 word2 __ word4 ...',
            ...
          ]
      },
      # segment2
      {
        'word_list': ...
        'y_pred': ...
        'y_answer': ...
        'z_pred': ...
        'z_answer': ...
        'z_pred_text': ...
        'z_answer_text': ...
      },
      ...
    ]
}
```
