# EvaluationScripts
## Input Format
The input file must be dumped by `json`.

The input data is a dictionary which contains the results of classification and identification of the evidence spans.
The result for each chunk is a dictionary object wihch contains `word_list`,`y_pred`, `y_answer`, `z_pred`, `z_answer`, `z_pred_text` and `z_answer_text` (see below).

```python
{
  "label_list": ["label1", "label2", ...], # labels
  "results":
    # results for each segment
    [
      # chunk1
      {
        "word_list": ["word1", "word2", ...], # words in the chunk
        "y_pred": [0, 1, 1, 0, ...], # prediction of classification (#DB fields elements)
        "y_answer": [0, 1, 0, 0, ...], # correct answer of classification (#DB fields elements)
        "z_pred":
          # predictions of evidence spans for each DB field (#DB fields elements)
          [
            [0, 1, 1, 0, ...], # prediction for DB field 0 (#words elements)
            [1, 0, 0, 1, ...], # prediction for DB field 1 (#words elements)
            ...
          ],
        "z_answer":
          # correct answer of evidence spans for each DB field (#DB fields elements)
          [
            [0, 1, 1, 1, ...], # correct answer for DB field 0 (#words elements)
            [1, 1, 0, 1, ...], # correct answer for DB field 1 (#words elements)
            ...
          ],
        "z_pred_text":
          # space separated text of z_pred for each DB fieald (#DB fields elements)
          [
            "__ word2 word3 __ ...",
            "word1 __ __ word4 ...",
            ...
          ],
        "z_answer_text":
          # space separated text of z_answer for each DB fieald (#DB fields elements)
          [
            "__ word2 word3 word4 ...",
            "word1 word2 __ word4 ...",
            ...
          ]
      },
      # chunk2
      {
        "word_list": ... ,
        "y_pred": ... ,
        "y_answer": ... ,
        "z_pred": ... ,
        "z_answer": ... ,
        "z_pred_text": ... ,
        "z_answer_text": ...
      },
      ...
    ]
}
```

## Output Format
The output file is dumped by `json`.
```python
{
  "label_list": ["label1", "label2", ...],  #labels
  # evaluation result of classification
  "classification": {
    # evaluation of classification by chunk
    "by_chunk": {
      # for each chunk
        "each_chunk": [
          # chunk1
          {
            "TP": 2,  # the number of True Positive labels
            "FP": 2,  # the number of False Positive labels
            "FN": 2,  # the number of False Negative labels
            "TN": 0,  # the number of True Negative labels
            "precision": 0.5,
            "recall": 0.5,
            "F-measure": 0.5,
            "exact_match": 0, # whether the prediction are completly same with the correct answer
            "partial_match": 1  # whether the predicition are partialy same with the correct answer
          },
          # chunk2
          {
            ...
          },
          ...
        ],
        # for total chunks
        "total": {
            "TP": 5, # sum of TP
            "FP": 3, # sum of FP
            "FN": 3, # sum of FN
            "TN": 4, # sum of TN
            "macro_precision": 0.6666666666666666,  # macro average of precision
            "macro_recall": 0.5, # macro average of recall
            "macro_F-measure": 0.5555555555555556, # macro average of F-measure
            "micro_precision": 0.625, # micro average of precision
            "micro_recall": 0.625, # micro average of recall
            "micro_F-measure": 0.625, # micro average of F-measure
            "exact_match": 0.3333333333333333, # ratio of chunks where the prediction are completely same with the correct answer
            "partial_match": 0.6666666666666666 # ratio of chunks where the prediction are partialy same with the correct answer
        },
      # evaluation of classification for each label
      "by_label": [
        # label1
        {
          "TP": 0, # the number of True Positive chunks
          "FP": 0, # the number of False Positive chunks
          "FN": 2, # the number of False Negative chunks
          "TN": 1, # the number of True Negative chunks
          "precision": 1.0,
          "recall": 0.0,
          "F-measure": 0.0
        },
        # label2
        {
          ...
        },
        ...
      ]
    },
  },
  # evaluation result of evidence spans identification
  "evidence": [
    # for label1
    {
      # for each chunk
      "each_chunk": [
        # chunk1
        {
          "TPs": [2, 1, 0, 0],    # the numbers of True Positive unigram, bigram, trigram and 4-gram
            "FPs": [0, 0, 0, 0],  # the numbers of False Positive unigram, bigram, trigram and 4-gram
            "FNs": [0, 0, 0, 0],  # the numbers of False Negative unigram, bigram, trigram and 4-gram
            "precision": 1.0, # precision on words (unigrams)
            "recall": 1.0,    # recall on words (unigrams)
            "F-measure": 1.0, # F-measure on words (unigrams)
            "BLEU": 0.7071067811865476,   # BLEU up to 4-gram
            "ROUGE": 0.7071067811865476,  # ROUGE up to 4-gram
            "exact_match": 1, # whether the predicted evidence spans are completely same with the correct answer
            "partial_match": 1 # whether the predicted evidence spans are partialy same with the correct answer
        },
        # chunk2
        null, # the result becomes null if the chunk is a negative example of classification for label1 or the prediction of classification is wrong
        ...
      ],
      # for total chunk
      "total": {
          "TPs": [2, 1, 0, 0],  # sum of TPs
          "FPs": [0, 0, 0, 0],  # sum of FPs
          "FNs": [2, 1, 0, 0],  # sum of FNs
          "macro_precision": 1.0, # macro average of precision on words (unigrams)
          "macro_recall": 0.5,  # macro average of recall on words (unigrams)
          "macro_F-measure": 0.5, # macro average of F-measure on words (unigrams)
          "macro_BLEU": 0.6035533905932737, # macro average of BLEU up to 4-gram
          "macro_ROUGE": 0.3535533905932738,  # macro average of ROUGE up to 4-gram
          "micro_precision": 1.0, # micro average of precision on words (unigrams)
          "micro_recall": 0.5,  # micro average of recall on words (unigrams)
          "micro_F-measure": 0.6666666666666666,  # micro average of F-measure on words (unigrams)
          "micro_BLEU": 0.7071067811865476, # micro average of BLEU up to 4-gram
          "micro_ROUGE": 0.537284965911771, # micro average of ROUGE up to 4-gram
          "exact_match": 0.5, # ratio of chunks where the prediction are completely same with the correct answer
          "partial_match": 0.5  # ratio of chunks where the prediction are partialy same with the correct answer
      }
    },
    # for label2
    {
      "each_chunk": ... ,
      "total": ...
    },
    ...
  ]
}

```
