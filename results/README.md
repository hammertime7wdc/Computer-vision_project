# Results and testing evidence

Store final, reproducible evaluation artifacts here. Keep temporary caches,
notebooks outputs, and virtual environments outside this directory.

Recommended layout:

```text
results/
├── test_1/       Figure-8 line detection and tracking evidence
├── test_2/       Traffic-sign specialization evidence
├── metrics/      Precision, recall, mAP, confusion matrix
└── predictions/  Annotated sample predictions and command logs
```

## Required result record

For the checkpoint used in the final demonstration, add the dataset split,
model name, confidence threshold, precision, recall, mAP, and relevant
observations. Include enough context to reproduce each number.

The current checkout does not contain final metric files or videos. Add them
before submission and link them from the root
[`README.md`](../README.md) and [`demonstration/README.md`](../demonstration/README.md).
