#!/bin/bash
venv/bin/python eval_compo.py \
  --coco-val-folder ./datasets/val2014/ \
  --annotations-path datasets/annotations/captions_val2014.json \
  --split-dataset-path data/dataset_splits/dataset_splits_1.json \
  --checkpoint ./checkpoint/test/test_check_g/coco_prefix-009.pt

