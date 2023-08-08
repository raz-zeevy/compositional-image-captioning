#!/bin/bash
venv/bin/python eval_compo.py \
  --coco_val_folder ./datasets/val2014/
  --annotations_path datasets/annotations/captions_val2014.json \
  --split_dataset_path data/dataset_splits/dataset_splits_1.json \
  --checkpoint ./checkpoint/test/test_check_g/coco_prefix-009.pt