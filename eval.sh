#!/bin/bash
venv/bin/python eval_compo.py \
  --data-folder ./datasets/coco2014_preprocessed/
  --dataset-splits data/coco/oscar_split_ViT-B_32_test.pkl \
  --checkpoint ./checkpoint/coco_prefix_latest.pt

