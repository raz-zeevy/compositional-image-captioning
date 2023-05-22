#!/bin/bash
venv/bin/python eval.py --dataset-splits ./data/coco/oscar_split_ViT-B_32_test.pkl \
      --checkpoint ./models/VIT-B_32.pt

