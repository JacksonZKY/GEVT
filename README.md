# GEVT
Official PyTorch implementation of the paper "Image Provenance Analysis via Graph Encoding with Vision Transformer“

## Overview
Our approach introduces a novel end-to-end framework for image provenance analysis, leveraging a transformer-based architecture to construct accurate directed provenance graphs. The model utilizes a patch attention mechanism to extract fine-grained local and global features, combined with a weighted graph distance loss to emphasize manipulated regions during training. For direction determination, we integrate graph topology into the model using a graph structure masked attention module, which encodes structural relationships within the provenance graph. Additionally, learnable precedence embeddings and auxiliary virtual nodes are employed to accurately predict the flow of transformations. This unified design enables the model to effectively handle diverse manipulation types and capture hierarchical relationships among images, ensuring robust and precise provenance graph construction.

## Bibtex
If you find this codeuseful, please cite our paper:
 ```
@article{zhang2024image,
title={Image Provenance Analysis via Graph Encoding with Vision Transformer},
author={Zhang, Keyang and Kong, Chenqi and Wang, Shiqi and Rocha, Anderson and Li, Haoliang},
journal={arXiv preprint arXiv:2408.14170},
year={2024}
 ```
