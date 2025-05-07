# Transformer-Augmented YOLOv8 for Wildfire Detection

## Overview

This project presents an enhanced wildfire detection system that builds on YOLOv8 by integrating Vision Transformer (ViT) backbones to improve both accuracy and semantic understanding in challenging fire, smoke, and non-fire detection scenarios. Through a three-stage modular training pipeline, the system evolves from traditional detection to semantically informed, lightweight, and interpretable models suitable for edge deployment.

## Key Innovations

* Transformer Backbones: YOLOv8’s default CNN backbone is replaced with ViT variants including TinyViT, MobileViT, SwinV2, and EfficientViT.

* Multi-Stage Training:

  * **Stage A**: End-to-end training with the ViT backbone integrated into YOLOv8.
  * **Stage B**: Contrastive learning using MiniLM for aligning image features with text-based semantic labels.
  * **Stage C**: YOLO detection head is trained using frozen, semantically enriched encoder features.

* Semantic Tools: CLIP-style classifiers, BLIP captioning, and attention heatmaps from ViT backbones provide interpretability and insight into model reasoning.

## Model Pipeline

| Stage | Goal                                 | Action                                  |
| ----- | ------------------------------------ | --------------------------------------- |
| A     | Learn fire/smoke/non-fire features   | Full YOLOv8 + ViT fine-tuning           |
| B     | Align vision with language semantics | Contrastive learning (MiniLM + NT-Xent) |
| C     | Detect using semantic features       | Train YOLO head with frozen encoder     |

## Model Performance

| Backbone      | Stage | mAP\@50 | mAP\@50–95 | Precision | Recall | Notes                       |
| ------------- | ----- | ------- | ---------- | --------- | ------ | --------------------------- |
| YOLOv8 (Base) | A     | 0.908   | 0.706      | 0.886     | 0.858  | Baseline CNN                |
| TinyViT       | C     | 0.752   | 0.563      | 0.788     | 0.723  | Best semantic alignment     |
| MobileViT     | C     | 0.733   | 0.558      | 0.814     | 0.722  | Most improved after Stage B |
| SwinV2        | C     | 0.758   | 0.550      | 0.758     | 0.707  | Strong spatial modeling     |
| EfficientViT  | A     | 0.742   | 0.557      | 0.772     | 0.709  | Best out-of-box accuracy    |

## Dataset Summary

* Dataset: MergedYOLO
* Classes: Fire, Smoke, Non-fire
* Composition:

  * 3,000 Fire images
  * 3,000 Smoke images
  * 3,000 Non-fire images
* Sources: D-Fire and Forest Fire/Smoke/Non-fire datasets
* Annotations: YOLO format (class\_id, x\_center, y\_center, width, height)
* Splits: 80% Training, 10% Validation, 10% Test
* Input Resolutions: 224×224 and 640×640 depending on stage

## Experiments and Evaluation

* **Detection Metrics (Stages A & C)**:

  * mAP\@50, mAP\@50–95, Precision, Recall

* **Contrastive Learning Metrics (Stage B)**:

  * InfoNCE loss
  * Cosine similarity accuracy
  * F1-Score, Macro Precision, Macro Recall

* **Qualitative Evaluation**:

  * BLIP captioning used to validate semantic consistency
  * ViT heatmaps visualized spatial attention
  * CLIP-style cosine similarity evaluated alignment with textual labels

## Tech Stack

* Frameworks: PyTorch, Ultralytics YOLOv8, HuggingFace Transformers, TIMM
* Visualization Tools: OpenCV, PIL, Matplotlib
* Backbones Evaluated: TinyViT, MobileViT, SwinV2-Tiny, EfficientViT

## Key Takeaways

* Transformer-based backbones enhance semantic modeling over CNNs.
* MobileViT showed the most improvement across stages, especially post-contrastive training.
* SwinV2 offered stable and robust learning across training stages.
* EfficientViT delivered strong performance without additional tuning, making it ideal for edge deployment.

## Future Work

* Integration with real-time PTZ camera systems
* Inclusion of advanced augmentations like fog, blur, and jitter
* Extension to severity grading and fire spread modeling

