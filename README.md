##  Part 1 — Conceptual Design: Real-Time Emotion Recognition from a Webcam

Goal. Build a real-time facial emotion recognition (FER) system that runs locally on a laptop using the built-in webcam. For each video frame, the system detects a face, preprocesses it, and classifies the expression into eight categories: neutral, happiness, surprise, sadness, anger, disgust, fear, and contempt. The application overlays a bounding box, label, and confidence on the live video.

System overview. The pipeline is deliberately modular so I can swap parts and compare designs:

Face detection & cropping. Start with a lightweight detector (OpenCV Haar cascade) for simplicity and speed; consider upgrading to a modern detector/landmarks later if latency permits. A small margin around the detected face is retained to preserve context.

Preprocessing & normalization. Convert to RGB, resize to 224×224, and normalize with ImageNet mean/std so pretrained backbones behave as expected. I will investigate optional face alignment (e.g., using eye positions) to reduce pose variance and improve robustness.

Feature extraction (backbones). Compare multiple learned representations:

GoogLeNet (Inception v1) — efficient, competitive baseline; I can fine-tune end-to-end or extract its penultimate embedding (~1024-D).

EfficientNetV2-B0 — modern, parameter-efficient CNN; likely to offer better accuracy/latency trade-offs.

DeiT-Base (Vision Transformer) — transformer model; either fine-tune the full model or export the pre-logits / CLS token (~768-D) as a fixed feature.

AlexNet (feature extractor) — fast classical baseline for feature+SVM experiments.

Small scratch CNN — a compact network I’ll design to understand capacity/overfitting on FER+ and to benchmark against pretrained models.

Classifier. I will evaluate two regimes:

End-to-end softmax head (fine-tuning the backbone with cross-entropy and label smoothing).

Backbone + SVM (SVC) using frozen deep embeddings as features. Linear SVM (with C and class weights) provides a strong baseline on modest data; RBF SVM can be explored if nonlinearity yields gains that justify added cost.

Inference loop. Capture frames, detect face, crop+normalize, run the backbone (GPU if available; CPU/DirectML otherwise), classify via softmax head or SVM, and render results at ~15–30 FPS with low latency.

What the model should be invariant to. The solution should be largely agnostic to background (tight crops help), moderately robust to illumination changes, small pose/scale shifts, mild occlusions (e.g., glasses), and moderate blur/noise. These requirements shape augmentation and alignment choices. Extreme poses or heavy occlusions are out of scope for the first iteration.

Training & model selection. I will:

Train on an augmented training split and monitor validation loss/accuracy for early stopping and checkpointing (keep the best model by validation loss).

Use optimizers appropriate to each backbone (e.g., AdamW + cosine schedule for ViTs, Adam/SGD for CNNs).

Track per-class metrics and confusion matrices to expose common confusions (e.g., fear↔surprise, anger↔disgust, neutral↔sadness).

For SVM pipelines, perform light hyperparameter sweeps (e.g., C, kernel) and consider probability calibration if needed.

Why SVM? On fixed-dimensional deep embeddings, SVMs are strong margin-based classifiers that often generalize well with modest data and offer a simple tunable boundary (especially linear SVM). They separate representation learning (backbone) from classification, which helps diagnose where performance bottlenecks arise.

Runtime & deployment. To maintain responsiveness: favor efficient backbones (GoogLeNet/EfficientNet) if DeiT is too slow on CPU; use mixed precision on GPU; cache allocations; avoid unnecessary copies; and keep pre/post-processing lightweight.

Ethics & risks. FER datasets can carry demographic and contextual biases. I will report per-class metrics and document limitations; the application runs fully on-device to protect privacy. Expression labels can be ambiguous; label smoothing or using FER+ label distributions may better reflect uncertainty. Lastly, webcam conditions can differ from curated datasets (domain shift); I may collect a small, consented calibration set to probe this gap.

##  Part 2 — Data Acquisition & Splits

Primary dataset. I am using FER+ (FER2013Plus), a curated re-annotation of the FER2013 dataset with eight target expressions (anger, contempt, disgust, fear, happiness, neutral, sadness, surprise). FER+ consists of in-the-wild face crops (originally 48×48 grayscale), which I upscale to 224×224 and replicate across three channels for ImageNet-pretrained backbones.

Source (download): Kaggle – FER2013Plus (FER+)
https://www.kaggle.com/datasets/subhaditya/fer2013plus

References: FER13/Kaggle (2013) challenge dataset and the FER+ re-annotation paper by Barsoum et al. (2016), which introduces label distributions gathered by crowd workers.

Project-specific split. In line with the recommended train / validation / test paradigm, I maintain three disjoint subsets:

Training set — used to fit model parameters and, for the SVM route, to learn the SVC on deep embeddings. Data augmentation (minor rotations/shift/zoom/horizontal flip; optional brightness/contrast jitter) is applied only to this set to improve generalization.

Validation set (hold-out) — fixed 20% carved from the training directory (single hold-out split, not k-fold). Used each epoch for model selection and early stopping. No training occurs here; augmentation is minimal or none, to approximate deployment conditions.

Test set (unknown) — kept sealed during development and evaluated once for the final report, providing an unbiased estimate of performance.

Observed counts on my copy.

Train: 22,712 images (8 classes)

Validation: 5,674 images (8 classes)

Test: 7,099 images (8 classes)

Class mapping (alphabetical): anger=0, contempt=1, disgust=2, fear=3, happiness=4, neutral=5, sadness=6, surprise=7

Important differences between train vs. validation.

Augmentation: present only in training; absent in validation/test to avoid inflating metrics.

Shuffling: training is shuffled to reduce correlation; validation/test evaluation is deterministic.

Identity leakage caveat: FER+ does not include subject IDs, so image-level splitting cannot guarantee person-disjoint partitions. To mitigate, I (a) rely on augmentation, (b) select models via validation loss (not test), and (c) keep the test set untouched until the end. If identity-level separation becomes necessary, a dataset with subject IDs or manual identity grouping would be required.

Sample characteristics.

Resolution & format: native 48×48 grayscale; upscaled to 224×224 and expanded to 3 channels for pretrained backbones.

Acquisition & sensors: scraped from diverse sources; mixed cameras and conditions; unconstrained “in-the-wild” faces.

Ambient conditions: broad variability in illumination, pose, background clutter, and mild occlusions; this variability is aligned with the webcam deployment target.

Class balance: expected imbalance (e.g., happiness often over-represented). I address this with class weighting (for SVM and some end-to-end setups), careful thresholding, and macro-averaged metrics (macro-F1) alongside accuracy.

Quality control & integrity. I verified class folders and counts, ensured the folder structure is consistent, and confirmed the eight-class mapping used by my loaders. I also standardized preprocessing (resize + ImageNet normalization) across all backbones to keep comparisons fair.

Deliverables & handling. I have physically downloaded FER+ and instantiated the three splits above. The training and validation partitions are in active use for model development and selection; the test partition is reserved exclusively for final evaluation and reporting.


## Part 3 — Preprocessing, Segmentation & Feature Extraction (First Update)
Overview

This update documents the pipeline pieces I have implemented so far for facial emotion recognition on FERPlus (8 classes: anger, contempt, disgust, fear, happiness, neutral, sadness, surprise). The focus here is on data preprocessing, (light) segmentation, and feature extraction—plus a rationale for the methods I chose.

I implemented and tested three complementary paths:

AlexNet (fixed feature extractor) + SVM classifier
Transfer learning by freezing the backbone; extract a 9216-D embedding and train a linear SVM.

AlexNet (fine-tuned end-to-end) with an 8-class head
Transfer learning by adapting the final classifier and updating backbone weights.

DeiT-Base (Vision Transformer) fine-tuned end-to-end
Modern transformer backbone trained with a regularized schedule & mixed precision.

Each path uses the same, comparable preprocessing and the same train/val/test split protocol, so results are comparable in the next phase.


How I construct Train/Val/Test in code:

For the Keras generator setup (used earlier), I employ validation_split=0.20 inside ImageDataGenerator to carve 20% of the training directory as validation.

For the PyTorch/timm DeiT pipeline, I use StratifiedShuffleSplit on the TRAIN folder to create an 80/20 train/val split that preserves class ratios. The TEST folder stays untouched.


Normalization & Resizing

All deep backbones here are ImageNet-pretrained (or initialized). To align with those statistics:

Resize: images → 224×224

Convert to RGB: (FERPlus is grayscale; I replicate channels)

Normalize:

mean = [0.485, 0.456, 0.406]

std = [0.229, 0.224, 0.225]

Justification:
Using ImageNet normalization keeps the input distribution consistent with what the pretrained models expect, improving transfer performance and convergence stability.

Augmentation (Train Only)

AutoAugment (IMAGENET policy) — color/contrast/translation/geometry perturbations

RandomHorizontalFlip

RandomErasing (p≈0.2)

Occasional small rotations and shifts (when using Keras generators)

Justification:
Augmentations add robustness to pose, illumination, and mild occlusions; they also reduce overfitting on FERPlus (which has modest per-class counts, especially for contempt).

(Light) Segmentation

FERPlus images are already face crops.

For webcam inference, I will add Haar cascade face detection to crop faces on-the-fly.

No heavy segmentation is required at training time; the face is already isolated. If needed later, I can add landmark-based alignment to reduce roll/scale variance.



Feature Extraction Methods Implemented (What I ran)
1) AlexNet (Fixed Features) + SVM

It is on the furplus_1 file, first part
Backbone: torchvision.models.alexnet(pretrained=True)
I freeze the network and take features → avgpool → Flatten → 9216-D vector.

Preprocessing: resize 224, ImageNet mean/std, train-only augmentation.

Feature extraction: run the entire train/val/test once to produce:

X_train (N×9216), y_train

X_val (N×9216), y_val

X_test (N×9216), y_test

Classifier: StandardScaler + Linear SVM (C=1.0, class_weight="balanced")


Outputs: Accuracy, classification report, and confusion matrices on val and test.

Why this approach?
This is transfer learning by fixed features. It is computationally light (no backprop through the backbone), surprisingly strong on modest datasets, and gives a clean separation between representation (deep features) and classifier (SVM). The SVM margin often handles class overlap nicely when features are good.

2) AlexNet (Fine-Tuned End-to-End)

It is on the furplus_1 file, second part
Backbone: start from ImageNet weights; replace the final FC layer with an 8-class head.

Training: optimize all weights (or optionally freeze early conv blocks) using AdamW, label smoothing, and a cosine LR schedule; early stopping on validation loss.

Validation use: monitor validation metrics each epoch to:

stop training when no improvement (patience),

save the best checkpoint.

Why this approach?
This is transfer learning by fine-tuning. If the target domain (FERPlus facial expressions) differs from ImageNet categories, updating convolutional features can improve class-specific cues (eyebrows, eye/mouth shapes). It’s heavier than fixed features but can close accuracy gaps, especially for subtle classes (e.g., contempt).

3) DeiT-Base (Vision Transformer)


It is on the ferplus_vision_transformer file
Backbone: timm.create_model('deit_base_patch16_224', pretrained=True, num_classes=8)

Training: end-to-end fine-tuning with AdamW, label smoothing, CosineLRScheduler, AutoAugment, RandomErasing, and mixed precision (autocast + GradScaler) on GPU.

Split policy: Stratified 80/20 on TRAIN for val; keep TEST untouched.

Early stopping on validation loss; save best checkpoint + inference bundle (state_dict + meta.json with classes and normalization).

Why this approach?
Transformers (ViTs) can capture long-range dependencies and often show strong transfer when regularized well. DeiT has a good compute/accuracy trade-off for 224×224 inputs; with modern augmentations and cosine scheduling it’s a solid benchmark against CNNs.

