# Facial-Emotion-Recognition
## Conceptual Design — Real-Time Emotion Recognition from a Webcam
Problem statement & high-level goal (Part 1)

The goal of my semester project is to build a real-time facial emotion recognition (FER) system that runs on a laptop webcam. Given a live video stream, the system will detect a face, preprocess it, and classify the expression into eight categories consistent with FERPlus: neutral, happiness, surprise, sadness, anger, disgust, fear, and contempt. The final application will display the webcam feed with bounding boxes and emotion labels (and confidence) overlaid.

At this stage, the focus is on conceptual design rather than implementation details. The plan balances classic CV steps (face detection and alignment) with deep feature extractors and a downstream SVM (SVC) classifier. I will experiment with several backbone models—including pretrained CNNs/ViTs and a small CNN I design from scratch—and compare them using a common validation protocol. The end product should be fast enough for interactive use (low latency) and robust to small pose changes, lighting variations, and background clutter.

Data requirements & dataset plan (Part 2)

I will primarily use FERPlus, a cleaned and re-annotated version of FER2013, which provides expression labels for ~35K grayscale face crops at 48×48 resolution along the eight expression classes listed above. I will convert these labels into hard class targets (argmax of the provided label distribution) for standard classification, but I may also keep the full distributions for optional experimentation (e.g., Kullback–Leibler divergence or label smoothing).

To support model development correctly, I will maintain three disjoint subsets:

Training set: used to fit model parameters and/or the SVM classifier. I will apply data augmentation only here (random horizontal flips, small rotations ±15°, small shifts/zoom, brightness/contrast jitter) to improve generalization and approximate real-world webcam conditions.

Validation set: held-out for model selection and early stopping. No training happens on this split; augmentation is minimal or none so it approximates deployment conditions. I will use it to choose the best backbone, SVM hyperparameters, and preprocessing options.

Test set: never touched during development. It provides the final, unbiased estimate of performance. I will report accuracy, macro-F1, and a confusion matrix.

Because contempt is relatively rare in FERPlus, I expect class imbalance. I’ll address it via class weighting, balanced sampling, or focal loss during end-to-end training, and class_weight="balanced" when training SVC.

(Optionally, for future robustness) I may supplement FERPlus with small curated samples from other FER datasets (e.g., RAF-DB or AffectNet subsets) to probe domain shift. If I do so, I will keep the official FERPlus test set pristine for final evaluation.

Proposed pipeline & what must be learned (Part 3)

1) Face detection & alignment.
Even though FERPlus images are already face crops, webcam frames are not. I will run a lightweight face detector (initially OpenCV Haar cascade for simplicity; later possibly a modern detector) and crop to the face region. I will experiment with alignment using eye corners or facial landmarks; the goal is to reduce pose variation so the downstream classifier can be more invariant to roll/scale.

2) Preprocessing & invariances.
I’ll standardize the input to what the backbone expects. For pretrained ImageNet models that operate on RGB 224×224, I will resize and apply ImageNet mean/std normalization. The method should be largely agnostic to background (tight cropping helps), moderately agnostic to illumination (augmentation + normalization), and tolerant to small pose/scale changes (alignment + augmentation). I’ll explicitly test robustness to glasses, mild occlusions, and mild blur.

3) Feature extractors (multiple backbones).
I’ll compare several models:

GoogLeNet (Inception v1) — a classic, efficient CNN; I’ve already prototyped training it end-to-end and also using it as a fixed feature extractor (1024-D embedding before the final FC).

AlexNet — older but fast; useful as a baseline and for quick feature extraction.

DeiT-Base (Vision Transformer) — a modern transformer model; I’ll either fine-tune it end-to-end or extract its CLS token embedding (typically 768-D).

Scratch CNN — a small custom network (3–5 conv blocks + GAP + linear head) trained from scratch on FERPlus to understand data requirements, capacity, and overfitting behavior.

For each backbone, I’ll consider two regimes:

End-to-end softmax: Replace the final classification head with an 8-class layer and fine-tune with cross-entropy (and label smoothing). This establishes a strong baseline.

Backbone + SVM: Freeze the backbone, extract a compact embedding per image (e.g., GoogLeNet 1024-D, DeiT 768-D), then train an SVC downstream. This often yields competitive accuracy with less overfitting on small datasets, and it gives a crisp separation between representation learning and classification.

Why SVM?
SVMs are powerful margin-based classifiers that work well on fixed-dimensional embeddings, especially when data is modest. With linear SVM I get speed and interpretability (a single hyperparameter C). I can also try RBF SVM (tuning C and gamma) for nonlinear separation, though I’ll weigh accuracy gains against inference cost.

4) Training strategy & early stopping.
I’ll train on the augmented training split and monitor validation loss/accuracy for early stopping. I may use AdamW with a cosine schedule (warm-up) for transformer models and standard Adam/SGD for CNNs. I’ll log metrics per epoch and save the best checkpoint based on validation performance.

Features to calculate & properties to be agnostic to (Part 4)

Features.
For deep models, the “features” are the penultimate embeddings produced by the network (e.g., the 1024-D vector in GoogLeNet or the CLS token in DeiT). If I prototype traditional baselines, I might compute HOG/LBP descriptors for comparison, but the main path uses deep embeddings.

Agnostic properties.
The solution should ignore background and be robust to minor head pose, illumination, scale, small occlusions, and camera noise. Augmentations and alignment explicitly target these nuisances. The system is not expected to handle extreme poses or heavy occlusions (e.g., masks covering most of the face) in the first iteration.

Metrics, evaluation, and analysis plan (Part 5)

I will report:

Top-1 accuracy on validation and test.

Macro-F1 to account for class imbalance.

Confusion matrices to expose which emotion pairs are commonly confused (e.g., fear vs. surprise, anger vs. disgust, neutral vs. sadness).

(Optional) Calibration (e.g., reliability diagrams) if probabilities are required; SVM scores can be calibrated via Platt scaling.

I’ll perform ablation studies:

Backbone choice (GoogLeNet vs DeiT vs AlexNet vs Scratch CNN).

End-to-end softmax vs SVM on embeddings.

With vs without face alignment.

Effect of stronger augmentations and class weighting.

Deployment sketch & runtime constraints (Part 6)

The inference pipeline in the app will be:

Capture frame from webcam.

Detect face; crop with a small margin.

Resize to 224×224, normalize.

Run backbone (GPU if available, else CPU/DirectML).

Classify with SVC (or the model head).

Overlay label + confidence; loop at ~15–30 FPS if possible.

To keep latency low I will:

favor GoogLeNet or Mobile-friendly models if DeiT is too slow on CPU,

use mixed precision when running on a GPU,

cache allocations, and avoid unnecessary image copies.

Risks, ethics, and open questions (Part 7)

Dataset bias & fairness: FERPlus is web-scraped; distribution across age, ethnicity, and lighting can bias outcomes. I will monitor per-class/per-subset performance and document limitations.

Ambiguity of expressions: Some frames express mixed emotions; “ground truth” is not always absolute. That’s why label smoothing or using FERPlus label distributions could help.

Privacy: All processing stays on device; no images are stored or sent. I will display a clear note in the README about responsible use.

Domain shift: Webcam conditions differ from curated datasets. I may collect a small personal calibration set (with consent) to fine-tune or at least evaluate domain shift.

What I need to learn next (Part 8)

Practical face alignment (landmarks, similarity transforms).

Proper training regularization (weight decay, label smoothing, early stopping).

SVM hyperparameter tuning and calibration on deep embeddings.

Debugging with error analysis: read confusion matrices, inspect misclassifications, and adjust augmentations/alignments accordingly.

Lightweight deployment tricks (mixed precision, smaller backbones, quantization if needed).
