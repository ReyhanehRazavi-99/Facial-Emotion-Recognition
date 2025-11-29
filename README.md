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



## part 4

Fine-tuning strategies and experimental notebooks

After training baseline models, I systematically compared three different fine-tuning strategies for two different backbones: AlexNet (a classic CNN) and DeiT (Data-efficient Image Transformer, a Vision Transformer). Instead of creating separate files for each configuration, I implemented all three setups for each backbone inside two Jupyter notebooks:

– 3diffeent_alexnet.ipynb
– 3different-vision_transformers.ipynb

In each notebook, the three fine-tuning strategies are:

a) Linear probe. In this setup, all backbone weights are frozen, and only the final fully connected classification layer is replaced and trained on FERPlus. The notebook configures the model so that the AlexNet or DeiT backbone acts purely as a fixed feature extractor. This approach is computationally cheap and provides a lower bound on performance, showing how transferable the pretrained features are without any adaptation to the emotion recognition task.

b) Partial fine-tune. In this setup, the early and mid-level layers of the backbone remain frozen, while the last block or last stage of the backbone and the final classifier are unfrozen and trainable. Within each notebook, this is implemented by selectively setting requires_grad for the last part of the network. The motivation is to allow the model to adapt its highest-level features to FERPlus while preserving most of the general representation learned during pretraining. This provides a compromise between efficiency and flexibility and usually improves performance over a pure linear probe.

c) Full fine-tune. In this setup, all layers of the backbone and the classifier are unfrozen and updated during training. The notebooks configure the optimizer to update the entire network. This gives the model maximum flexibility to adapt to the FERPlus domain at the cost of higher computation and a higher risk of overfitting if regularization and early stopping are not used carefully.

Both notebooks follow a similar structure: loading FERPlus, applying preprocessing and data augmentation, selecting one of the three fine-tuning modes, training the model accordingly, and evaluating it using overall accuracy and class-wise confusion matrices on the validation and/or test sets. By keeping all three strategies in a single notebook per backbone, it is easy to compare the different settings side by side and reuse the same data-loading and evaluation code.

Model comparison and selection

By comparing the validation accuracies and confusion matrices across the three strategies inside each notebook, a clear pattern emerged. For both AlexNet and DeiT, performance improved as I moved from the linear probe to the partial fine-tune and then to the full fine-tune configuration. However, the fully fine-tuned DeiT model consistently achieved the highest overall accuracy and produced the most balanced confusion matrix, particularly on more challenging classes such as contempt, disgust, and fear. Fully fine-tuned AlexNet still lagged behind fully fine-tuned DeiT, especially in distinguishing subtle expressions and separating neutral from low-intensity sadness or anger.

Based on these results, I selected the full fine-tune DeiT configuration as the final model to deploy for real-time emotion recognition using the webcam.

Exported inference artifacts and external storage

After identifying the best model, I exported it into a small, deployment-ready set of artifacts. Because these files are very large and exceed GitHub’s per-file size limit, they are stored externally on Google Drive and can be downloaded from the link provided. The main artifacts are:

– deit_best.ckpt
This is the full training checkpoint saved at the epoch with the best validation performance. It contains the complete model weights, optimizer state, learning-rate scheduler state, and training metadata such as the best epoch and best accuracy. It is mainly used for reproducibility and for continuing training or further experiments.
https://drive.google.com/file/d/11RZYopQp29cE6Hx4HF9qgYuBQ9p0ZonU/view?usp=sharing



– deit_inference_model_state.pth
This is a compact state-dict containing only the model weights needed for inference, without any optimizer or scheduler information. This is the file that the real-time inference script loads by default. It is smaller than the full checkpoint and therefore more convenient for deployment.

https://drive.google.com/file/d/1XmZd-2bdaNF2sTXSRCsPPwtANv3h2o2C/view?usp=sharing



– deit_classify_scripted.pt
This is a TorchScript version of the classifier, obtained by scripting or tracing the trained model. TorchScript allows the model to be run without the original Python training code and makes it easier to integrate the model into other applications or different runtime environments, with potentially faster inference.

https://drive.google.com/file/d/1-EHdPDzAbQkpHDus4cILXzbuBpht-X6w/view?usp=sharing




– deit_inference_meta.json
This JSON file stores the metadata required at inference time. It includes the list of emotion class labels (anger, contempt, disgust, fear, happiness, neutral, sadness, and surprise), the expected input image size, normalization parameters, and any configuration flags used during training. This ensures that the preprocessing performed at inference time matches the training configuration exactly.

https://drive.google.com/file/d/1pJOIl-HC2gsiWmYebn0b8mPpdgVjV2BW/view?usp=sharing



– run_webcam_deit.py
This is the main inference script. It reads the metadata from deit_inference_meta.json, builds the DeiT model architecture, loads the weights from deit_inference_model_state.pth (or from the TorchScript file if configured that way), opens a webcam stream using OpenCV, preprocesses each frame (cropping and resizing the face region, converting to the correct tensor format, applying normalization), feeds the processed frame through the model, and overlays the predicted emotion label on the live video feed.

The Google Drive link contains all of these files with the same names as above. Users are instructed to download them and place them into the appropriate project folder before running the webcam demo.

How to run the real-time emotion recognition demo

To run the webcam-based emotion recognition demo, the user first clones the repository and sets up a Python environment. On my machine, I used a conda environment named “emotion7,” but any environment name can be used.

Step 1: Clone the repository and move into the project directory.

cd Facial-Emotion-Recognition/emotion6

Step 2: Create and activate a Python or conda environment (example using conda).

conda create -n emotion7 python=3.10
conda activate emotion7

Step 3: Install the necessary packages. At a minimum, the environment should include PyTorch, torchvision, timm, OpenCV, NumPy, and Matplotlib.

pip install torch torchvision timm opencv-python numpy matplotlib

(If a requirements.txt file is provided in the repo, the user can instead run: pip install -r requirements.txt)

Step 4: Download the model artifacts from Google Drive. The link to the Google Drive folder is given in the README. The user should download deit_inference_model_state.pth, deit_inference_meta.json, and optionally deit_best.ckpt and deit_classify_scripted.pt, and place them in the same folder as run_webcam_deit.py (for example, the emotion6 directory).

Step 5: Run the webcam script.

python run_webcam_deit.py

If everything is configured correctly, the script will load the metadata and model weights, open the webcam, and start processing each frame. The predicted emotion label will be displayed on top of the live video stream in real time. For the report, I captured screenshots of myself displaying different emotions (such as happiness, sadness, anger, and surprise) to illustrate how the model behaves under realistic conditions. My images are inthis repo with Rihanna added to their names.

Challenges and limitations

Although the real-time system demonstrates that a fully fine-tuned DeiT model can recognize emotions from a webcam feed, there are several important challenges and limitations.

First, there is a domain shift between FERPlus and the webcam input. FERPlus images are grayscale, tightly cropped around the face, and often collected under more controlled conditions. The webcam feed, however, is RGB and can include variations in lighting, background clutter, camera angle, and partial occlusions from hair, glasses, or hands. This mismatch can reduce performance when the model is applied to real-time video, and some expressions that are easy to classify in FERPlus become more ambiguous.

Second, FERPlus exhibits class imbalance. Emotions such as happiness and neutral are more frequent than others like contempt and disgust. As a result, the model may be biased toward predicting the majority classes during live use, particularly when expressions are subtle, mixed, or low in intensity.

Third, there is label uncertainty. FERPlus labels are derived from multiple annotators, but the model trains on a single aggregated label per image. This discards information about disagreement among raters. When the webcam captures ambiguous expressions that humans might also disagree on, the model still outputs one hard label, which can sometimes give a misleading impression of high certainty.

Fourth, there are computational constraints. Vision Transformers such as DeiT are more computationally demanding than shallower CNNs like AlexNet. On a CPU-only machine, the frame rate can drop, especially at higher input resolutions, which limits how smooth and “real-time” the experience is without GPU acceleration.

Despite these limitations, the fully fine-tuned DeiT model clearly outperforms the AlexNet configurations in my experiments and serves as a strong proof-of-concept for real-time emotion recognition. The two notebooks (3diffeent_alexnet.ipynb and 3different-vision_transformers.ipynb) document how different fine-tuning strategies affect performance, and the exported artifacts together with run_webcam_deit.py and the Google Drive weights provide a compact, reproducible pipeline that others can download, run, analyze, and extend.



The results show a clear progression in performance as models become more expressive.
Using AlexNet only as a feature extractor produced limited classification capability, especially for nuanced emotions such as contempt and sadness.
Fine-tuning AlexNet improved class separation, indicating that domain-specific learning is crucial for emotion recognition.
Ultimately, the Vision Transformer achieved the strongest performance due to its self-attention mechanism, which captures global facial structures and subtle expression patterns more effectively.
These findings justify continuing the project using transformer-based architectures. (Look at the graphs in the code also images uploaded, named 1,2, and 3 for the 3 models respectievely.)
