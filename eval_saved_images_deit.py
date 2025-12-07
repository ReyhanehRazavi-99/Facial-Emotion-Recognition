import os
import json
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.metrics import confusion_matrix, classification_report


# ---------- 1. Load model and metadata ----------

def load_deit_model(
    weights_path="deit_inference_model_state.pth",
    meta_path="deit_inference_meta.json"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load metadata (class names, etc.)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    # FERPlus classes (8 classes)
    class_names = meta.get("classes", [
        "anger", "contempt", "disgust", "fear",
        "happiness", "neutral", "sadness", "surprise"
    ])
    num_classes = len(class_names)

    # Create DeiT model (this matches what we used in training)
    import timm
    model_name = meta.get("model_name", "deit_base_patch16_224")
    print(f"Creating model: {model_name} with {num_classes} classes")

    model = timm.create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes
    )

    # Load weights
    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # Clean possible prefixes like "module." or "model."
    cleaned_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        if k.startswith("model."):
            k = k[len("model."):]
        cleaned_state[k] = v

    model.load_state_dict(cleaned_state, strict=False)
    model.to(device)
    model.eval()

    return model, class_names, device


# ---------- 2. Build test dataset loader ----------

def make_test_loader(image_root="images", img_size=224, batch_size=32):
    # Standard ImageNet normalization (what we used for training)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = datasets.ImageFolder(root=image_root, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    print("Found the following classes in your test folder:")
    for idx, cls_name in enumerate(dataset.classes):
        print(f"  [{idx}] {cls_name}")

    return dataset, loader


# ---------- 3. Run evaluation ----------

def evaluate_on_folder(
    image_root="images",
    weights_path="deit_inference_model_state.pth",
    meta_path="deit_inference_meta.json"
):
    model, model_classes, device = load_deit_model(weights_path, meta_path)
    dataset, loader = make_test_loader(image_root=image_root)

    true_labels = []
    pred_labels = []

    # Map dataset class indices (folder names) -> class names
    dataset_class_names = dataset.classes  # e.g., ['anger', 'disgust', ...]
    print("\nModel classes:", model_classes)
    print("Dataset classes:", dataset_class_names)

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            targets = targets.numpy()

            for t_idx, p_idx in zip(targets, preds):
                true_name = dataset_class_names[t_idx]   # label from folder
                pred_name = model_classes[p_idx]         # label from model
                true_labels.append(true_name)
                pred_labels.append(pred_name)

    # Overall accuracy (compare label names)
    n_correct = sum(t == p for t, p in zip(true_labels, pred_labels))
    accuracy = n_correct / len(true_labels)
    print(f"\nTest accuracy on folder '{image_root}': {accuracy:.4f}")
    print(f"Correct: {n_correct} / {len(true_labels)}")

    # Confusion matrix only over classes that actually appear in the test set
    present_classes = sorted(set(true_labels))
    label_to_idx = {c: i for i, c in enumerate(present_classes)}

    y_true = [label_to_idx[t] for t in true_labels]
    # If model predicts a class we don't have in this test set (e.g., 'contempt'),
    # we skip those for the confusion matrix.
    y_pred = []
    y_true_filtered = []
    for t_name, p_name in zip(true_labels, pred_labels):
        if p_name not in label_to_idx:
            # e.g., predicted 'contempt' but we have no 'contempt' folder
            continue
        y_true_filtered.append(label_to_idx[t_name])
        y_pred.append(label_to_idx[p_name])

    cm = confusion_matrix(y_true_filtered, y_pred, labels=list(range(len(present_classes))))
    print("\nConfusion matrix (rows = true, cols = predicted):")
    print("Classes:", present_classes)
    print(cm)

    print("\nClassification report:")
    print(classification_report(y_true_filtered, y_pred, target_names=present_classes))


if __name__ == "__main__":
    evaluate_on_folder(
        image_root="images",
        weights_path="deit_inference_model_state.pth",
        meta_path="deit_inference_meta.json",
    )
