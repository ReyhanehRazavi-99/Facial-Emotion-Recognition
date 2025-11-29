# run_webcam_deit.py
# Webcam FER with DeiT (timm) using your saved artifacts

import os, json, argparse, time
import numpy as np
import cv2
import torch
import torch.nn.functional as F

# ---------- small helpers ----------
def find_file(folder, name_or_list):
    if isinstance(name_or_list, str):
        candidates = [name_or_list]
    else:
        candidates = list(name_or_list)
    for nm in candidates:
        p = os.path.join(folder, nm)
        if os.path.isfile(p):
            return p
    return None

def load_meta(meta_path):
    with open(meta_path, "r") as f:
        meta = json.load(f)
    class_names = meta.get("class_names")
    img_size    = int(meta.get("img_size", 224))
    mean        = meta.get("imagenet_mean", [0.485, 0.456, 0.406])
    std         = meta.get("imagenet_std",  [0.229, 0.224, 0.225])
    return class_names, img_size, mean, std

def get_device():
    try:
        import torch_directml  # optional (works on AMD/Intel/NVIDIA)
        return torch_directml.device(), "DirectML"
    except Exception:
        pass
    if torch.cuda.is_available():  return torch.device("cuda"), "CUDA"
    if torch.backends.mps.is_available(): return torch.device("mps"), "MPS"
    return torch.device("cpu"), "CPU"

def preprocess_bgr(bgr, size, mean, std, device):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(rgb).float().permute(2,0,1).unsqueeze(0) / 255.0
    mean_t = torch.tensor(mean).view(1,3,1,1)
    std_t  = torch.tensor(std ).view(1,3,1,1)
    x = (x - mean_t) / std_t
    return x.to(device)

@torch.no_grad()
def predict_crop(model, crop_bgr, size, mean, std, device, class_names):
    x = preprocess_bgr(crop_bgr, size, mean, std, device)
    out = model(x)
    if isinstance(out, (tuple, list)):
        out = out[0]
    probs = F.softmax(out, dim=1).squeeze(0).cpu().numpy()
    idx = int(np.argmax(probs))
    return class_names[idx], float(probs[idx])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts_dir", default=r"C:\work\emotion6",
                    help="Folder containing deit_classify_scripted.pt / deit_inference_model_state.pth / deit_inference_meta(.json)")
    ap.add_argument("--use_scripted", action="store_true",
                    help="Use TorchScript model deit_classify_scripted.pt (fastest, no timm needed)")
    ap.add_argument("--cam", type=int, default=0, help="Webcam index (0 default)")
    ap.add_argument("--backend", default="dshow", help="OpenCV backend: dshow/msmf/default (Windows)")
    ap.add_argument("--minsize", type=int, default=60, help="Minimum face size in pixels")
    args = ap.parse_args()

    artifacts = os.path.abspath(args.artifacts_dir)
    print("Artifacts dir:", artifacts)

    # Resolve meta (with or without .json extension)
    meta_path = find_file(artifacts, ["deit_inference_meta.json", "deit_inference_meta"])
    if not meta_path:
        raise FileNotFoundError("Missing meta JSON: deit_inference_meta(.json)")

    class_names, img_size, mean, std = load_meta(meta_path)
    if not class_names:
        raise RuntimeError("class_names missing from meta JSON.")
    print("Classes:", class_names)

    device, backend_name = get_device()
    print("Device:", backend_name)

    # Choose model source
    if args.use_scripted:
        ts_path = find_file(artifacts, "deit_classify_scripted.pt")
        if not ts_path:
            raise FileNotFoundError("Missing TorchScript: deit_classify_scripted.pt (use without --use_scripted to fallback to state_dict)")
        print("Loading TorchScript model:", ts_path)
        model = torch.jit.load(ts_path, map_location="cpu")
        # move to device (DirectML / CUDA / CPU)
        try:
            import torch_directml  # noqa
            model = model.to(device)
        except Exception:
            if isinstance(device, torch.device):
                model = model.to(device)
        model.eval()
    else:
        # Rebuild timm DeiT and load state dict
        import timm
        state_path = find_file(artifacts, "deit_inference_model_state.pth")
        if not state_path:
            raise FileNotFoundError("Missing state_dict: deit_inference_model_state.pth (or use --use_scripted)")
        print("Loading timm model + state_dict:", state_path)
        model = timm.create_model('deit_base_patch16_224',
                                  pretrained=False, num_classes=len(class_names))
        model.load_state_dict(torch.load(state_path, map_location="cpu"), strict=True)
        model = model.to(device).eval()

    # Face detector
    face_cascade = cv2.CascadeClassifier(
        os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    )

    # Webcam open
    cap_kwargs = {}
    if args.backend.lower() == "dshow":
        cap_kwargs["apiPreference"] = cv2.CAP_DSHOW
    elif args.backend.lower() == "msmf":
        cap_kwargs["apiPreference"] = cv2.CAP_MSMF

    cap = cv2.VideoCapture(args.cam, **cap_kwargs)
    if not cap.isOpened():
        print("Could not open webcam. Try --cam 1 or --backend msmf/dshow.")
        return

    print("Press 'q' to quit.")
    t0, frames = time.time(), 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2,
                                              minNeighbors=5, minSize=(args.minsize, args.minsize))

        if len(faces) == 0:
            # (optional) classify whole frame if no face
            label, conf = predict_crop(model, frame, img_size, mean, std, device, class_names)
            cv2.putText(frame, f"{label} ({conf:.2f})", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
        else:
            for (x,y,w,h) in faces:
                m = int(0.15*w)
                x0, y0 = max(x-m,0), max(y-m,0)
                x1, y1 = min(x+w+m, frame.shape[1]), min(y+h+m, frame.shape[0])
                crop = frame[y0:y1, x0:x1]
                label, conf = predict_crop(model, crop, img_size, mean, std, device, class_names)
                cv2.rectangle(frame, (x0,y0), (x1,y1), (0,255,0), 2)
                cv2.putText(frame, f"{label} ({conf:.2f})", (x0, y0-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)

        # FPS overlay every ~20 frames
        if frames % 20 == 0:
            dt = time.time() - t0
            fps = frames / max(1e-3, dt)
            cv2.putText(frame, f"FPS: {fps:.1f} | {backend_name}", (10, frame.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow("Emotion (DeiT)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
