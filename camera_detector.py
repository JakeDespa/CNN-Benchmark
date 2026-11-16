import sys
import os
import time
import argparse
from pathlib import Path

try:
    import cv2
except ImportError:
    print("\n" + "="*70)
    print("[ERROR] OpenCV ('cv2') is not installed in your Python environment.")
    print("Please install OpenCV by running the following command in your terminal:")
    print("\n    pip install opencv-python")
    print("="*70 + "\n")
    sys.exit(1)

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ======================= CONFIG & DEFAULTS =======================
BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = BASE_DIR / "results" / "checkpoints"

# Classes in standard alphabetical ImageFolder order
DEFAULT_CLASSES = ['metal', 'paper', 'plastic']

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# Available trained model options matching checkpoints
MODEL_CONFIGS = {
    'EfficientNet-B0':    dict(fn='efficientnet_b0', size=224, file='EfficientNet-B0.pth'),
    'ResNet-18':          dict(fn='resnet18', size=224, file='ResNet-18.pth'),
    'MobileNetV2':        dict(fn='mobilenet_v2', size=224, file='MobileNetV2.pth'),
    'DenseNet-121':       dict(fn='densenet121', size=224, file='DenseNet-121.pth'),
    'ConvNeXt-Tiny':      dict(fn='convnext_tiny', size=224, file='ConvNeXt-Tiny.pth'),
    'AlexNet':            dict(fn='alexnet', size=224, file='AlexNet.pth'),
    'VGG-11':             dict(fn='vgg11', size=224, file='VGG-11.pth'),
    'GoogLeNet':          dict(fn='googlenet', size=224, file='GoogLeNet.pth', has_aux=True),
    'Inception-v3':       dict(fn='inception_v3', size=299, file='Inception-v3.pth', has_aux=True),
    'SqueezeNet 1.1':     dict(fn='squeezenet1_1', size=224, file='SqueezeNet_1.1.pth'),
    'Wide ResNet-50-2':   dict(fn='wide_resnet50_2', size=224, file='Wide_ResNet-50-2.pth'),
    'ResNeXt-50 32x4d':   dict(fn='resnext50_32x4d', size=224, file='ResNeXt-50_32x4d.pth'),
    'ShuffleNetV2 x0.5':  dict(fn='shufflenet_v2_x0_5', size=224, file='ShuffleNetV2_x0.5.pth'),
    'MNASNet 0.5':        dict(fn='mnasnet0_5', size=224, file='MNASNet_0.5.pth'),
    'MobileNetV3-Small':  dict(fn='mobilenet_v3_small', size=224, file='MobileNetV3-Small.pth'),
    'RegNetY-400MF':      dict(fn='regnet_y_400mf', size=224, file='RegNetY-400MF.pth'),
    'EfficientNetV2-S':   dict(fn='efficientnet_v2_s', size=224, file='EfficientNetV2-S.pth'),
}
# =================================================================

def replace_head(model, num_classes):
    if isinstance(model, models.SqueezeNet):
        model.classifier[1] = nn.Conv2d(model.classifier[1].in_channels, num_classes, kernel_size=1)
        model.num_classes = num_classes
        return model

    last = None
    for name, m in model.named_modules():
        if isinstance(m, nn.Linear):
            last = name
    parent_name, _, child = last.rpartition('.')
    parent = model.get_submodule(parent_name) if parent_name else model
    old = model.get_submodule(last)
    new = nn.Linear(old.in_features, num_classes)
    if isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(child)] = new
    else:
        setattr(parent, child, new)
    return model

def load_trained_model(model_name, num_classes, device):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model name: {model_name}")

    cfg = MODEL_CONFIGS[model_name]
    ckpt_path = CHECKPOINT_DIR / cfg['file']
    
    if not ckpt_path.exists():
        # Fallback search if exact filename varies
        matches = list(CHECKPOINT_DIR.glob(f"*{model_name.replace(' ', '*')}*.pth"))
        if matches:
            ckpt_path = matches[0]
        else:
            raise FileNotFoundError(f"Checkpoint not found for {model_name} at {ckpt_path}")

    kwargs = {}
    if cfg.get('has_aux'):
        kwargs['aux_logits'] = True

    model = models.get_model(cfg['fn'], weights=None, **kwargs)
    if cfg.get('has_aux'):
        for attr in ('aux1', 'aux2', 'AuxLogits'):
            if hasattr(model, attr):
                setattr(model, attr, None)
        model.aux_logits = False

    model = replace_head(model, num_classes)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device).eval()
    print(f"[INFO] Loaded checkpoint: {ckpt_path.name}")
    return model, cfg['size']

def get_transforms(img_size):
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

def draw_hud(frame, model_name, device, top_class, top_conf, probabilities, classes, fps):
    h, w, _ = frame.shape
    
    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (15, 15, 20), -1)
    # Semi-transparent side panel for probability bars
    panel_w = 260
    cv2.rectangle(overlay, (w - panel_w, 95), (w, 95 + len(classes) * 35 + 10), (15, 15, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Top Header Text
    cv2.putText(frame, f"Model: {model_name} ({device.type.upper()})", (15, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    
    # Color palette for classes (metal=cyan, paper=green, plastic=orange/magenta)
    class_colors = {
        'metal': (255, 200, 50),     # Cyan-ish
        'paper': (50, 255, 100),     # Bright Green
        'plastic': (50, 150, 255)    # Orange/Yellow
    }
    
    color = class_colors.get(top_class.lower(), (0, 255, 255))
    cv2.putText(frame, f"Detected: {top_class.upper()} ({top_conf:.1f}%)", (15, 65), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    
    # FPS badge
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 110, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    # Class Probability Bars
    y_start = 120
    for idx, cls in enumerate(classes):
        prob = probabilities[idx]
        cls_color = class_colors.get(cls.lower(), (200, 200, 200))
        
        # Class Label
        cv2.putText(frame, f"{cls[:10].capitalize()}:", (w - panel_w + 10, y_start + idx * 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
        
        # Bar Background
        bar_x = w - panel_w + 90
        bar_max_w = 110
        cv2.rectangle(frame, (bar_x, y_start + idx * 35 - 12), (bar_x + bar_max_w, y_start + idx * 35), (60, 60, 60), -1)
        
        # Active Bar
        bar_fill = int(bar_max_w * prob)
        if bar_fill > 0:
            cv2.rectangle(frame, (bar_x, y_start + idx * 35 - 12), (bar_x + bar_fill, y_start + idx * 35), cls_color, -1)
            
        # Percentage Text
        cv2.putText(frame, f"{prob*100:.0f}%", (bar_x + bar_max_w + 10, y_start + idx * 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)

    # Center Targeting Reticle / ROI Box
    box_size = min(h, w) // 2
    x1, y1 = (w - box_size) // 2, (h - box_size) // 2
    x2, y2 = x1 + box_size, y1 + box_size
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    
    # Corner brackets for target box
    l = 20
    cv2.line(frame, (x1, y1), (x1 + l, y1), color, 4)
    cv2.line(frame, (x1, y1), (x1, y1 + l), color, 4)
    cv2.line(frame, (x2, y1), (x2 - l, y1), color, 4)
    cv2.line(frame, (x2, y1), (x2, y1 + l), color, 4)
    cv2.line(frame, (x1, y2), (x1 + l, y2), color, 4)
    cv2.line(frame, (x1, y2), (x1, y2 - l), color, 4)
    cv2.line(frame, (x2, y2), (x2 - l, y2), color, 4)
    cv2.line(frame, (x2, y2), (x2, y2 - l), color, 4)

    # Bottom Instructions
    cv2.putText(frame, "[M] Switch Model  |  [S] Save Frame  |  [Q] Quit", (15, h - 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

def main():
    parser = argparse.ArgumentParser(description="Real-time Camera Material Classifier")
    parser.add_argument("--model", type=str, default="ResNet-18", 
                        help="Model to use (e.g. ResNet-18, EfficientNet-B0, MobileNetV2, DenseNet-121, ConvNeXt-Tiny)")
    parser.add_argument("--cam", type=int, default=0, help="Camera device index (default: 0)")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using PyTorch Device: {device}")

    model_names = list(MODEL_CONFIGS.keys())
    current_model_idx = model_names.index(args.model) if args.model in model_names else 0

    model_name = model_names[current_model_idx]
    model, img_size = load_trained_model(model_name, len(DEFAULT_CLASSES), device)
    transform = get_transforms(img_size)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera device {args.cam}.")
        return

    print("\n[READY] Camera started!")
    print("  Controls:")
    print("   - Press 'M' to cycle through trained models")
    print("   - Press 'S' to save a screenshot of the detection")
    print("   - Press 'Q' or ESC to exit\n")

    fps, prev_time = 0.0, time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame from camera.")
            break

        # Flip horizontally for natural mirror effect
        frame = cv2.flip(frame, 1)

        # Convert OpenCV BGR frame to PIL RGB image for PyTorch transforms
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)

        # Preprocess & run inference
        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            if device.type == 'cuda':
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(input_tensor)
            else:
                outputs = model(input_tensor)

            probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()

        top_idx = int(np.argmax(probs))
        top_class = DEFAULT_CLASSES[top_idx]
        top_conf = probs[top_idx] * 100.0

        # Calculate FPS
        curr_time = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(curr_time - prev_time, 1e-5))
        prev_time = curr_time

        # Draw HUD overlays on frame
        draw_hud(frame, model_name, device, top_class, top_conf, probs, DEFAULT_CLASSES, fps)

        cv2.imshow("Material Classifier - Real-Time Detection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):  # 'q' or ESC
            break
        elif key in (ord('m'), ord('M')):
            # Switch to next available model
            current_model_idx = (current_model_idx + 1) % len(model_names)
            model_name = model_names[current_model_idx]
            print(f"\n[INFO] Switching model to: {model_name}...")
            try:
                model, img_size = load_trained_model(model_name, len(DEFAULT_CLASSES), device)
                transform = get_transforms(img_size)
            except Exception as e:
                print(f"[ERROR] Could not load model {model_name}: {e}")
        elif key in (ord('s'), ord('S')):
            filename = f"detection_{top_class}_{int(time.time())}.png"
            cv2.imwrite(filename, frame)
            print(f"[SAVED] Screenshot saved to: {filename}")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Camera stream closed.")

if __name__ == '__main__':
    main()
