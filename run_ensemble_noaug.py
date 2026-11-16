"""
CNN Ensemble (all 45 pairs) — PyTorch — no data augmentation
Trains 10 torchvision CNNs (GoogLeNet, ResNet-18, SqueezeNet 1.1, DenseNet-121,
MobileNetV2, ShuffleNetV2 x0.5, MNASNet 0.5, EfficientNet-B0, MobileNetV3-Small,
RegNetY-400MF), then builds every pair (10 choose 2 = 45 ensembles) by soft voting:
the two models' softmax probabilities are averaged and the argmax is the prediction.

Recorded for every model and every ensemble: accuracy, precision, recall, F1,
training time, inference time, model size, #params, learning rate, epochs, batch size,
loss function.

Usage:
    python run_ensemble_noaug.py

Place train/val/test image folders alongside this script (standard ImageFolder layout).
"""

import os
import gc
import time
import copy
import random
import traceback
import warnings
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.metrics import (precision_recall_fscore_support, accuracy_score,
                             confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split
from PIL import Image, UnidentifiedImageError

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ======================= CONFIG =======================
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "results"

EPOCHS       = 1
BATCH_SIZE   = 32              # lower to 16 if you hit CUDA OOM on big models
LR           = 1e-4            # 1e-4 is good for fine-tuning; use ~1e-3 if PRETRAINED = False
LR_OVERRIDES = {}              # per-model LR, e.g. {'ResNet-18': 5e-5}
WEIGHT_DECAY = 1e-4
PRETRAINED   = True            # True = ImageNet weights (transfer learning)
USE_AMP      = True            # mixed precision (faster, less memory on GPU)
NUM_WORKERS  = 4
SPLIT        = (0.70, 0.15, 0.15)  # only used when dataset has no train/val folders
PATIENCE     = None            # early stopping on val accuracy, e.g. 5 (None = off)
MEASURE_CPU_LATENCY = True
SEED         = 42
LOSS_NAME    = 'CrossEntropyLoss'
OPTIM_NAME   = 'AdamW'
# ======================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if DEVICE.type == 'cuda':
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision('high')
    except AttributeError:
        pass


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


set_seed(SEED)

for sub in ('checkpoints', 'history', 'reports', 'probs'):
    os.makedirs(OUT_DIR / sub, exist_ok=True)

# The 10 models to train and ensemble
MODELS = {
    'GoogLeNet':          dict(fn='googlenet', has_aux=True),
    'ResNet-18':          dict(fn='resnet18'),
    'SqueezeNet 1.1':     dict(fn='squeezenet1_1'),
    'DenseNet-121':       dict(fn='densenet121'),
    'MobileNetV2':        dict(fn='mobilenet_v2'),
    'ShuffleNetV2 x0.5':  dict(fn='shufflenet_v2_x0_5'),
    'MNASNet 0.5':        dict(fn='mnasnet0_5'),
    'EfficientNet-B0':    dict(fn='efficientnet_b0'),
    'MobileNetV3-Small':  dict(fn='mobilenet_v3_small'),
    'RegNetY-400MF':      dict(fn='regnet_y_400mf'),
}
assert len(MODELS) == 10

# Ensembling by averaging probabilities needs every model to see identical test images
assert len({cfg.get('size', 224) for cfg in MODELS.values()}) == 1, \
    'all models must share the same input size'


# ======================== DATA ========================

def find_dataset_dirs():
    """Resolves train, val, test subdirectories locally."""
    def locate_split(split_name):
        direct = BASE_DIR / split_name
        if direct.is_dir():
            return direct
        for item in BASE_DIR.glob(f"*{split_name}*"):
            if item.is_dir():
                sub = item / split_name
                if sub.is_dir():
                    return sub
                return item
        return None

    tr = locate_split('train')
    va = locate_split('val')
    te = locate_split('test')
    return tr, va, te


TRAIN_DIR, VAL_DIR, TEST_DIR = find_dataset_dirs()
HAS_SPLIT_DIRS = TRAIN_DIR is not None and VAL_DIR is not None

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def make_transforms(size):
    """
    Creates deterministic evaluation transforms (resize + center crop + normalize).
    No data augmentation is used for train, val, or test sets.
    """
    tf = transforms.Compose([
        transforms.Resize(int(size * 1.14)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return tf, tf


_BAD_FILES = set()


def _safe_loader(path):
    """
    Loads images safely, replacing corrupted/unreadable files with a black placeholder.
    """
    try:
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    except (UnidentifiedImageError, OSError) as e:
        if path not in _BAD_FILES:
            _BAD_FILES.add(path)
            print(f'  [WARN] unreadable image, using black placeholder: {path} ({e})')
        return Image.new('RGB', (256, 256), color=(0, 0, 0))


_SPLIT_IDX = {}


def _get_split_idx(data_root):
    if not _SPLIT_IDX:
        targets = np.array(datasets.ImageFolder(str(data_root)).targets)
        idx = np.arange(len(targets))
        tr, tmp = train_test_split(idx, test_size=1 - SPLIT[0], stratify=targets, random_state=SEED)
        val_share = SPLIT[1] / (SPLIT[1] + SPLIT[2])
        va, te = train_test_split(tmp, test_size=1 - val_share, stratify=targets[tmp], random_state=SEED)
        _SPLIT_IDX.update(train=tr, val=va, test=te)
    return _SPLIT_IDX


def make_loaders(size):
    train_tf, eval_tf = make_transforms(size)
    if HAS_SPLIT_DIRS:
        tr = datasets.ImageFolder(str(TRAIN_DIR), train_tf, loader=_safe_loader)
        va = datasets.ImageFolder(str(VAL_DIR), eval_tf, loader=_safe_loader)
        t_dir = TEST_DIR if TEST_DIR is not None else VAL_DIR
        te = datasets.ImageFolder(str(t_dir), eval_tf, loader=_safe_loader)
        classes = tr.classes
    else:
        s = _get_split_idx(BASE_DIR)
        full_train = datasets.ImageFolder(str(BASE_DIR), train_tf, loader=_safe_loader)
        full_eval = datasets.ImageFolder(str(BASE_DIR), eval_tf, loader=_safe_loader)
        tr, va, te = Subset(full_train, s['train']), Subset(full_eval, s['val']), Subset(full_eval, s['test'])
        classes = full_train.classes

    kw = dict(batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == 'cuda'),
              persistent_workers=NUM_WORKERS > 0)
    return (DataLoader(tr, shuffle=True, drop_last=len(tr) > BATCH_SIZE, **kw),
            DataLoader(va, shuffle=False, **kw),
            DataLoader(te, shuffle=False, **kw),
            classes)


# ============== MODEL BUILDER & HELPERS ===============

def build_model(cfg, num_classes):
    weights = 'DEFAULT' if PRETRAINED else None
    kwargs = dict(cfg.get('kwargs', {}))
    if cfg.get('has_aux'):
        kwargs['aux_logits'] = bool(PRETRAINED)
    model = models.get_model(cfg['fn'], weights=weights, **kwargs)
    if cfg.get('has_aux') and PRETRAINED:
        for attr in ('aux1', 'aux2', 'AuxLogits'):
            if hasattr(model, attr):
                setattr(model, attr, None)
        model.aux_logits = False
    return replace_head(model, num_classes)


def replace_head(model, num_classes):
    # SqueezeNet's classifier is a 1x1 conv, not a Linear layer
    if isinstance(model, models.SqueezeNet):
        model.classifier[1] = nn.Conv2d(model.classifier[1].in_channels, num_classes, kernel_size=1)
        model.num_classes = num_classes
        return model

    # every other model: swap its last nn.Linear
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


def get_logits(out):
    # Inception/GoogLeNet can return a namedtuple in train mode
    if hasattr(out, 'logits'):
        return out.logits
    if isinstance(out, tuple):
        return out[0]
    return out


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def sync():
    if DEVICE.type == 'cuda':
        torch.cuda.synchronize()


def run_epoch(model, loader, criterion, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    loss_sum, n = 0.0, 0
    preds, tgts = [], []
    amp_on = USE_AMP and DEVICE.type == 'cuda'
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=amp_on):
                logits = get_logits(model(x))
                loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            loss_sum += loss.item() * x.size(0)
            n += x.size(0)
            preds.append(logits.argmax(1).cpu())
            tgts.append(y.cpu())
    preds, tgts = torch.cat(preds).numpy(), torch.cat(tgts).numpy()
    return loss_sum / n, (preds == tgts).mean(), preds, tgts


@torch.no_grad()
def measure_latency(model, size, device, warmup=20, runs=100):
    """Single-image (batch=1) FP32 latency in milliseconds."""
    model = model.to(device).eval()
    x = torch.randn(1, 3, size, size, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs * 1000


def safe_name(label):
    return label.replace(' ', '_').replace('/', '-')


@torch.no_grad()
def predict_probs(model, loader):
    """Softmax probabilities for every image in the loader (loader must NOT shuffle)."""
    model.eval()
    probs, tgts = [], []
    amp_on = USE_AMP and DEVICE.type == 'cuda'
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=amp_on):
            logits = get_logits(model(x))
        probs.append(torch.softmax(logits.float(), dim=1).cpu())
        tgts.append(y)
    return torch.cat(probs).numpy(), torch.cat(tgts).numpy()


def save_test_probs(model, label, loader):
    probs, tgts = predict_probs(model, loader)
    os.makedirs(OUT_DIR / 'probs', exist_ok=True)
    np.savez_compressed(OUT_DIR / 'probs' / f'{safe_name(label)}.npz', probs=probs, targets=tgts)


# ================= TRAINING LOOP =====================

def train_model(label, cfg):
    size = cfg.get('size', 224)
    lr = LR_OVERRIDES.get(label, LR)
    set_seed(SEED)

    train_loader, val_loader, test_loader, classes = make_loaders(size)
    model = build_model(cfg, len(classes)).to(DEVICE)
    n_params, n_trainable = count_params(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP and DEVICE.type == 'cuda')

    safe = safe_name(label)
    ckpt_path = OUT_DIR / 'checkpoints' / f'{safe}.pth'
    best_acc, best_epoch, bad_epochs = -1.0, 0, 0
    train_time, history = 0.0, []

    print(f'\n===== {label} | params {n_params/1e6:.2f}M | input {size}px | lr {lr} =====')
    for epoch in range(1, EPOCHS + 1):
        sync(); t0 = time.perf_counter()
        tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, criterion, optimizer, scaler)
        sync(); ep_time = time.perf_counter() - t0
        train_time += ep_time
        scheduler.step()

        va_loss, va_acc, _, _ = run_epoch(model, val_loader, criterion)
        history.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc,
                            val_loss=va_loss, val_acc=va_acc, epoch_time_sec=ep_time))
        print(f'  ep {epoch:02d}/{EPOCHS} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | '
              f'val loss {va_loss:.4f} acc {va_acc:.4f} | {ep_time:.1f}s')

        if va_acc > best_acc:
            best_acc, best_epoch, bad_epochs = va_acc, epoch, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            bad_epochs += 1
            if PATIENCE and bad_epochs >= PATIENCE:
                print(f'  early stopping at epoch {epoch}')
                break

    # ---- test with the best checkpoint ----
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    te_loss, te_acc, preds, tgts = run_epoch(model, test_loader, criterion)
    p_m, r_m, f_m, _ = precision_recall_fscore_support(tgts, preds, average='macro', zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(tgts, preds, average='weighted', zero_division=0)

    # ---- save test-set probabilities (used later to build the pairwise ensembles) ----
    save_test_probs(model, label, test_loader)

    # ---- inference time (batch=1, FP32) ----
    lat_gpu = measure_latency(copy.deepcopy(model), size, DEVICE) if DEVICE.type == 'cuda' else np.nan
    lat_cpu = measure_latency(copy.deepcopy(model).cpu(), size, torch.device('cpu'), 5, 30) \
              if MEASURE_CPU_LATENCY else np.nan

    # ---- save extras ----
    pd.DataFrame(history).to_csv(OUT_DIR / 'history' / f'{safe}.csv', index=False)
    with open(OUT_DIR / 'reports' / f'{safe}_report.txt', 'w') as f:
        f.write(classification_report(tgts, preds, target_names=classes, zero_division=0))
        f.write('\nConfusion matrix:\n')
        f.write(np.array2string(confusion_matrix(tgts, preds)))

    row = {
        'model': label,
        'accuracy': te_acc,
        'precision_macro': p_m, 'recall_macro': r_m, 'f1_macro': f_m,
        'precision_weighted': p_w, 'recall_weighted': r_w, 'f1_weighted': f_w,
        'test_loss': te_loss,
        'best_val_acc': best_acc, 'best_epoch': best_epoch,
        'train_time_sec': train_time,
        'train_time_per_epoch_sec': train_time / len(history),
        'inference_ms_gpu_bs1': lat_gpu,
        'inference_ms_cpu_bs1': lat_cpu,
        'model_size_mb': os.path.getsize(ckpt_path) / 1e6,
        'params_total': n_params, 'params_millions': n_params / 1e6, 'params_trainable': n_trainable,
        'learning_rate': lr, 'epochs_planned': EPOCHS, 'epochs_run': len(history),
        'batch_size': BATCH_SIZE, 'loss_function': LOSS_NAME, 'optimizer': OPTIM_NAME,
        'input_size': size, 'pretrained': PRETRAINED, 'mixed_precision': USE_AMP, 'augmentation': 'none',
        'num_classes': len(classes),
    }
    print(f'  >> TEST acc {te_acc:.4f} | P {p_m:.4f} R {r_m:.4f} F1 {f_m:.4f} | '
          f'GPU {lat_gpu:.1f} ms | CPU {lat_cpu:.1f} ms | {row["model_size_mb"]:.1f} MB')
    del model, optimizer, scaler
    return row


# =============== PLOTTING HELPERS =====================

def generate_individual_plots(df):
    """Summary bar/scatter plots for the 10 individual models."""
    fig, ax = plt.subplots(2, 2, figsize=(16, 11))
    d = df.sort_values('accuracy')

    ax[0, 0].barh(d['model'], d['accuracy'])
    ax[0, 0].set_title('Test accuracy')
    ax[0, 1].barh(d['model'], d['train_time_sec'] / 60)
    ax[0, 1].set_title('Training time (min)')

    if 'inference_ms_gpu_bs1' in df and not df['inference_ms_gpu_bs1'].isna().all():
        ax[1, 0].scatter(df['inference_ms_gpu_bs1'], df['accuracy'])
        for _, r in df.iterrows():
            ax[1, 0].annotate(r['model'], (r['inference_ms_gpu_bs1'], r['accuracy']), fontsize=8)
        ax[1, 0].set_xlabel('GPU latency, batch=1 (ms)')
    else:
        ax[1, 0].scatter(df['inference_ms_cpu_bs1'], df['accuracy'])
        for _, r in df.iterrows():
            ax[1, 0].annotate(r['model'], (r['inference_ms_cpu_bs1'], r['accuracy']), fontsize=8)
        ax[1, 0].set_xlabel('CPU latency, batch=1 (ms)')
    ax[1, 0].set_ylabel('Accuracy')
    ax[1, 0].set_title('Accuracy vs. latency')

    ax[1, 1].scatter(df['params_millions'], df['accuracy'])
    for _, r in df.iterrows():
        ax[1, 1].annotate(r['model'], (r['params_millions'], r['accuracy']), fontsize=8)
    ax[1, 1].set_xscale('log')
    ax[1, 1].set_xlabel('Parameters (millions, log)')
    ax[1, 1].set_ylabel('Accuracy')
    ax[1, 1].set_title('Accuracy vs. size')

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'summary_plots.png', dpi=150)
    plt.close()


def generate_validation_curves(df):
    """Validation loss/accuracy curves for every trained model."""
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    for label in df['model']:
        path = OUT_DIR / 'history' / f'{safe_name(label)}.csv'
        if not path.exists():
            print('missing history:', path)
            continue
        h = pd.read_csv(path)
        ax[0].plot(h['epoch'], h['val_loss'], marker='o', label=label)
        ax[1].plot(h['epoch'], h['val_acc'], marker='o', label=label)
    ax[0].set_title('Validation loss')
    ax[1].set_title('Validation accuracy')
    for a_ in ax:
        a_.set_xlabel('epoch')
    ax[1].legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'validation_curves.png', dpi=150)
    plt.close()


def generate_ensemble_plots(ens, ind, P):
    """Heatmap of pair accuracies and accuracy-vs-latency scatter for ensembles."""
    names = [m for m in MODELS if m in P]
    idx = {m: i for i, m in enumerate(names)}
    M = np.full((len(names), len(names)), np.nan)
    for m in names:
        M[idx[m], idx[m]] = ind[m]['accuracy']
    for _, r in ens.iterrows():
        i, j = idx[r['model_a']], idx[r['model_b']]
        M[i, j] = M[j, i] = r['accuracy']

    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    im = ax[0].imshow(M, cmap='viridis')
    ax[0].set_xticks(range(len(names)))
    ax[0].set_xticklabels(names, rotation=60, ha='right', fontsize=8)
    ax[0].set_yticks(range(len(names)))
    ax[0].set_yticklabels(names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            if not np.isnan(M[i, j]):
                ax[0].text(j, i, f'{M[i, j]:.3f}', ha='center', va='center', fontsize=6, color='white')
    ax[0].set_title('Test accuracy: pair ensembles (off-diagonal) vs. single models (diagonal)')
    plt.colorbar(im, ax=ax[0], fraction=0.046)

    lat_col = 'inference_ms_gpu_bs1'
    if ens[lat_col].isna().all():
        lat_col = 'inference_ms_cpu_bs1'
    ax[1].scatter(ens[lat_col], ens['accuracy'], s=25)
    for _, r in ens.nlargest(6, 'accuracy').iterrows():
        ax[1].annotate(r['ensemble'], (r[lat_col], r['accuracy']), fontsize=7)
    device_label = 'GPU' if 'gpu' in lat_col else 'CPU'
    ax[1].set_xlabel(f'Summed {device_label} latency of both members, batch=1 (ms)')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title('Ensemble accuracy vs. inference time (top 6 labelled)')

    plt.tight_layout()
    plt.savefig(OUT_DIR / 'ensemble_summary_plots.png', dpi=150)
    plt.close()


# ============ STEP 2: PAIRWISE ENSEMBLES =============

def load_test_probs(label):
    """Load saved test probabilities; if missing, rebuild from the checkpoint."""
    path = OUT_DIR / 'probs' / f'{safe_name(label)}.npz'
    if not path.exists():
        ckpt = OUT_DIR / 'checkpoints' / f'{safe_name(label)}.pth'
        if not ckpt.exists():
            raise FileNotFoundError(f'no probabilities and no checkpoint for {label}')
        print(f'  rebuilding test probabilities for {label} from its checkpoint...')
        cfg = MODELS[label]
        _, _, test_loader, classes = make_loaders(cfg.get('size', 224))
        model = build_model(cfg, len(classes)).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        save_test_probs(model, label, test_loader)
        del model
        gc.collect()
        if DEVICE.type == 'cuda':
            torch.cuda.empty_cache()
    d = np.load(path)
    return d['probs'], d['targets']


def same_or_pair(x, y):
    return x if x == y else f'{x}/{y}'


def build_ensembles(results_csv):
    """Build all 45 pairwise ensembles from saved test probabilities."""
    ind = {r['model']: r for r in pd.read_csv(results_csv).to_dict('records') if r['model'] in MODELS}
    missing = [m for m in MODELS if m not in ind]
    if missing:
        print('WARNING — these models have no results yet, so pairs containing them are skipped:', missing)

    # Load test probabilities for every available model
    P, T = {}, None
    for m in MODELS:
        if m not in ind:
            continue
        try:
            P[m], t = load_test_probs(m)
        except Exception as e:
            print(f'  skipping {m}: {e}')
            continue
        if T is None:
            T = t
        assert np.array_equal(T, t), f'{m}: test set / ordering differs from the other models'

    PAIRS = list(combinations(MODELS.keys(), 2))
    assert len(PAIRS) == 45
    print(f'{len(P)} models usable -> {sum(a in P and b in P for a, b in PAIRS)} of {len(PAIRS)} pairs can be evaluated')

    ens_rows = []
    for a, b in PAIRS:
        if a not in P or b not in P:
            continue
        p = (P[a] + P[b]) / 2.0
        preds = p.argmax(1)
        acc = float((preds == T).mean())
        pm, rm, fm, _ = precision_recall_fscore_support(T, preds, average='macro', zero_division=0)
        pw, rw, fw, _ = precision_recall_fscore_support(T, preds, average='weighted', zero_division=0)
        nll = float(-np.log(np.clip(p[np.arange(len(T)), T], 1e-12, None)).mean())
        ra, rb = ind[a], ind[b]
        ens_rows.append({
            'ensemble': f'{a} + {b}', 'model_a': a, 'model_b': b,
            'combine_method': 'soft voting (mean of softmax probabilities)',
            'accuracy': acc,
            'precision_macro': pm, 'recall_macro': rm, 'f1_macro': fm,
            'precision_weighted': pw, 'recall_weighted': rw, 'f1_weighted': fw,
            'test_loss': nll,
            'acc_member_a': ra['accuracy'], 'acc_member_b': rb['accuracy'],
            'gain_over_best_member': acc - max(ra['accuracy'], rb['accuracy']),
            'train_time_sec': ra['train_time_sec'] + rb['train_time_sec'],
            'inference_ms_gpu_bs1': ra['inference_ms_gpu_bs1'] + rb['inference_ms_gpu_bs1'],
            'inference_ms_cpu_bs1': ra['inference_ms_cpu_bs1'] + rb['inference_ms_cpu_bs1'],
            'model_size_mb': ra['model_size_mb'] + rb['model_size_mb'],
            'params_total': ra['params_total'] + rb['params_total'],
            'params_millions': ra['params_millions'] + rb['params_millions'],
            'learning_rate': same_or_pair(ra['learning_rate'], rb['learning_rate']),
            'epochs_planned': EPOCHS,
            'epochs_run_member_a': ra['epochs_run'], 'epochs_run_member_b': rb['epochs_run'],
            'batch_size': BATCH_SIZE,
            'loss_function': f'{LOSS_NAME} (members); NLL of averaged probabilities (ensemble)',
            'optimizer': OPTIM_NAME, 'input_size': ra['input_size'],
            'pretrained': PRETRAINED, 'augmentation': 'none',
        })

    ens_csv = OUT_DIR / 'ensemble_results.csv'
    ens = pd.DataFrame(ens_rows)
    ens.to_csv(ens_csv, index=False)
    print(f'Saved {len(ens)} ensembles -> {ens_csv}')
    return ens, ind, P


# ==================== MAIN ============================

def main():
    gpu_name = f" ({torch.cuda.get_device_name(0)})" if DEVICE.type == 'cuda' else ''
    print(f"Device: {DEVICE}{gpu_name}")
    print(f"Dataset resolution:")
    print(f"  Train dir: {TRAIN_DIR}")
    print(f"  Val dir:   {VAL_DIR}")
    print(f"  Test dir:  {TEST_DIR}")

    _tl, _vl, _tel, CLASSES = make_loaders(224)
    print(f'\nClasses ({len(CLASSES)}): {CLASSES}')
    print(f'Train samples: {len(_tl.dataset)} | Val samples: {len(_vl.dataset)} | Test samples: {len(_tel.dataset)}')
    if HAS_SPLIT_DIRS and TEST_DIR is None:
        print('WARNING: no test/ folder found -> val/ is being used as the test set.')
    del _tl, _vl, _tel

    # ---- Step 1: train the 10 individual models ----
    RESULTS_CSV = OUT_DIR / 'results.csv'
    rows = pd.read_csv(RESULTS_CSV).to_dict('records') if RESULTS_CSV.exists() else []
    done = {r['model'] for r in rows}

    for label, cfg in MODELS.items():
        if label in done:
            print(f'Skipping {label} (already in results.csv)')
            continue
        try:
            rows.append(train_model(label, cfg))
            pd.DataFrame(rows).to_csv(RESULTS_CSV, index=False)
        except Exception:
            print(f'!! {label} FAILED (it will be retried on the next run):')
            traceback.print_exc()
        finally:
            gc.collect()
            if DEVICE.type == 'cuda':
                torch.cuda.empty_cache()

    print('\nDone training. Results:', RESULTS_CSV)
    if _BAD_FILES:
        print(f'\n[WARNING] {len(_BAD_FILES)} unreadable image file(s) replaced with black placeholder:')
        for p in sorted(_BAD_FILES):
            print('   ', p)
        print('Delete or re-export these files, then rerun for clean results.')

    # ---- Individual model plots ----
    if RESULTS_CSV.exists():
        df = pd.read_csv(RESULTS_CSV)
        df = df[df['model'].isin(MODELS.keys())].reset_index(drop=True)
        generate_individual_plots(df)
        generate_validation_curves(df)

        print("\nIndividual model results:")
        show_cols = ['model', 'accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
                     'train_time_sec', 'inference_ms_gpu_bs1', 'inference_ms_cpu_bs1',
                     'model_size_mb', 'params_millions', 'learning_rate', 'epochs_run',
                     'batch_size', 'loss_function']
        pd.set_option('display.width', 250)
        pd.set_option('display.max_columns', None)
        print(df.sort_values('accuracy', ascending=False)[show_cols].reset_index(drop=True).round(4).to_string())

    # ---- Step 2: all 45 pairwise ensembles ----
    print('\n' + '=' * 60)
    print('Step 2: Building all 45 pairwise ensembles (soft voting)')
    print('=' * 60)
    ens, ind, P = build_ensembles(RESULTS_CSV)

    # ---- Ensemble plots ----
    generate_ensemble_plots(ens, ind, P)

    # ---- Print ensemble results ----
    print("\nEnsemble results:")
    ens_cols = ['ensemble', 'accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
                'gain_over_best_member', 'train_time_sec', 'inference_ms_gpu_bs1', 'inference_ms_cpu_bs1',
                'model_size_mb', 'params_millions', 'learning_rate', 'epochs_planned', 'batch_size', 'loss_function']
    pd.set_option('display.max_rows', 100)
    print(ens.sort_values(['accuracy', 'f1_macro'], ascending=False)[ens_cols].reset_index(drop=True).round(4).to_string())

    print(f'\nAll results saved to: {OUT_DIR}')


if __name__ == '__main__':
    main()
