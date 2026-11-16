"""
Vision Benchmark & Ensemble Studio — Streamlit Application
A clean, simple, and intuitive UI to inspect CNN benchmarks,
evaluate 45 pairwise soft-voting ensembles, and run live image classification.

Usage:
    streamlit run app.py
"""

import os
import glob
import time
from pathlib import Path
from PIL import Image

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ======================= PAGE CONFIG =======================
st.set_page_config(
    page_title="Vision Benchmark & Ensemble Studio",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Clean CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 18px;
        border-radius: 8px;
        font-weight: 500;
    }
    .badge-gain {
        background-color: #DCFCE7;
        color: #166534;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-loss {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 3px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent

# Available experiment profiles across newtrain and limited folders
EXPERIMENT_PRESETS = {
    "GPU — With Augmentation (17 Models)": BASE_DIR / "newtrain" / "indiv_aug" / "results",
    "GPU — No Augmentation (17 Models)": BASE_DIR / "newtrain" / "indiv_noaug" / "results",
    "GPU — 45 Ensembles (10 Models + 45 Pairs)": BASE_DIR / "newtrain" / "ensemble_aug" / "results",
    "CPU Low-Spec — With Augmentation (17 Models)": BASE_DIR / "limited" / "indiv_aug" / "results",
    "CPU Low-Spec — No Augmentation (17 Models)": BASE_DIR / "limited" / "indiv_noaug" / "results",
    "CPU Low-Spec — 45 Ensembles (10 Models + 45 Pairs)": BASE_DIR / "limited" / "ensemble_aug" / "results",
}

# The 17 torchvision model configurations
MODEL_CONFIGS = {
    'AlexNet':            dict(fn='alexnet', size=224),
    'VGG-11':             dict(fn='vgg11', size=224),
    'GoogLeNet':          dict(fn='googlenet', size=224, has_aux=True),
    'ResNet-18':          dict(fn='resnet18', size=224),
    'Inception-v3':       dict(fn='inception_v3', size=299, has_aux=True),
    'SqueezeNet 1.1':     dict(fn='squeezenet1_1', size=224),
    'Wide ResNet-50-2':   dict(fn='wide_resnet50_2', size=224),
    'DenseNet-121':       dict(fn='densenet121', size=224),
    'ResNeXt-50 32x4d':   dict(fn='resnext50_32x4d', size=224),
    'MobileNetV2':        dict(fn='mobilenet_v2', size=224),
    'ShuffleNetV2 x0.5':  dict(fn='shufflenet_v2_x0_5', size=224),
    'MNASNet 0.5':        dict(fn='mnasnet0_5', size=224),
    'EfficientNet-B0':    dict(fn='efficientnet_b0', size=224),
    'MobileNetV3-Small':  dict(fn='mobilenet_v3_small', size=224),
    'RegNetY-400MF':      dict(fn='regnet_y_400mf', size=224),
    'EfficientNetV2-S':   dict(fn='efficientnet_v2_s', size=224),
    'ConvNeXt-Tiny':      dict(fn='convnext_tiny', size=224),
}

DEFAULT_CLASSES = ['metal', 'paper', 'plastic']


# ======================= DATA HELPERS =======================

@st.cache_data
def load_benchmark_data(results_dir: Path):
    """Loads results.csv, ensemble_results.csv, and checks for plots and reports."""
    res_csv = results_dir / "results.csv"
    ens_csv = results_dir / "ensemble_results.csv"

    df_indiv = pd.read_csv(res_csv) if res_csv.exists() else pd.DataFrame()
    df_ens = pd.read_csv(ens_csv) if ens_csv.exists() else pd.DataFrame()

    return df_indiv, df_ens


def find_sample_images():
    """Finds available sample images from dataset test or val folders."""
    sample_images = {}
    valid_exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')

    # Look in root dataset folders
    search_roots = [
        BASE_DIR / "test-20261007T152143Z-1-001" / "test",
        BASE_DIR / "val-20261007T145448Z-1-001" / "val",
        BASE_DIR / "train-20261007T150600Z-1-001" / "train",
    ]

    for root in search_roots:
        if root.is_dir():
            for cls_dir in root.iterdir():
                if cls_dir.is_dir():
                    cls_name = cls_dir.name
                    files = []
                    for ext in valid_exts:
                        files.extend(list(cls_dir.glob(ext)))
                    if files:
                        for f in files[:4]:  # Collect up to 4 samples per class
                            label = f"{cls_name.capitalize()} - {f.name}"
                            sample_images[label] = f
            if sample_images:
                break
    return sample_images


# ======================= MODEL LOADER & INFERENCE =======================

@st.cache_resource
def get_pytorch_model(model_name: str, ckpt_path: str, num_classes: int = 3):
    """Loads a PyTorch model and weights for inference."""
    try:
        import torch
        import torch.nn as nn
        from torchvision import models

        cfg = MODEL_CONFIGS.get(model_name)
        if not cfg:
            return None

        kwargs = {}
        if cfg.get('has_aux'):
            kwargs['aux_logits'] = False

        model = models.get_model(cfg['fn'], weights=None, **kwargs)

        # Replace classification head
        if isinstance(model, models.SqueezeNet):
            model.classifier[1] = nn.Conv2d(model.classifier[1].in_channels, num_classes, kernel_size=1)
            model.num_classes = num_classes
        else:
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

        # Load weights on CPU for responsive UI inference
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading {model_name}: {e}")
        return None


def run_single_inference(model, image: Image.Image, size: int = 224):
    """Runs single model inference and returns class probabilities."""
    import torch
    from torchvision import transforms

    tf = transforms.Compose([
        transforms.Resize(int(size * 1.14)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    x = tf(image.convert('RGB')).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        logits = out.logits if hasattr(out, 'logits') else (out[0] if isinstance(out, tuple) else out)
        probs = torch.softmax(logits.float(), dim=1).squeeze(0).numpy()
    return probs


# ======================= SIDEBAR =======================

with st.sidebar:
    st.image("https://img.icons8.com/isometric/100/artificial-intelligence.png", width=64)
    st.title("Settings")

    # Select experiment
    preset_names = list(EXPERIMENT_PRESETS.keys())
    selected_preset = st.selectbox(
        "Choose Experiment Profile:",
        preset_names,
        index=0,
        help="Select which folder's results to explore."
    )
    results_dir = EXPERIMENT_PRESETS[selected_preset]

    st.markdown("---")
    st.caption("📂 **Selected Results Directory:**")
    st.code(str(results_dir), language="text")

    df_indiv, df_ens = load_benchmark_data(results_dir)

    # Status indicators
    has_indiv = not df_indiv.empty
    has_ens = not df_ens.empty
    has_ckpts = (results_dir / "checkpoints").is_dir() and any((results_dir / "checkpoints").glob("*.pth"))

    st.markdown("### Profile Status")
    st.write("📊 Benchmark CSV:", "✅ Loaded" if has_indiv else "⚠️ Missing")
    st.write("🤝 Ensembles CSV:", "✅ Loaded" if has_ens else "ℹ️ Not present")
    st.write("💾 Model Weights:", "✅ Available" if has_ckpts else "⚠️ No checkpoints")

    st.markdown("---")
    st.caption("💡 **Tip:** Switch profiles anytime to compare GPU vs Low-Spec CPU benchmarks.")


# ======================= MAIN HEADER & KPIS =======================

st.markdown('<div class="main-title">🌿 Vision Benchmark & Ensemble Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Interactive exploration of CNN transfer learning benchmarks, 45 pairwise soft-voting ensembles, and real-time inference.</div>', unsafe_allow_html=True)

# Top KPI Metric Cards
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

if has_indiv:
    best_indiv = df_indiv.loc[df_indiv['accuracy'].idxmax()]
    fastest_model = df_indiv.loc[df_indiv['inference_ms_cpu_bs1'].idxmin()] if 'inference_ms_cpu_bs1' in df_indiv.columns else None
    smallest_model = df_indiv.loc[df_indiv['params_millions'].idxmin()]

    with kpi1:
        st.metric(
            label="Top Model Accuracy",
            value=f"{best_indiv['accuracy'] * 100:.1f}%",
            delta=f"{best_indiv['model']}"
        )
    with kpi2:
        if has_ens:
            best_ens = df_ens.loc[df_ens['accuracy'].idxmax()]
            st.metric(
                label="Top Ensemble Accuracy",
                value=f"{best_ens['accuracy'] * 100:.1f}%",
                delta=f"{best_ens['ensemble']}"
            )
        else:
            st.metric(
                label="Models Evaluated",
                value=f"{len(df_indiv)} models",
                delta="Complete"
            )
    with kpi3:
        if fastest_model is not None:
            st.metric(
                label="Lowest CPU Latency",
                value=f"{fastest_model['inference_ms_cpu_bs1']:.1f} ms",
                delta=f"{fastest_model['model']}"
            )
        else:
            st.metric(label="Inference Latency", value="N/A")
    with kpi4:
        st.metric(
            label="Most Lightweight Model",
            value=f"{smallest_model['params_millions']:.2f} M",
            delta=f"{smallest_model['model']} ({smallest_model['model_size_mb']:.1f} MB)"
        )
else:
    st.info(f"No `results.csv` found in `{results_dir}`. Please run a benchmark first or select an existing experiment from the sidebar.")


# ======================= TABS =======================

tab_lead, tab_ens, tab_infer, tab_plots = st.tabs([
    "📊 Leaderboard & Models",
    "🤝 Pairwise Ensembles (45 Pairs)",
    "🖼️ Live Image Inference",
    "📈 Graphs & Validation Curves",
])

# ----------------------------------------------------
# TAB 1: LEADERBOARD & MODEL DEEP-DIVE
# ----------------------------------------------------
with tab_lead:
    if has_indiv:
        st.subheader("Model Benchmark Leaderboard")

        # Filters
        fcol1, fcol2 = st.columns([3, 1])
        with fcol1:
            search_query = st.text_input("🔍 Search model name:", placeholder="e.g. ResNet, MobileNet, DenseNet...")
        with fcol2:
            sort_by = st.selectbox("Sort by:", ["accuracy", "inference_ms_cpu_bs1", "params_millions", "train_time_sec"], index=0)

        # Filter and format DataFrame
        display_df = df_indiv.copy()
        if search_query:
            display_df = display_df[display_df['model'].str.contains(search_query, case=False, na=False)]

        ascending = (sort_by in ["inference_ms_cpu_bs1", "params_millions", "train_time_sec"])
        display_df = display_df.sort_values(by=sort_by, ascending=ascending)

        # Prettify table columns
        cols_to_show = [
            'model', 'accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
            'inference_ms_cpu_bs1', 'params_millions', 'model_size_mb', 'train_time_sec'
        ]
        available_cols = [c for c in cols_to_show if c in display_df.columns]
        view_table = display_df[available_cols].copy()

        view_table['accuracy'] = view_table['accuracy'].apply(lambda x: f"{x*100:.2f}%")
        view_table['precision_macro'] = view_table['precision_macro'].apply(lambda x: f"{x*100:.2f}%")
        view_table['recall_macro'] = view_table['recall_macro'].apply(lambda x: f"{x*100:.2f}%")
        view_table['f1_macro'] = view_table['f1_macro'].apply(lambda x: f"{x*100:.2f}%")
        if 'inference_ms_cpu_bs1' in view_table.columns:
            view_table['inference_ms_cpu_bs1'] = view_table['inference_ms_cpu_bs1'].apply(lambda x: f"{x:.1f} ms")
        view_table['params_millions'] = view_table['params_millions'].apply(lambda x: f"{x:.2f} M")
        view_table['model_size_mb'] = view_table['model_size_mb'].apply(lambda x: f"{x:.1f} MB")
        view_table['train_time_sec'] = view_table['train_time_sec'].apply(lambda x: f"{x/60:.1f} min")

        st.dataframe(view_table, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Visual comparison charts
        st.subheader("Visual Comparisons")
        c1, c2 = st.columns(2)

        with c1:
            st.caption("Accuracy across Models")
            chart_df = df_indiv.set_index('model')[['accuracy']] * 100
            st.bar_chart(chart_df.sort_values('accuracy', ascending=True))

        with c2:
            st.caption("CPU Inference Latency (Batch Size = 1)")
            if 'inference_ms_cpu_bs1' in df_indiv.columns:
                lat_df = df_indiv.set_index('model')[['inference_ms_cpu_bs1']]
                st.bar_chart(lat_df.sort_values('inference_ms_cpu_bs1', ascending=False))

        # Model Deep Dive Section
        st.markdown("---")
        st.subheader("Model Deep-Dive Inspector")
        selected_model = st.selectbox("Inspect specific model:", df_indiv['model'].tolist())

        if selected_model:
            m_safe = selected_model.replace(" ", "_").replace("/", "-")
            rep_col1, rep_col2 = st.columns(2)

            with rep_col1:
                st.markdown(f"**Classification Report & Confusion Matrix ({selected_model})**")
                report_file = results_dir / "reports" / f"{m_safe}_report.txt"
                if report_file.exists():
                    with open(report_file, 'r') as f:
                        st.code(f.read(), language="text")
                else:
                    st.info("Report file not found.")

            with rep_col2:
                st.markdown(f"**Training & Validation History ({selected_model})**")
                history_file = results_dir / "history" / f"{m_safe}.csv"
                if history_file.exists():
                    h_df = pd.read_csv(history_file)
                    st.line_chart(h_df.set_index('epoch')[['train_acc', 'val_acc']])
                    st.line_chart(h_df.set_index('epoch')[['train_loss', 'val_loss']])
                else:
                    st.info("History CSV not found.")
    else:
        st.warning("No individual model results available for this profile.")


# ----------------------------------------------------
# TAB 2: PAIRWISE ENSEMBLES (45 PAIRS)
# ----------------------------------------------------
with tab_ens:
    if has_ens:
        st.subheader("Pairwise Soft-Voting Ensemble Explorer (10 Choose 2 = 45 Pairs)")
        st.caption("Soft voting averages the class probability outputs of both models: $\\text{Prediction} = \\operatorname{argmax} \\left(\\frac{P_A + P_B}{2}\\right)$")

        # Interactive Pair Compare Tool
        st.markdown("#### 🔬 Compare Any Model Pair")
        col_pa, col_pb = st.columns(2)

        unique_models = sorted(list(set(df_ens['model_a'].tolist() + df_ens['model_b'].tolist())))
        with col_pa:
            mod_a = st.selectbox("Select Model A:", unique_models, index=0)
        with col_pb:
            other_models = [m for m in unique_models if m != mod_a]
            mod_b = st.selectbox("Select Model B:", other_models, index=0)

        # Match pair row
        match = df_ens[((df_ens['model_a'] == mod_a) & (df_ens['model_b'] == mod_b)) |
                       ((df_ens['model_a'] == mod_b) & (df_ens['model_b'] == mod_a))]

        if not match.empty:
            erow = match.iloc[0]
            gain = erow['gain_over_best_member'] * 100
            gain_badge = f'<span class="badge-gain">+{gain:.2f}% Synergy Gain</span>' if gain >= 0 else f'<span class="badge-loss">{gain:.2f}% Drop</span>'

            card_col1, card_col2, card_col3 = st.columns(3)
            with card_col1:
                st.metric(f"Model A: {erow['model_a']}", f"{erow['acc_member_a']*100:.2f}%")
            with card_col2:
                st.metric(f"Model B: {erow['model_b']}", f"{erow['acc_member_b']*100:.2f}%")
            with card_col3:
                st.metric(f"Ensemble: {erow['ensemble']}", f"{erow['accuracy']*100:.2f}%")

            st.markdown(f"**Performance Delta:** {gain_badge}", unsafe_allow_html=True)
            st.write(f"⚡ **Combined CPU Latency:** `{erow['inference_ms_cpu_bs1']:.1f} ms` | 📦 **Combined Parameters:** `{erow['params_millions']:.2f} M`")

        st.markdown("---")

        # All 45 Ensembles Table
        st.subheader("All 45 Ensemble Pairs Leaderboard")
        sort_ens = st.selectbox("Sort ensembles by:", ["accuracy", "gain_over_best_member", "inference_ms_cpu_bs1", "params_millions"])
        asc_ens = (sort_ens in ["inference_ms_cpu_bs1", "params_millions"])

        ens_view = df_ens.sort_values(by=sort_ens, ascending=asc_ens).copy()
        ens_view['accuracy'] = ens_view['accuracy'].apply(lambda x: f"{x*100:.2f}%")
        ens_view['gain_over_best_member'] = ens_view['gain_over_best_member'].apply(lambda x: f"{'+' if x>=0 else ''}{x*100:.2f}%")
        ens_view['f1_macro'] = ens_view['f1_macro'].apply(lambda x: f"{x*100:.2f}%")
        if 'inference_ms_cpu_bs1' in ens_view.columns:
            ens_view['inference_ms_cpu_bs1'] = ens_view['inference_ms_cpu_bs1'].apply(lambda x: f"{x:.1f} ms")
        ens_view['params_millions'] = ens_view['params_millions'].apply(lambda x: f"{x:.2f} M")

        show_cols = ['ensemble', 'accuracy', 'gain_over_best_member', 'f1_macro', 'inference_ms_cpu_bs1', 'params_millions']
        st.dataframe(ens_view[show_cols], use_container_width=True, hide_index=True)

    else:
        st.info("No ensemble results found for this profile. Switch to an Ensemble profile in the sidebar to explore the 45 soft-voting pairs.")


# ----------------------------------------------------
# TAB 3: LIVE IMAGE INFERENCE
# ----------------------------------------------------
with tab_infer:
    st.subheader("Live Image Classification & Soft-Voting")
    st.caption("Upload an image or pick a test sample to test inference live using your saved model weights.")

    sample_dict = find_sample_images()

    inf_col1, inf_col2 = st.columns([1, 1])

    with inf_col1:
        st.markdown("##### 1. Choose Image")
        input_type = st.radio("Image Source:", ["Dataset Sample", "Upload Your Own"], horizontal=True)

        input_img = None
        img_label = ""

        if input_type == "Dataset Sample":
            if sample_dict:
                sel_sample = st.selectbox("Select Test Image:", list(sample_dict.keys()))
                if sel_sample:
                    input_img = Image.open(sample_dict[sel_sample])
                    img_label = sel_sample
            else:
                st.info("No sample images detected in local dataset folder.")
        else:
            uploaded_file = st.file_uploader("Upload an image (PNG, JPG, WEBP):", type=["jpg", "jpeg", "png", "webp"])
            if uploaded_file:
                input_img = Image.open(uploaded_file)
                img_label = uploaded_file.name

        if input_img is not None:
            st.image(input_img, caption=img_label, use_container_width=True)

    with inf_col2:
        st.markdown("##### 2. Choose Model Mode")
        mode = st.radio("Inference Mode:", ["Single Model", "2-Model Soft Voting Ensemble"], horizontal=True)

        ckpt_dir = results_dir / "checkpoints"
        available_models = [m for m in MODEL_CONFIGS.keys() if (ckpt_dir / f"{m.replace(' ', '_').replace('/', '-')}.pth").exists()]

        if not available_models:
            # Fallback to general checkpoints if current folder doesn't have them
            alt_ckpt = BASE_DIR / "newtrain" / "indiv_aug" / "results" / "checkpoints"
            if alt_ckpt.is_dir():
                available_models = [m for m in MODEL_CONFIGS.keys() if (alt_ckpt / f"{m.replace(' ', '_').replace('/', '-')}.pth").exists()]
                ckpt_dir = alt_ckpt

        if not available_models:
            st.warning("⚠️ No `.pth` checkpoints found in this profile. Run training first to enable live predictions.")
        else:
            if mode == "Single Model":
                chosen_model = st.selectbox("Select Model:", available_models, index=0)
                if st.button("🚀 Run Prediction", type="primary"):
                    if input_img is not None:
                        cfg = MODEL_CONFIGS[chosen_model]
                        ckpt_file = ckpt_dir / f"{chosen_model.replace(' ', '_').replace('/', '-')}.pth"
                        model_obj = get_pytorch_model(chosen_model, str(ckpt_file), num_classes=3)

                        if model_obj:
                            t0 = time.perf_counter()
                            probs = run_single_inference(model_obj, input_img, size=cfg['size'])
                            dt = (time.perf_counter() - t0) * 1000

                            pred_idx = np.argmax(probs)
                            pred_class = DEFAULT_CLASSES[pred_idx]
                            conf = probs[pred_idx] * 100

                            st.success(f"**Prediction: {pred_class.upper()}** ({conf:.1f}% confidence)")
                            st.caption(f"Inference latency: {dt:.1f} ms on CPU")

                            # Bar chart
                            prob_df = pd.DataFrame({
                                'Class': [c.capitalize() for c in DEFAULT_CLASSES],
                                'Probability': probs
                            }).set_index('Class')
                            st.bar_chart(prob_df)
                    else:
                        st.warning("Please select or upload an image first.")

            else:  # Ensemble
                c_m1, c_m2 = st.columns(2)
                with c_m1:
                    m1 = st.selectbox("Model 1:", available_models, index=0)
                with c_m2:
                    m2_opts = [m for m in available_models if m != m1] or available_models
                    m2 = st.selectbox("Model 2:", m2_opts, index=0)

                if st.button("🚀 Run Ensemble Prediction", type="primary"):
                    if input_img is not None:
                        ckpt1 = ckpt_dir / f"{m1.replace(' ', '_').replace('/', '-')}.pth"
                        ckpt2 = ckpt_dir / f"{m2.replace(' ', '_').replace('/', '-')}.pth"

                        obj1 = get_pytorch_model(m1, str(ckpt1), 3)
                        obj2 = get_pytorch_model(m2, str(ckpt2), 3)

                        if obj1 and obj2:
                            t0 = time.perf_counter()
                            p1 = run_single_inference(obj1, input_img, MODEL_CONFIGS[m1]['size'])
                            p2 = run_single_inference(obj2, input_img, MODEL_CONFIGS[m2]['size'])
                            p_ens = (p1 + p2) / 2.0
                            dt = (time.perf_counter() - t0) * 1000

                            pred_idx = np.argmax(p_ens)
                            pred_class = DEFAULT_CLASSES[pred_idx]
                            conf = p_ens[pred_idx] * 100

                            pred1 = DEFAULT_CLASSES[np.argmax(p1)]
                            pred2 = DEFAULT_CLASSES[np.argmax(p2)]
                            agree = (pred1 == pred2)

                            st.success(f"**Ensemble Prediction: {pred_class.upper()}** ({conf:.1f}%)")
                            st.write(f"Model 1 ({m1}): `{pred1}` ({p1.max()*100:.1f}%) | Model 2 ({m2}): `{pred2}` ({p2.max()*100:.1f}%)")
                            st.caption(f"Agreement status: {'✅ Both models agreed' if agree else '⚠️ Models disagreed, resolved by soft-voting'} (Latency: {dt:.1f} ms)")

                            ens_df = pd.DataFrame({
                                'Class': [c.capitalize() for c in DEFAULT_CLASSES],
                                f'{m1}': p1,
                                f'{m2}': p2,
                                'Ensemble (Avg)': p_ens,
                            }).set_index('Class')
                            st.bar_chart(ens_df)
                    else:
                        st.warning("Please select or upload an image first.")


# ----------------------------------------------------
# TAB 4: GRAPHS & PLOTS GALLERY
# ----------------------------------------------------
with tab_plots:
    st.subheader("Benchmark Plots & Curve Gallery")
    st.caption("All visualization plots generated directly from benchmark executions.")

    plot_files = {
        "📊 Benchmark Summary Plots": results_dir / "summary_plots.png",
        "📉 Validation Loss & Accuracy Curves": results_dir / "validation_curves.png",
        "🎨 Data Augmentation Preview": results_dir / "augmentation_preview.png",
        "🤝 Pairwise Ensemble Heatmap & Latency": results_dir / "ensemble_summary_plots.png",
    }

    found_any = False
    for title, p_path in plot_files.items():
        if p_path.exists():
            found_any = True
            st.markdown(f"#### {title}")
            st.image(str(p_path), use_container_width=True)
            st.markdown("---")

    if not found_any:
        st.info("No plot images found in the selected folder. Re-run training scripts or pick an experiment folder with saved plots.")

