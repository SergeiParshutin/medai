# app_streamlit_gland_seg.py
#
# Streamlit lietotne glandu segmentācijai ar U-Net / FPN (segmentation_models_pytorch).
# Skolēni:
#   - izvēlas attēlu no datu kopas (test) vai augšupielādē savu,
#   - izvēlas modeli (U-Net / FPN),
#   - maina threshold slideri,
#   - pēc izvēles ieslēdz/izslēdz ground truth masku.
#
# Uz attēla tiek uzlikta:
#   - modeļa prognozētā maska (sarkanā krāsā),
#   - ground truth maska (zaļā krāsā, ja pieejama un ja atzīmēts checkbox).

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image

import streamlit as st
import torch
import cv2
import segmentation_models_pytorch as smp

from gland_dataset import create_binary_mask_from_json


# === Ceļi uz datiem un modeļiem ===

GLAND_ROOT = Path("data")

TRAIN_IMG_DIR = GLAND_ROOT / "training" / "img"
TEST_IMG_DIR = GLAND_ROOT / "test" / "img"

TRAIN_ANN_DIR = GLAND_ROOT / "training" / "ann"
TEST_ANN_DIR = GLAND_ROOT / "test" / "ann"

MODELS_DIR = Path("models")
UNET_WEIGHTS = MODELS_DIR / "gland_unet.pth"
FPN_WEIGHTS = MODELS_DIR / "gland_fpn.pth"
DEEPLAB_WEIGHTS = MODELS_DIR / "gland_deeplab.pth"

ENCODER_NAME = "resnet34"   # tas pats, ko izmantoji treniņam
# encoder_weights var būt None, jo mēs ielādējam savus state_dict
ENCODER_WEIGHTS = None

TARGET_SIZE: Tuple[int, int] = (256, 256)


def get_device() -> torch.device:
    """Izvēlas labāko pieejamo ierīci: CUDA -> MPS -> CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
# === Palīgfunkcijas ===

def overlay_mask(
    image_np: np.ndarray,
    mask_np: np.ndarray,
    color=(1.0, 0.0, 0.0),
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Uzliek bināru masku virs attēla.
    image_np: (H,W,3), [0..255] vai [0..1]
    mask_np: (H,W), 0/1 vai 0..1
    color: (R,G,B) [0..1]
    alpha: maskas caurspīdīgums
    """
    img = image_np.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0

    # ja maska nav 0/1, pamaskejam
    if mask_np.max() > 1.0:
        m = (mask_np > 0).astype(bool)
    else:
        m = (mask_np > 0.5).astype(bool)

    mask_bool = m.astype(bool)

    overlay = img.copy()
    col = np.zeros_like(overlay)
    col[..., 0] = color[0]
    col[..., 1] = color[1]
    col[..., 2] = color[2]

    overlay[mask_bool] = (
        alpha * col[mask_bool] + (1.0 - alpha) * overlay[mask_bool]
    )

    return (overlay * 255).astype(np.uint8)


def load_image_from_path(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def get_ann_path_for_image(image_name: str, split: str) -> Optional[Path]:
    """
    Atrod anotācijas ceļu priekš attēla:
      image_name: piem., "train_60.bmp"
      split: "training" vai "test"
    Atgriež Path vai None, ja anotācija nav atrasta.
    """
    if split == "training":
        ann_dir = TRAIN_ANN_DIR
    else:
        ann_dir = TEST_ANN_DIR

    ann_path = ann_dir / f"{image_name}.json"
    if ann_path.exists():
        return ann_path
    return None


@st.cache_resource
def load_model(model_type: str, encoder_type: str):
    """
    Ielādē U-Net, FPN vai DeepLab modeli ar saglabātajiem svariem.
    Modeli kešojam, lai nebūtu jāielādē atkārtoti.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "U-Net":
        model = smp.Unet(
            encoder_name=encoder_type,
            encoder_weights=ENCODER_WEIGHTS,
            in_channels=3,
            classes=1,
        )
        weights_path = UNET_WEIGHTS
    elif model_type == "FPN":
        model = smp.FPN(
            encoder_name=encoder_type,
            encoder_weights=ENCODER_WEIGHTS,
            in_channels=3,
            classes=1,
        )
        weights_path = FPN_WEIGHTS
    else:
        model = smp.DeepLabV3(
            encoder_name=encoder_type,
            encoder_weights=ENCODER_WEIGHTS,
            in_channels=3,
            classes=1,
        )
        weights_path = DEEPLAB_WEIGHTS

    if not weights_path.exists():
        raise FileNotFoundError(f"Modeļa svaru datne nav atrasta: {weights_path}")

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return model, device


def predict_mask(model, device, image: Image.Image) -> np.ndarray:
    """
    Izlaiž attēlu caur modeli un atgriež maskas varbūtības karti
    oriģinālajā izmērā: (H_orig, W_orig), vērtības [0..1].
    """
    # oriģinālais izmērs
    img_np = np.array(image, dtype=np.float32)  # (H,W,3)
    H, W, _ = img_np.shape

    # normalizācija
    img_norm = img_np / 255.0

    # resize uz treniņa izmēru (TARGET_SIZE)
    th, tw = TARGET_SIZE
    img_resized = cv2.resize(img_norm, (tw, th), interpolation=cv2.INTER_LINEAR)  # (th, tw, 3)

    # uz tensoru
    img_t = torch.tensor(img_resized, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_t)
        probs_small = torch.sigmoid(logits)[0, 0].cpu().numpy()  # (th, tw)

    # upscale atpakaļ uz oriģinālo izmēru
    probs = cv2.resize(probs_small, (W, H), interpolation=cv2.INTER_LINEAR)  # (H,W)

    return probs


# === Streamlit UI ===

def main():
    st.set_page_config(
        page_title="Glandu segmentācija ar U-Net / FPN / DeepLab",
        layout="wide",
    )

    st.title("Glandu segmentācijas demo ar U-Net / FPN / DeepLab")

    st.markdown(
        """
        Šī lietotne izmanto **piemācītus segmentācijas modeļus** (`U-Net`, `FPN` un `DeepLab`),
        lai segmentētu glandas histoloģijas attēlos.

        - Izvēlies modeli (U-Net / FPN / DeepLab),
        - izvēlies attēlu no datu kopas (test) vai augšupielādē savu,
        - maini **threshold slideri** un vēro, kā mainās maska,
        - pēc izvēles ieslēdz/izslēdz **ground truth** masku (ja tā ir pieejama).
        """
    )

    # --- Sidebar: modeļa izvēle, threshold, ground truth toggle ---
    st.sidebar.header("Iestatījumi")

    model_type = st.sidebar.radio(
        "Modelis",
        options=["U-Net", "FPN", "DeepLabV3"],
        help="Modeļi izmanto to pašu datu formātu – viegli salīdzināmi.",
    )

    encoder_type = st.sidebar.radio(
        "Enkoderis",
        options=["resnet34", "efficientnet-b0"],
        help="Ir jāizvēlas tas enkoderis, kurš bija izmantots modeļa apmācības laikā.",
    )
    
    threshold = st.sidebar.slider(
        "Threshold segmentācijas maskai",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Pikseļi ar varbūtību virs threshold tiks uzskatīti par '1' (glandu reģions).",
    )

    show_gt = st.sidebar.checkbox(
        "Rādīt ground truth masku (zaļš)",
        value=True,
    )

    # Modeļa ielāde
    try:
        model, device = load_model(model_type, encoder_type)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    # --- Attēla avota izvēle ---
    st.sidebar.header("Attēla izvēle")

    source_mode = st.sidebar.radio(
        "Attēla avots",
        options=["No datu kopas (test)", "Augšupielādēt attēlu"],
    )

    image: Optional[Image.Image] = None
    image_name: Optional[str] = None
    gt_mask_np: Optional[np.ndarray] = None
    has_gt = False
    split = "test"

    if source_mode == "No datu kopas (test)":
        test_images = sorted([p.name for p in TEST_IMG_DIR.glob("*.bmp")])
        if not test_images:
            st.sidebar.warning(f"Mapē {TEST_IMG_DIR} nav atrasti .bmp attēli.")
        else:
            image_name = st.sidebar.selectbox("Izvēlies attēlu no test kopas", test_images)
            if image_name is not None:
                img_path = TEST_IMG_DIR / image_name
                image = load_image_from_path(img_path)

                # GT maskas atrašana
                ann_path = get_ann_path_for_image(image_name, split="test")
                if ann_path is not None:
                    gt_mask_np = create_binary_mask_from_json(ann_path).astype(np.float32)
                    has_gt = True
    else:
        uploaded = st.sidebar.file_uploader(
            "Augšupielādē attēlu (BMP/PNG/JPG)",
            type=["bmp", "png", "jpg", "jpeg"],
        )
        if uploaded is not None:
            image_name = uploaded.name
            image = Image.open(uploaded).convert("RGB")
            # Augšupielādētam attēlam GT nav
            has_gt = False

    if image is None:
        st.info("Lūdzu, izvēlies attēlu no datu kopas vai augšupielādē savu attēlu.")
        return

    # --- Oriģinālais attēls ---
    c1,c2 = st.columns(2)
    st.subheader(f"Ielādētais attēls: {image_name}")
    with c1:
        st.image(image, caption="Oriģinālais attēls", width='stretch')

    img_np = np.array(image)
    H, W, _ = img_np.shape

    # Ja ir ground truth maska, pārbaudām izmērus
    if has_gt and gt_mask_np is not None:
        if gt_mask_np.shape != (H, W):
            st.warning(
                f"GT maskas izmērs {gt_mask_np.shape} nesakrīt ar attēla izmēru {(H, W)}. "
                "Varētu būt anotāciju problēma."
            )

    # --- Modeļa prognoze ---
    st.subheader(f"{model_type} prognoze")

    prob_map = predict_mask(model, device, image)  # (H,W), [0..1]
    pred_mask = (prob_map >= threshold).astype(np.uint8)

    # Overlay attēli
    pred_overlay = overlay_mask(img_np, pred_mask, color=(1.0, 0.0, 0.0), alpha=0.5)

    if has_gt and show_gt and gt_mask_np is not None:
        # GT overlay atsevišķi
        gt_overlay = overlay_mask(img_np, gt_mask_np, color=(0.0, 1.0, 0.0), alpha=0.5)

        # Kombinētais overlay: vispirms GT (zaļš), pēc tam pred (sarkans)
        combined = overlay_mask(
            overlay_mask(img_np, gt_mask_np, color=(0.0, 1.0, 0.0), alpha=0.5),
            pred_mask,
            color=(1.0, 0.0, 0.0),
            alpha=0.5,
        )
    else:
        gt_overlay = None
        combined = pred_overlay

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Prognozētā maska (binary)**")
        st.image(
            pred_mask * 255,
            caption=f"Threshold = {threshold:.2f}",
            width='stretch',
            clamp=True
        )

    with col2:
        if has_gt and show_gt and gt_overlay is not None:
            st.markdown("**Ground truth maska (zaļš pārklājums)**")
            st.image(gt_overlay, width='stretch')
        else:
            st.markdown("**Ground truth maska nav parādīta**")
            if not has_gt:
                st.info("Šim attēlam nav pieejama anotācija (GT maska).")

    with col3:
        st.markdown("**Kombinētais pārklājums (pred + GT)**")
        st.image(
            combined,
            caption="Sarkanā – prognoze, zaļā – ground truth (ja ieslēgts)",
            width='stretch'
        )

    # --- Vienkārša statistika ---
    st.markdown("---")
    st.subheader("Segmentācijas statistikas piemēri")

    pred_area = float(pred_mask.mean()) * 100.0
    st.write(f"**Prognozētais glandu laukums:** {pred_area:.2f}% no attēla")

    if has_gt and gt_mask_np is not None:
        gt_area = float(gt_mask_np.mean()) * 100.0
        st.write(f"**Ground truth glandu laukums:** {gt_area:.2f}% no attēla")

        # Vienkāršs IoU (vienam threshold)
        inter = np.logical_and(pred_mask == 1, gt_mask_np == 1).sum()
        union = np.logical_or(pred_mask == 1, gt_mask_np == 1).sum()
        iou = inter / union if union > 0 else 0.0
        st.write(f"**IoU (vienkāršots):** {iou:.3f}")

        st.markdown(
            """
            *Interpretācija:*  
            - Ja IoU ir tuvu 1.0, prognozētā maska labi sakrīt ar ground truth.  
            - Ja IoU ir zems, tad vai nu modelis neatrod glandu reģionus, vai arī
              prognozē daudz lieku pikseļu ārpus reālajām glandu robežām.
            """
        )
    else:
        st.info("IoU nevar aprēķināt, jo nav pieejama ground truth maska.")


if __name__ == "__main__":
    main()