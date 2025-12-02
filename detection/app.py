# app_streamlit_yolo_malaria.py
#
# Streamlit lietotne malārijas patogēnu detekcijai ar piemācītu YOLO modeli.
# Skolēni ielādē attēlu (no train/val kopas), lietotne:
#  - pārbauda, vai ir pieejama anotācija (.txt YOLO formātā),
#  - izlaiž attēlu caur modeli,
#  - uzzīmē ground truth bbox (pēc izvēles) un modeļa prognozes,
#  - ļauj mainīt confidence slieksni ar slaideri.

from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw

import streamlit as st
from ultralytics import YOLO


# === Ceļi uz datiem un modeli ===

YOLO_ROOT = Path("data/malaria_yolo")
IMAGES_TRAIN = YOLO_ROOT / "images" / "train"
IMAGES_VAL = YOLO_ROOT / "images" / "val"
LABELS_TRAIN = YOLO_ROOT / "labels" / "train"
LABELS_VAL = YOLO_ROOT / "labels" / "val"

# Ceļš uz labāko piemācīto modeli (pielāgo, ja nepieciešams)
BEST_MODEL_PATH = Path("runs/detect/yolo_malaria_v1/weights/best.pt")

# Klases (tai jāatbilst malaria.yaml un treniņam)
INFECTED_CLASSES = [
    "trophozoite",
    "ring",
    "schizont",
    "gametocyte",
]


# === Palīgfunkcijas ===

def load_yolo_labels(label_path: Path):
    """
    Nolasa YOLO formāta .txt anotāciju:
      katra rinda: class_id xc yc w h   (visi [0..1] intervālā).
    Atgriež sarakstu ar (class_id, xc, yc, w, h).
    """
    boxes = []
    if not label_path.exists():
        return boxes

    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split()
            cls_id = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])
            boxes.append((cls_id, xc, yc, w, h))

    return boxes


def yolo_to_xyxy(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int):
    """
    Konvertē koordinātas no YOLO formāta (normalizēti [0..1])
    uz pikseļiem: (x1, y1, x2, y2).
    """
    cx = xc * img_w
    cy = yc * img_h
    bw = w * img_w
    bh = h * img_h

    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2

    return x1, y1, x2, y2


def draw_boxes(
    image: Image.Image,
    boxes_xyxy: List[Tuple[float, float, float, float]],
    labels: List[str],
    color: str,
):
    """
    Uzzīmē taisnstūrus virs attēla.

    - image: PIL attēls
    - boxes_xyxy: saraksts ar (x1, y1, x2, y2)
    - labels: teksta anotācijas (tāds pats garums kā boxes_xyxy)
    - color: krāsa (piem., "green", "red")

    Atgriež jaunu PIL attēlu ar uzvilktām kastēm.
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)

    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        if labels is not None and i < len(labels):
            # Tekstu zīmējam nedaudz virs kastes
            text = labels[i]
            draw.text((x1, max(0, y1 - 12)), text, fill=color)

    return img


def find_label_file(stem: str) -> Path | None:
    """
    Meklē anotācijas datni (YOLO .txt) priekš dotā faila vārda (bez paplašinājuma)
    gan train, gan val mapēs.
    """
    candidate_train = LABELS_TRAIN / f"{stem}.txt"
    candidate_val = LABELS_VAL / f"{stem}.txt"

    if candidate_train.exists():
        return candidate_train
    if candidate_val.exists():
        return candidate_val
    return None


@st.cache_resource
def load_trained_model(model_path: Path):
    """
    Ielādē piemācīto YOLO modeli vienu reizi (cache_resource),
    lai katrs pieprasījums nelādētu to no jauna.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Modelis nav atrasts: {model_path}")
    model = YOLO(str(model_path))
    return model


# === Streamlit UI ===

def main():
    st.set_page_config(
        page_title="Malārijas patogēnu detekcija ar YOLO",
        layout="wide",
    )

    st.title("Malārijas patogēnu detekcijas demo ar YOLO")
    st.markdown(
        """
        Šī lietotne izmanto **piemācītu YOLO modeli**, lai detektētu malārijas patogēnus
        asinssmērējuma attēlos (mikroskopijas dati).

        - Ielādē attēlu no datu kopas (train/val),
        - Lietotne pārbauda, vai šim attēlam ir ground truth anotācija (.txt YOLO formātā),
        - Attēls tiek izlaists caur modeli,
        - Tiek uzzīmēti:
          - **modeļa prognozes** (sarkanā krāsā),
          - pēc izvēles – **ground truth anotācijas** (zaļā krāsā),
        - Ar **slaideri** var mainīt confidence slieksni modeļa prognožu filtrēšanai.
        """
    )

    # Sānu panelis – iestatījumi
    st.sidebar.header("Iestatījumi")

    conf_th = st.sidebar.slider(
        "Confidence slieksnis prognozēm",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Prognozes ar zemāku pārliecību tiks ignorētas.",
    )

    show_gt = st.sidebar.checkbox(
        "Rādīt ground truth anotācijas (zaļš)",
        value=True,
    )

    # Modeļa ielāde (vienu reizi)
    try:
        model = load_trained_model(BEST_MODEL_PATH)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    st.sidebar.header("Attēla izvēle")

    # Iespēja skolēnam augšupielādēt attēlu (faila nosaukumam jāatbilst train/val attēlam, lai atrastu anotāciju)
    upload_mode = st.sidebar.radio(
        "Attēla avots",
        options=["Augšupielādēt failu", "Izvēlēties no val kopas"],
        help="Vienkāršībai skolēni var izvēlēties attēlu no validācijas kopas vai augšupielādēt paši.",
    )

    image = None
    image_name = None

    if upload_mode == "Augšupielādēt failu":
        uploaded_file = st.sidebar.file_uploader(
            "Augšupielādē attēlu (PNG/JPG)",
            type=["png", "jpg", "jpeg"],
        )
        if uploaded_file is not None:
            image_name = uploaded_file.name
            image = Image.open(uploaded_file).convert("RGB")
    else:
        # Vienkāršs selectbox no validācijas attēliem
        val_images = sorted([p.name for p in IMAGES_VAL.glob("*.png")])
        if not val_images:
            st.sidebar.warning(f"Mapē {IMAGES_VAL} nav atrasti attēli.")
        else:
            image_name = st.sidebar.selectbox("Izvēlies attēlu no val kopas", val_images)
            if image_name is not None:
                image_path = IMAGES_VAL / image_name
                if image_path.exists():
                    image = Image.open(image_path).convert("RGB")

    if image is None:
        st.info("Lūdzu, augšupielādē attēlu vai izvēlies to no validācijas kopas.")
        return

    st.subheader(f"Ielādētais attēls: {image_name}")
    st.image(image, caption="Oriģinālais attēls", use_column_width=True)

    w, h = image.size

    # === Ground truth anotāciju ielāde (ja pieejamas) ===
    # Pamatojamies uz faila nosaukumu (bez paplašinājuma)
    stem = Path(image_name).stem
    label_file = find_label_file(stem)

    gt_boxes_xyxy: List[Tuple[float, float, float, float]] = []
    gt_labels: List[str] = []

    if label_file is not None:
        yolo_boxes = load_yolo_labels(label_file)
        for (cls_id, xc, yc, bw, bh) in yolo_boxes:
            x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, bw, bh, w, h)
            gt_boxes_xyxy.append((x1, y1, x2, y2))
            # Pārbaude – ja cls_id robežās
            if 0 <= cls_id < len(INFECTED_CLASSES):
                cls_name = INFECTED_CLASSES[cls_id]
            else:
                cls_name = f"class_{cls_id}"
            gt_labels.append(f"GT: {cls_name}")
    else:
        st.warning(
            f"Ground truth anotācija (.txt) netika atrasta ne train, ne val mapē priekš '{stem}'. "
            "GT robežas netiks zīmētas."
        )

    # === Modeļa prognozes ===
    st.subheader("Modeļa prognozes")

    # YOLO var tieši pieņemt PIL attēlu; norādām confidence slieksni
    results = model.predict(
        source=image,
        conf=conf_th,
        verbose=False,
    )
    pred = results[0]

    pred_boxes_xyxy: List[Tuple[float, float, float, float]] = []
    pred_labels: List[str] = []

    # Ultralytics YOLOv8 dod boxes ar xyxy pikseļos, conf un cls
    if pred.boxes is not None and len(pred.boxes) > 0:
        boxes_xyxy = pred.boxes.xyxy.cpu().numpy()
        conf_array = pred.boxes.conf.cpu().numpy()
        cls_ids = pred.boxes.cls.cpu().numpy().astype(int)

        # model.names satur klases (pēc treniņa to vajadzētu sakrist ar INFECTED_CLASSES)
        model_names = model.names

        for (bx1, by1, bx2, by2), c, cid in zip(boxes_xyxy, conf_array, cls_ids):
            pred_boxes_xyxy.append((bx1, by1, bx2, by2))
            name = model_names.get(cid, f"class_{cid}") if isinstance(model_names, dict) else model_names[cid]
            pred_labels.append(f"Pred: {name} {c:.2f}")
    else:
        st.info("Modelis neatrada nevienu objektu ar pašreizējo confidence slieksni.")

    # === Zīmējam kombinēto attēlu: preds (sarkans) + izvēles GT (zaļš) ===

    # Sākam ar oriģinālo attēlu
    combined = image.copy()

    # Vispirms uzzīmējam modeļa prognozes (lai GT varētu būt virsū, ja gribam)
    if pred_boxes_xyxy:
        combined = draw_boxes(
            combined,
            pred_boxes_xyxy,
            labels=pred_labels,
            color="red",
        )

    # Pēc izvēles – ground truth
    if show_gt and gt_boxes_xyxy:
        combined = draw_boxes(
            combined,
            gt_boxes_xyxy,
            labels=gt_labels,
            color="green",
        )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Prognozes + (pēc izvēles) ground truth")
        st.image(
            combined,
            caption="Sarkanā krāsā – modeļa prognozes, zaļā – ground truth (ja ieslēgts)",
            use_column_width=True,
        )

    with col2:
        st.subheader("Īsa kopsavilkuma statistika")

        st.write(f"Attēla izmērs: **{w} × {h}** pikseļi")

        st.write(f"Ground truth objektu skaits: **{len(gt_boxes_xyxy)}**")
        st.write(f"Modeļa prognožu skaits (pēc threshold): **{len(pred_boxes_xyxy)}**")

        if gt_boxes_xyxy:
            st.write("Ground truth klases (kopā):")
            st.write(", ".join(sorted(set(lbl.replace('GT: ', '') for lbl in gt_labels))))

        if pred_boxes_xyxy:
            st.write("Prognozētās klases (kopā):")
            st.write(", ".join(sorted(set(lbl.split()[1] for lbl in pred_labels))))

        st.markdown(
            """
            **Ierosinājumi diskusijai ar skolēniem:**
            - Vai modelis atrod visus ground truth objektus?
            - Vai ir kādi “false positives” (prognozes, kur objekta patiesībā nav)?
            - Kā confidence sliekšņa maiņa ietekmē rezultātu – vai vairāk/mazāk objekti, kas ir pareizi?
            """
        )


if __name__ == "__main__":
    main()