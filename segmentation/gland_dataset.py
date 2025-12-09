# gland_dataset.py

from pathlib import Path
from typing import List, Tuple, Optional

import json
import base64
import zlib
from io import BytesIO

import numpy as np
from PIL import Image
import cv2

import torch
from torch.utils.data import Dataset


def create_binary_mask_from_json(json_path: Path) -> np.ndarray:
    """
    No Supervisely bitmap anotācijām izveido bināru masku.
    Rezultāts: maska (H, W), kur 1 = glands, 0 = fons.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    h = data["size"]["height"]
    w = data["size"]["width"]

    mask = np.zeros((h, w), dtype=np.uint8)

    for obj in data.get("objects", []):
        if obj.get("geometryType") != "bitmap":
            continue

        bm = obj.get("bitmap")
        if not bm:
            continue

        data_b64 = bm.get("data")
        origin = bm.get("origin", [0, 0])
        if not data_b64:
            continue

        x0, y0 = origin

        # base64 -> zlib -> PNG
        compressed = base64.b64decode(data_b64)
        png_bytes = zlib.decompress(compressed)

        img = Image.open(BytesIO(png_bytes))
        arr = np.array(img)

        if arr.ndim == 3:
            arr = arr[..., 0]

        bmp_mask = (arr > 0).astype(np.uint8)
        h_mask, w_mask = bmp_mask.shape

        x1 = min(w, x0 + w_mask)
        y1 = min(h, y0 + h_mask)
        mw = x1 - x0
        mh = y1 - y0
        if mw <= 0 or mh <= 0:
            continue

        mask[y0:y1, x0:x1] = np.maximum(
            mask[y0:y1, x0:x1],
            bmp_mask[0:mh, 0:mw],
        )

    return mask


class GlandSegmentationDataset(Dataset):
    """
    PyTorch Dataset glandu segmentācijai.

    root/
      training/
        img/*.bmp
        ann/*.bmp.json
      test/
        img/*.bmp
        ann/*.bmp.json

    target_size: (H, W) – uz kādu izmēru pārizmērot visus attēlus un maskas.
    """

    def __init__(
        self,
        root_dir: Path,
        split: str = "training",
        transform: Optional[object] = None,
        target_size: Optional[Tuple[int, int]] = (256, 256),
    ):
        assert split in ["training", "test"], "split jābūt 'training' vai 'test'"
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        self.target_size = target_size

        img_dir = self.root_dir / split / "img"
        ann_dir = self.root_dir / split / "ann"

        self.samples: List[Tuple[Path, Path]] = []

        for img_path in sorted(img_dir.glob("*.bmp")):
            ann_path = ann_dir / f"{img_path.name}.json"
            if not ann_path.exists():
                continue
            self.samples.append((img_path, ann_path))

        if not self.samples:
            raise RuntimeError(f"Neatradu nevienu attēla+anotācijas pāri mapē {img_dir}")

        print(f"GlandSegmentationDataset ({split}): {len(self.samples)} paraugi.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, ann_path = self.samples[idx]

        # 1) attēls
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img, dtype=np.float32) / 255.0  # (H,W,3)

        # 2) maska
        mask_np = create_binary_mask_from_json(ann_path).astype(np.float32)  # (H,W)

        # 3) pārizmērošana uz fiksētu izmēru (H_target, W_target)
        if self.target_size is not None:
            th, tw = self.target_size
            # cv2.resize izmanto (W, H)
            img_np = cv2.resize(img_np, (tw, th), interpolation=cv2.INTER_LINEAR)
            mask_np = cv2.resize(mask_np, (tw, th), interpolation=cv2.INTER_NEAREST)

        # 4) transformācijas (ja lieto Albumentations u.c.)
        if self.transform is not None:
            transformed = self.transform(image=img_np, mask=mask_np)
            img_np = transformed["image"]
            mask_np = transformed["mask"]

        # 5) uz tenzoriem – izmantojam torch.tensor, lai būtu normāls storage
        img_tensor = torch.tensor(img_np, dtype=torch.float32).permute(2, 0, 1).contiguous()
        mask_tensor = torch.tensor(mask_np, dtype=torch.float32).unsqueeze(0).contiguous()

        return img_tensor, mask_tensor, img_path.name