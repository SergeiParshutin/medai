# gland_dataset.py
"""
Glandu segmentācijas datu kopas modulis.

Pieņem datu struktūru:

data/
  gland_seg/
    training/
      img/*.bmp
      ann/*.bmp.json
    test/
      img/*.bmp
      ann/*.bmp.json

Katram attēlam "12345.bmp" anotācijas datne ir:
  "ann/12345.bmp.json"

JSON formāts (Supervisely tipa):
{
  "size": {"width": W, "height": H},
  "objects": [
    {
      "classTitle": "...",
      "points": {
        "exterior": [[x1, y1], [x2, y2], ..., [xn, yn]]
      }
    },
    ...
  ]
}

No poligoniem tiek izveidota bināra maska:
  1 = glands, 0 = fons.
"""

from pathlib import Path
from typing import List, Tuple, Optional

import json

import numpy as np
from PIL import Image
import cv2

import torch
from torch.utils.data import Dataset

def create_binary_mask_from_json(json_path: Path) -> np.ndarray:
        """
        No anotācijas JSON (Supervisely tipa poligoni) izveido bināru masku.
        Rezultāts: maska ar izmēru (H, W), kur 1 = glands, 0 = fons.

        Parametri:
        json_path: ceļš uz *.bmp.json anotācijas datni.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        h = data["size"]["height"]
        w = data["size"]["width"]

        # 0 = fons, 1 = glands
        mask = np.zeros((h, w), dtype=np.uint8)

        for obj in data.get("objects", []):
            # Ja grib filtru pēc konkrētām klasēm:
            # if obj["classTitle"] not in ["gland", "benign", "malignant"]:
            #     continue

            points = obj["points"]["exterior"]  # saraksts ar [x, y] punktiem
            if len(points) < 3:
                # poligons ar < 3 punktiem nav derīgs
                continue

            # OpenCV fillPoly sagaida int32 masīvu formā (N, 1, 2)
            pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 1)

        return mask

class GlandSegmentationDataset(Dataset):
    """
    PyTorch Dataset glandu segmentācijai.

    Pieņem datu struktūru:

      root/
        training/
          img/*.bmp
          ann/*.bmp.json
        test/
          img/*.bmp
          ann/*.bmp.json

    Parametri:
      root_dir: saknes mape (piem., Path("data/gland_seg"))
      split: "training" vai "test"
      transform: papildu transformācijas (piem., Albumentations), kas saņem
                 image (H,W,3) un mask (H,W) un atgriež tos pašos laukus.
    """

    def __init__(self, root_dir: Path, split: str = "training", transform: Optional = None):
        assert split in ["training", "test"], "split jābūt 'training' vai 'test'"
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform

        img_dir = self.root_dir / split / "img"
        ann_dir = self.root_dir / split / "ann"

        self.samples: List[Tuple[Path, Path]] = []

        # Meklējam visus BMP attēlus un tiem atbilstošās *.bmp.json anotācijas
        for img_path in sorted(img_dir.glob("*.bmp")):
            # "12345.bmp" -> "12345.bmp.json"
            ann_path = ann_dir / f"{img_path.name}.json"
            if not ann_path.exists():
                # ja nav anotācijas, izlaižam (var arī izdrukāt brīdinājumu)
                # print(f"[BRĪDINĀJUMS] Trūkst anotācijas priekš {img_path.name}")
                continue
            self.samples.append((img_path, ann_path))

        if not self.samples:
            raise RuntimeError(f"Neatradu nevienu attēla+anotācijas pāri mapē {img_dir}")

        print(f"GlandSegmentationDataset ({split}): {len(self.samples)} paraugi.")

    def __len__(self) -> int:
        return len(self.samples)
    
    def _create_binary_mask_from_json(self, json_path: Path) -> np.ndarray:
        """
        No anotācijas JSON (Supervisely tipa poligoni) izveido bināru masku.
        Rezultāts: maska ar izmēru (H, W), kur 1 = glands, 0 = fons.

        Parametri:
        json_path: ceļš uz *.bmp.json anotācijas datni.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        h = data["size"]["height"]
        w = data["size"]["width"]

        # 0 = fons, 1 = glands
        mask = np.zeros((h, w), dtype=np.uint8)

        for obj in data.get("objects", []):
            # Ja grib filtru pēc konkrētām klasēm:
            # if obj["classTitle"] not in ["gland", "benign", "malignant"]:
            #     continue

            points = obj["points"]["exterior"]  # saraksts ar [x, y] punktiem
            if len(points) < 3:
                # poligons ar < 3 punktiem nav derīgs
                continue

            # OpenCV fillPoly sagaida int32 masīvu formā (N, 1, 2)
            pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 1)

        return mask

    def __getitem__(self, idx: int):
        img_path, ann_path = self.samples[idx]

        # 1) Ielādējam attēlu (BMP -> RGB)
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img, dtype=np.float32) / 255.0  # [0,1], (H, W, 3)

        # 2) Iegūstam masku no anotācijas
        mask_np = self._create_binary_mask_from_json(ann_path).astype(np.float32)  # (H, W)

        # 3) Ja definēta transformācija (augmentācija), pielietojam
        if self.transform is not None:
            # Piemēram, Albumentations transformācijas sagaida argumentus "image" un "mask"
            transformed = self.transform(image=img_np, mask=mask_np)
            img_np = transformed["image"]
            mask_np = transformed["mask"]

        # 4) Konvertējam uz PyTorch tenzoriem formā (C, H, W)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # (3, H, W)
        mask_tensor = torch.from_numpy(mask_np).unsqueeze(0)    # (1, H, W)

        # Atgriežam arī faila nosaukumu (noderīgs debugam / vizualizācijai)
        return img_tensor, mask_tensor, img_path.name
    
    