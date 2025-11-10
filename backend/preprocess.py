from __future__ import annotations

import os
# Keep TensorFlow informational output quieter and disable oneDNN optimizations that trigger verbose messages.
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import zipfile
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image
import random
import math
import json

try:
    import tensorflow as tf
except Exception:
    tf = None


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def extract_zip(zip_path: str, target_dir: str) -> None:
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Zip file not found: {zip_path}")
    ensure_dir(target_dir)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(target_dir)


def list_image_files(folder: str, exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png")) -> List[str]:
    files = []
    for root, _, filenames in os.walk(folder):
        for fn in filenames:
            if fn.lower().endswith(exts):
                files.append(os.path.join(root, fn))
    files.sort()
    return files


def infer_label_from_path(path: str, root_folder: str) -> str:
    rel = os.path.relpath(path, root_folder)
    parts = rel.split(os.sep)
    if len(parts) < 2:
        return "unknown"
    return parts[0]


def load_and_process_image(path: str, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize(size, Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def random_flip(img: np.ndarray) -> np.ndarray:
    if random.random() < 0.5:
        return np.fliplr(img).copy()
    return img


def random_rotate(img: np.ndarray, max_deg: float = 15.0) -> np.ndarray:
    angle = random.uniform(-max_deg, max_deg)
    pil = Image.fromarray((img * 255).astype(np.uint8))
    pil = pil.rotate(angle, resample=Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    return arr


def random_zoom(img: np.ndarray, min_zoom: float = 0.9, max_zoom: float = 1.05) -> np.ndarray:
    h, w, _ = img.shape
    zoom = random.uniform(min_zoom, max_zoom)
    new_h = int(h * zoom)
    new_w = int(w * zoom)
    pil = Image.fromarray((img * 255).astype(np.uint8))
    pil = pil.resize((new_w, new_h), Image.BILINEAR)
    # center-crop or pad back to original
    if zoom >= 1.0:
        # crop center
        left = (new_w - w) // 2
        top = (new_h - h) // 2
        pil = pil.crop((left, top, left + w, top + h))
    else:
        # pad
        new_img = Image.new("RGB", (w, h), (int(255 * 0.5),) * 3)
        left = (w - new_w) // 2
        top = (h - new_h) // 2
        new_img.paste(pil, (left, top))
        pil = new_img
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    return arr


def augment_image(img: np.ndarray) -> np.ndarray:
    """Apply a sequence of light augmentations to an image (numpy float32 in [0,1])."""
    img = random_flip(img)
    img = random_rotate(img, max_deg=12)
    img = random_zoom(img, min_zoom=0.92, max_zoom=1.08)
    return img


def build_label_map(classes: List[str]) -> Dict[str, int]:
    classes_sorted = sorted(set(classes))
    return {c: i for i, c in enumerate(classes_sorted)}


def one_hot(label_idx: int, num_classes: int) -> np.ndarray:
    vec = np.zeros((num_classes,), dtype=np.float32)
    vec[label_idx] = 1.0
    return vec


def train_val_split(paths: List[str], labels: List[str], val_fraction: float = 0.2, seed: int = 42) -> Tuple[List[str], List[str], List[str], List[str]]:
    assert 0.0 <= val_fraction < 1.0
    rng = random.Random(seed)
    combined = list(zip(paths, labels))
    rng.shuffle(combined)
    n_val = int(len(combined) * val_fraction)
    val = combined[:n_val]
    train = combined[n_val:]
    train_paths, train_labels = zip(*train) if train else ([], [])
    val_paths, val_labels = zip(*val) if val else ([], [])
    return list(train_paths), list(train_labels), list(val_paths), list(val_labels)


def prepare_dataset(raw_dir: str = "infosys_dataset", out_dir: str = "data/processed",
                    image_size: Tuple[int, int] = (224, 224), val_split: float = 0.2,
                    augment: bool = True, save_np: bool = True, seed: int = 42) -> Dict[str, str]:
    ensure_dir(out_dir)
    files = list_image_files(raw_dir)
    if len(files) == 0:
        raise ValueError(f"No image files found in {raw_dir}")
    labels = [infer_label_from_path(p, raw_dir) for p in files]
    label_map = build_label_map(labels)

    # map labels to indices
    label_idxs = [label_map[l] for l in labels]

    # split
    t_paths, t_labels, v_paths, v_labels = train_val_split(files, labels, val_fraction=val_split, seed=seed)

    def load_list(paths_list: List[str], labels_list: List[str], augment_it: bool = False):
        X = []
        Y = []
        for p, lab in zip(paths_list, labels_list):
            arr = load_and_process_image(p, size=image_size)
            if augment_it and augment:
                arr = augment_image(arr)
            X.append(arr)
            Y.append(label_map[lab])
        if len(X) == 0:
            return np.zeros((0, image_size[0], image_size[1], 3), dtype=np.float32), np.zeros((0,), dtype=np.int32)
        return np.stack(X).astype(np.float32), np.array(Y, dtype=np.int32)

    X_train, y_train = load_list(t_paths, t_labels, augment_it=False)
    X_val, y_val = load_list(v_paths, v_labels, augment_it=False)

    saved = {}
    if save_np:
        ensure_dir(out_dir)
        train_x_path = os.path.join(out_dir, "X_train.npy")
        train_y_path = os.path.join(out_dir, "y_train.npy")
        val_x_path = os.path.join(out_dir, "X_val.npy")
        val_y_path = os.path.join(out_dir, "y_val.npy")
        np.save(train_x_path, X_train)
        np.save(train_y_path, y_train)
        np.save(val_x_path, X_val)
        np.save(val_y_path, y_val)
        # save class mapping
        classes_path = os.path.join(out_dir, "classes.json")
        with open(classes_path, "w", encoding="utf8") as f:
            json.dump(label_map, f, indent=2)
        saved = {
            "X_train": train_x_path,
            "y_train": train_y_path,
            "X_val": val_x_path,
            "y_val": val_y_path,
            "classes": classes_path,
        }

    return saved


def visualize_augmentations(image_path: str, n: int = 6, out_dir: Optional[str] = None) -> List[Image.Image]:
    """Return a list of PIL Images showing augmentations of a single image. Optionally save to out_dir."""
    arr = load_and_process_image(image_path)
    imgs = []
    for i in range(n):
        aug = augment_image(arr)
        pil = Image.fromarray((aug * 255).astype(np.uint8))
        imgs.append(pil)
        if out_dir:
            ensure_dir(out_dir)
            pil.save(os.path.join(out_dir, f"aug_{i}.png"))
    return imgs


if __name__ == "__main__":
    # quick smoke demo when invoked directly
    print("Preprocess helper module. Run prepare_dataset(raw_dir, out_dir) to prepare dataset.")
