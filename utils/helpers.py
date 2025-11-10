from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Union, Optional

import numpy as np
from PIL import Image

__all__ = [
	"pil_to_cv2",
	"cv2_to_pil",
	"np_to_pil",
	"pil_to_np",
	"load_image",
	"save_image",
	"ensure_dir",
	"clean_dir",
	"is_image_file",
	"human_readable_time",
	"text_progress_bar",
	"resize_with_aspect",
]


def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
	if pil_img.mode == "RGBA":
		pil_img = pil_img.convert("RGB")
	arr = np.array(pil_img)
	# PIL -> RGB ; OpenCV expects BGR
	if arr.ndim == 3:
		return arr[:, :, ::-1].copy()
	return arr


def cv2_to_pil(cv_img: np.ndarray) -> Image.Image:
	if cv_img.ndim == 3:
		# BGR -> RGB
		cv_rgb = cv_img[:, :, ::-1]
		return Image.fromarray(cv_rgb)
	return Image.fromarray(cv_img)


def np_to_pil(arr: np.ndarray) -> Image.Image:
	"""Convert a numpy array (H,W,C) in RGB order to PIL Image.

	Assumes array dtype uint8 or float in 0..1.
	"""
	if arr.dtype != np.uint8:
		# scale floats to 0..255
		arr = np.clip(arr, 0.0, 1.0)
		arr = (arr * 255).astype(np.uint8)
	return Image.fromarray(arr)


def pil_to_np(pil_img: Image.Image) -> np.ndarray:
	"""Convert PIL Image to numpy array in RGB order.

	Returns:
		numpy array (H, W, C) with dtype uint8
	"""
	if pil_img.mode == "RGBA":
		pil_img = pil_img.convert("RGB")
	arr = np.array(pil_img)
	return arr


def load_image(path: Union[str, Path], as_cv2: bool = False) -> Optional[Union[Image.Image, np.ndarray]]:
	path = Path(path)
	if not path.exists():
		return None
	try:
		pil_img = Image.open(path).convert("RGB")
	except Exception:
		return None
	if as_cv2:
		return pil_to_cv2(pil_img)
	return pil_img


def save_image(image: Union[Image.Image, np.ndarray], path: Union[str, Path], quality: int = 90) -> None:
	path = Path(path)
	ensure_dir(path.parent)

	if isinstance(image, np.ndarray):
		# If array has 3 channels and appears BGR (values > 0..255), convert to RGB
		try:
			from cv2 import imwrite  # local import
			# Convert BGR->RGB for PIL saving if necessary
			if image.ndim == 3 and image.shape[2] == 3:
				img_to_save = image[:, :, ::-1]
			else:
				img_to_save = image
			pil_img = np_to_pil(img_to_save)
		except Exception:
			pil_img = np_to_pil(image)
	else:
		pil_img = image

	save_kwargs = {}
	suffix = path.suffix.lower()
	if suffix in {".jpg", ".jpeg"}:
		save_kwargs["quality"] = quality
		save_kwargs["optimize"] = True

	pil_img.save(path, **save_kwargs)


def ensure_dir(path: Union[str, Path]) -> Path:
	"""Ensure a directory exists and return the Path."""
	p = Path(path)
	p.mkdir(parents=True, exist_ok=True)
	return p


def clean_dir(path: Union[str, Path], pattern: Optional[str] = None) -> int:
	"""Remove files in a directory. If pattern provided, remove matching glob pattern.

	Returns number of files removed.
	"""
	path = Path(path)
	if not path.exists():
		return 0
	removed = 0
	if pattern:
		for f in path.glob(pattern):
			try:
				f.unlink()
				removed += 1
			except Exception:
				continue
	else:
		for f in path.iterdir():
			if f.is_file():
				try:
					f.unlink()
					removed += 1
				except Exception:
					continue
	return removed


def is_image_file(path: Union[str, Path]) -> bool:
	"""Quick check if path looks like an image file by suffix."""
	path = Path(path)
	return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif"}


def human_readable_time(seconds: float) -> str:
	seconds = int(round(seconds))
	if seconds < 60:
		return f"{seconds}s"
	minutes, sec = divmod(seconds, 60)
	if minutes < 60:
		return f"{minutes}m {sec}s"
	hours, minutes = divmod(minutes, 60)
	return f"{hours}h {minutes}m"


def text_progress_bar(current: int, total: int, length: int = 30) -> str:
	if total <= 0:
		return "[{}] 0%".format(" " * length)
	frac = max(0.0, min(1.0, float(current) / float(total)))
	filled = int(round(length * frac))
	bar = "#" * filled + "-" * (length - filled)
	percent = int(frac * 100)
	return f"[{bar}] {percent}% ({current}/{total})"


def resize_with_aspect(image: Union[Image.Image, np.ndarray], target_size: Tuple[int, int], fill_color=(0, 0, 0)) -> Image.Image:
	if isinstance(image, np.ndarray):
		image = np_to_pil(image)

	src_w, src_h = image.size
	tgt_w, tgt_h = target_size

	# Compute scale and new size
	scale = min(tgt_w / src_w, tgt_h / src_h)
	new_w = int(src_w * scale)
	new_h = int(src_h * scale)

	image_resized = image.resize((new_w, new_h), Image.LANCZOS)

	# Create background and paste centered
	background = Image.new("RGB", (tgt_w, tgt_h), fill_color)
	offset = ((tgt_w - new_w) // 2, (tgt_h - new_h) // 2)
	background.paste(image_resized, offset)
	return background

