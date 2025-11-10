from __future__ import annotations

from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

__all__ = [
	"draw_predictions",
	"to_pil",
	"image_grid",
	"plot_confusion_matrix",
	"plot_class_distribution",
]


def to_pil(img: Union[np.ndarray, Image.Image]) -> Image.Image:
	if isinstance(img, Image.Image):
		return img.convert("RGB")
	arr = np.asarray(img)
	if arr.ndim == 3 and arr.shape[2] == 3:
		# OpenCV uses BGR
		arr = arr[:, :, ::-1]
	return Image.fromarray(arr)


def _get_font(size: int = 16) -> ImageFont.FreeTypeFont:
	"""Return a PIL font. Falls back to default if freetype not available."""
	try:
		return ImageFont.truetype("arial.ttf", size)
	except Exception:
		return ImageFont.load_default()


def draw_predictions(
	image: Union[np.ndarray, Image.Image],
	predictions: List[Dict],
	box_color: Tuple[int, int, int] = (0, 255, 0),
	text_color: Tuple[int, int, int] = (0, 0, 0),
	draw_confidence: bool = True,
) -> Image.Image:
	pil = to_pil(image)
	draw = ImageDraw.Draw(pil)
	font = _get_font(14)

	for pred in predictions:
		r = pred.get("region")
		if isinstance(r, dict):
			x, y, w, h = r["x"], r["y"], r["width"], r["height"]
		else:
			x, y, w, h = r

		# Ensure coordinates are integers and inside image
		x, y, w, h = int(x), int(y), int(w), int(h)
		x2, y2 = x + w, y + h

		# Draw rectangle (PIL uses RGB)
		draw.rectangle([x, y, x2, y2], outline=box_color, width=3)

		# Prepare label
		label = pred.get("class_name", "")
		if draw_confidence:
			prob = pred.get("probability", 0.0)
			label = f"{label}: {prob:.0%}"

		# Text background
		text_size = draw.textsize(label, font=font)
		text_bg = [x, max(y - text_size[1] - 4, 0), x + text_size[0] + 4, y]
		draw.rectangle(text_bg, fill=(255, 255, 255))
		draw.text((x + 2, max(y - text_size[1] - 2, 0)), label, fill=text_color, font=font)

	return pil


def image_grid(
	images: List[Union[np.ndarray, Image.Image]],
	titles: Optional[List[str]] = None,
	cols: int = 3,
	thumb_size: Tuple[int, int] = (224, 224),
) -> Image.Image:
	imgs = [to_pil(im).resize(thumb_size, Image.LANCZOS) for im in images]
	n = len(imgs)
	if n == 0:
		return Image.new("RGB", (thumb_size[0], thumb_size[1]), (255, 255, 255))

	rows = (n + cols - 1) // cols
	grid_w = cols * thumb_size[0]
	grid_h = rows * thumb_size[1] + (20 * rows if titles else 0)
	grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
	draw = ImageDraw.Draw(grid)
	font = _get_font(12)

	for idx, img in enumerate(imgs):
		row = idx // cols
		col = idx % cols
		x = col * thumb_size[0]
		y = row * thumb_size[1] + (20 * row if titles else 0)
		grid.paste(img, (x, y))
		if titles and idx < len(titles):
			txt = titles[idx]
			w, h = draw.textsize(txt, font=font)
			draw.rectangle([x, y + thumb_size[1], x + w + 6, y + thumb_size[1] + h + 4], fill=(255, 255, 255))
			draw.text((x + 3, y + thumb_size[1] + 2), txt, fill=(0, 0, 0), font=font)

	return grid


def plot_confusion_matrix(
	cm: np.ndarray,
	class_names: List[str],
	normalize: bool = False,
	figsize: Tuple[int, int] = (8, 6),
	cmap: str = "Blues",
) -> plt.Figure:
	if normalize:
		cm_sum = cm.sum(axis=1, keepdims=True)
		cm = cm.astype("float") / np.where(cm_sum == 0, 1, cm_sum)

	fig, ax = plt.subplots(figsize=figsize)
	sns.heatmap(cm, annot=True, fmt=".2f" if normalize else "d", cmap=cmap, ax=ax,
				xticklabels=class_names, yticklabels=class_names)
	ax.set_xlabel("Predicted")
	ax.set_ylabel("True")
	ax.set_title("Confusion Matrix")
	plt.tight_layout()
	return fig


def plot_class_distribution(counts: Dict[str, int], figsize: Tuple[int, int] = (8, 4)) -> plt.Figure:
	labels = list(counts.keys())
	values = [counts[k] for k in labels]
	fig, ax = plt.subplots(figsize=figsize)
	sns.barplot(x=labels, y=values, ax=ax)
	ax.set_ylabel("Count")
	ax.set_xlabel("Class")
	ax.set_title("Class Distribution")
	ax.set_xticklabels(labels, rotation=45)
	plt.tight_layout()
	return fig

