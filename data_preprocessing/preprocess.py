import os
import cv2
import hashlib
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img

RAW_DIR = "dataset_raw"
CLEAN_DIR = "dataset_clean/dataset"
AUG_DIR = "dataset_clean/dataset_augmented"
LABELS_FILE = "dataset_clean/labels.csv"
LABELS_ONEHOT_FILE = "dataset_clean/labels_onehot.csv"
IMG_SIZE = (224, 224)

# Haar cascade for face detection
FACE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ==============================
# Utils
# ==============================

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def file_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def is_black_white(img):
    if len(img.shape) < 3 or img.shape[2] == 1:
        return True
    b, g, r = cv2.split(img)
    return (np.array_equal(b, g) and np.array_equal(b, r))

def detect_and_crop_face(img):
    """Detect face and crop largest one if available."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        return None
    # take largest face
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    face_crop = img[y:y+h, x:x+w]
    return face_crop

# ==============================
# Steps
# ==============================

def convert_to_png():
    print("=== STEP 1: Converting to PNG with face detection ===")
    for cls in os.listdir(RAW_DIR):
        src_dir = os.path.join(RAW_DIR, cls)
        dst_dir = os.path.join(CLEAN_DIR, cls)
        ensure_dir(dst_dir)
        for f in os.listdir(src_dir):
            path = os.path.join(src_dir, f)
            img = cv2.imread(path)
            if img is None:
                continue

            if is_black_white(img):
                continue  # skip BW

            face_crop = detect_and_crop_face(img)

            if cls in ["clear face", "wrinkles"]:
                if face_crop is None:
                    print(f"Rejected (no face) {f} in {cls}")
                    continue
                img_out = face_crop
            elif cls in ["puffy eyes", "darkspots"]:
                if face_crop is not None:
                    img_out = face_crop
                else:
                    # Manual review if no face
                    cv2.imshow(f"Review {cls} (k=keep, r=reject, q=quit)", img)
                    key = cv2.waitKey(0) & 0xFF
                    if key == ord("k"):
                        img_out = img
                    elif key == ord("r"):
                        continue
                    elif key == ord("q"):
                        cv2.destroyAllWindows()
                        return
            else:
                img_out = img

            # Resize and save
            img_out = cv2.resize(img_out, IMG_SIZE)
            out_name = os.path.splitext(f)[0] + ".png"
            out_path = os.path.join(dst_dir, out_name)
            cv2.imwrite(out_path, img_out)
    cv2.destroyAllWindows()
    print("Conversion + face crop complete")

def remove_duplicates():
    print("=== STEP 2: Removing duplicates ===")
    hashes = {}
    removed = 0
    for root, _, files in os.walk(CLEAN_DIR):
        for f in files:
            path = os.path.join(root, f)
            h = file_hash(path)
            if h in hashes:
                os.remove(path)
                removed += 1
            else:
                hashes[h] = path
    print(f"Removed {removed} duplicates")

def rename_files():
    print("=== STEP 3: Renaming files sequentially ===")
    for cls in os.listdir(CLEAN_DIR):
        cls_dir = os.path.join(CLEAN_DIR, cls)
        if not os.path.isdir(cls_dir):
            continue
        files = sorted(os.listdir(cls_dir))
        for idx, f in enumerate(files, 1):
            old_path = os.path.join(cls_dir, f)
            new_name = f"{cls.replace(' ', '_')}_{idx:03d}.png"
            new_path = os.path.join(cls_dir, new_name)
            os.rename(old_path, new_path)
    print("Renaming complete")

def augment_images():
    print("=== STEP 4: Augmenting dataset ===")
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    for cls in os.listdir(CLEAN_DIR):
        src_dir = os.path.join(CLEAN_DIR, cls)
        dst_dir = os.path.join(AUG_DIR, cls)
        ensure_dir(dst_dir)

        files = os.listdir(src_dir)
        num_aug = len(files) * 3
        count = 0
        i = 0

        while count < num_aug:
            img_path = os.path.join(src_dir, files[i % len(files)])
            img = load_img(img_path, target_size=IMG_SIZE)
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)

            aug_iter = datagen.flow(x, batch_size=1)
            for _ in range(1):
                batch = next(aug_iter)
                aug_img = array_to_img(batch[0])
                out_path = os.path.join(dst_dir, f"aug_{count+1:03d}.png")
                aug_img.save(out_path)
                count += 1
                if count >= num_aug:
                    break
            i += 1

        print(f"Augmented {cls}: {num_aug} new images")

def generate_labels():
    print("=== STEP 5: Generating labels.csv and labels_onehot.csv ===")
    classes = sorted(os.listdir(CLEAN_DIR))
    rows = []
    for cls in classes:
        cls_dir = os.path.join(CLEAN_DIR, cls)
        for f in os.listdir(cls_dir):
            rows.append([f, cls])

    df = pd.DataFrame(rows, columns=["filename", "class"])
    df.to_csv(LABELS_FILE, index=False)

    onehot = pd.get_dummies(df["class"])
    df_onehot = pd.concat([df["filename"], onehot], axis=1)
    df_onehot.to_csv(LABELS_ONEHOT_FILE, index=False)
    print("Labels generated")

# ==============================
# Main
# ==============================
if __name__ == "__main__":
    convert_to_png()
    remove_duplicates()
    rename_files()
    augment_images()
    generate_labels()
    print("=== Preprocessing Done ===")
