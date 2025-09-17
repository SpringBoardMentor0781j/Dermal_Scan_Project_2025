import os
import cv2
import numpy as np
import pandas as pd

RAW_DATASET = "dataset_raw"
MANUAL_DATASET = "dataset_manual"
CATEGORIES = ["clear face", "darkspots", "puffy eyes", "wrinkles"]
TARGET_SIZE = (224, 224)

# Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def is_black_and_white(image):
    """Reject grayscale or black & white images"""
    if len(image.shape) < 3 or image.shape[2] == 1:
        return True
    if np.array_equal(image[:, :, 0], image[:, :, 1]) and np.array_equal(image[:, :, 1], image[:, :, 2]):
        return True
    return False

def get_next_filename(dst_dir):
    """Generate sequential filename 0001.png, 0002.png, ..."""
    existing = [f for f in os.listdir(dst_dir) if f.endswith(".png")]
    if not existing:
        return "0001.png"
    nums = [int(f.split(".")[0]) for f in existing if f.split(".")[0].isdigit()]
    return f"{max(nums) + 1:04d}.png"

def preprocess_and_save(img, dst_dir):
    """Resize and save image to PNG with sequential naming"""
    img = cv2.resize(img, TARGET_SIZE)
    os.makedirs(dst_dir, exist_ok=True)
    filename = get_next_filename(dst_dir)
    save_path = os.path.join(dst_dir, filename)
    cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return os.path.basename(save_path)

def manual_review():
    counts = {cat: 0 for cat in CATEGORIES}
    limits = {cat: 350 for cat in CATEGORIES}
    records = []

    for category in CATEGORIES:
        print(f"\n=== Reviewing category: {category.upper()} ===")
        src_dir = os.path.join(RAW_DATASET, category)
        dst_dir = os.path.join(MANUAL_DATASET, category)
        os.makedirs(dst_dir, exist_ok=True)

        files = sorted(os.listdir(src_dir))
        for file in files:
            path = os.path.join(src_dir, file)
            img = cv2.imread(path)

            if img is None or is_black_and_white(img):
                continue

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

            # face requirement logic
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                roi = img[y:y+h, x:x+w]
                display_img = cv2.resize(roi, (400, 400))
            else:
                if category in ["clear face", "wrinkles"]:
                    continue
                display_img = cv2.resize(img, (400, 400))

            cv2.imshow("Review", display_img)
            print(f"[{counts[category]}/{limits[category]}] {category}: {file}")
            key = cv2.waitKey(0) & 0xFF

            save_to = None
            if key == ord('r'):
                continue
            elif key == ord('k'):
                save_to = category
            elif key == ord('c'):
                save_to = "clear face"
            elif key == ord('p'):
                save_to = "puffy eyes"
            elif key == ord('d'):
                save_to = "darkspots"
            elif key == ord('w'):
                save_to = "wrinkles"

            if save_to and counts[save_to] < limits[save_to]:
                dst_dir_sel = os.path.join(MANUAL_DATASET, save_to)
                filename = preprocess_and_save(display_img, dst_dir_sel)
                counts[save_to] += 1
                records.append((filename, save_to))

            if all(counts[c] >= limits[c] for c in CATEGORIES):
                cv2.destroyAllWindows()
                print("\n=== Completed manual review ===")
                return records

    cv2.destroyAllWindows()
    print("\n=== Manual review finished (dataset exhausted) ===")
    return records

def rename_and_generate_labels():
    print("\n=== Renaming files and generating CSVs ===")
    all_records = []
    onehot_records = []

    for category in CATEGORIES:
        cat_dir = os.path.join(MANUAL_DATASET, category)
        files = sorted([f for f in os.listdir(cat_dir) if f.endswith(".png")])
        for idx, file in enumerate(files, start=1):
            new_name = f"{idx:04d}.png"
            old_path = os.path.join(cat_dir, file)
            new_path = os.path.join(cat_dir, new_name)
            os.rename(old_path, new_path)

            all_records.append((new_name, category))

            onehot = [0]*len(CATEGORIES)
            onehot[CATEGORIES.index(category)] = 1
            onehot_records.append((new_name, *onehot))

    # Save labels.csv
    df = pd.DataFrame(all_records, columns=["filename", "class"])
    df.to_csv(os.path.join(MANUAL_DATASET, "labels.csv"), index=False)

    # Save labels_onehot.csv
    df_onehot = pd.DataFrame(onehot_records, columns=["filename"] + CATEGORIES)
    df_onehot.to_csv(os.path.join(MANUAL_DATASET, "labels_onehot.csv"), index=False)

    print("Saved labels.csv and labels_onehot.csv")

if __name__ == "__main__":
    manual_review()
    rename_and_generate_labels()
