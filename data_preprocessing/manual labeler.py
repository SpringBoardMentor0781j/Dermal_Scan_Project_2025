import os
import cv2
import pandas as pd

# ---------------------------
# CONFIGURATION
# ---------------------------

# Path to main dataset folder containing class subfolders
DATASET_PATH = "dataset_final"

# Path to save YOLO-style label txt files
LABELS_PATH = os.path.join(DATASET_PATH, "labels_coords")

# Path for consolidated CSV of all coordinates
CSV_COORDS_PATH = os.path.join(DATASET_PATH, "labels_coords.csv")

# Ensure the label folder exists
os.makedirs(LABELS_PATH, exist_ok=True)

# Define the exact class names
CLASSES = ["clear_face", "darkspots", "puffy_eyes", "wrinkles"]

# ---------------------------
# GLOBAL VARIABLES
# ---------------------------

# For drawing bounding boxes with mouse
drawing = False
ix, iy = -1, -1  # initial mouse coordinates
boxes = []  # list to store boxes for current image
current_img = None  # current image being annotated
current_class_id = None  # current selected class index
current_class_name = None  # current selected class name
current_img_path = None  # full path of current image

# List to store all labels for final CSV
all_labels = []

# ---------------------------
# MOUSE CALLBACK FUNCTION
# ---------------------------
def mouse_callback(event, x, y, flags, param):
    """Draw a rectangle on the image when mouse is dragged."""
    global ix, iy, drawing, boxes, current_img

    if event == cv2.EVENT_LBUTTONDOWN:  # mouse pressed
        drawing = True
        ix, iy = x, y  # save starting coordinates

    elif event == cv2.EVENT_LBUTTONUP:  # mouse released
        drawing = False
        # Draw rectangle on image for visual feedback
        cv2.rectangle(current_img, (ix, iy), (x, y), (0, 255, 0), 2)

        # Normalize coordinates to YOLO format (0-1)
        h, w = current_img.shape[:2]
        x_center = ((ix + x) / 2) / w
        y_center = ((iy + y) / 2) / h
        width = abs(x - ix) / w
        height = abs(y - iy) / h

        # Save box with class index and normalized coords
        boxes.append((current_class_id, x_center, y_center, width, height))

# ---------------------------
# ANNOTATION FUNCTION
# ---------------------------
def annotate_image(img_path):
    """Display an image, draw boxes, relabel on key press, and save immediately."""
    global current_img, boxes, current_class_id, current_class_name, current_img_path, all_labels

    # Load the image from disk
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not open {img_path}")
        return

    current_img = img.copy()  # working copy
    boxes = []  # reset boxes for this image
    current_img_path = img_path

    # Default class from folder name
    current_class_name = os.path.basename(os.path.dirname(img_path))
    if current_class_name not in CLASSES:
        print(f"Skipping unknown folder: {current_class_name}")
        return
    current_class_id = CLASSES.index(current_class_name)

    # Draw existing boxes if label file exists
    txt_name = f"{current_class_name}_{os.path.splitext(os.path.basename(img_path))[0]}.txt"
    label_path = os.path.join(LABELS_PATH, txt_name)
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    b_cls, x, y, w, h = parts
                    b_cls = int(b_cls)
                    x, y, w, h = map(float, [x, y, w, h])
                    boxes.append((b_cls, x, y, w, h))
                    # Draw existing boxes
                    h_img, w_img = img.shape[:2]
                   
                    cv2.rectangle(current_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Create window and bind mouse
    window_name = f"Annotator - {current_class_name}"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    print(f"Annotating: {img_path} | Default Class: {current_class_name}")

    while True:
        cv2.imshow(window_name, current_img)
        key = cv2.waitKey(1) & 0xFF

        # ---------------------------
        # Immediate relabel keys
        # ---------------------------
        if key == ord("c"):  # Clear face
            current_class_name = "clear_face"
            current_class_id = CLASSES.index(current_class_name)
        elif key == ord("d"):  # Darkspots
            current_class_name = "darkspots"
            current_class_id = CLASSES.index(current_class_name)
        elif key == ord("w"):  # Wrinkles
            current_class_name = "wrinkles"
            current_class_id = CLASSES.index(current_class_name)
        elif key == ord("p"):  # Puffy eyes
            current_class_name = "puffy_eyes"
            current_class_id = CLASSES.index(current_class_name)

        # ---------------------------
        # Save immediately after class key pressed
        # ---------------------------
        if key in [ord("c"), ord("d"), ord("w"), ord("p")]:
            # Build txt filename: {classname}_{filename}.txt
            txt_name = f"{current_class_name}_{os.path.splitext(os.path.basename(img_path))[0]}.txt"
            label_path = os.path.join(LABELS_PATH, txt_name)
            # Save label file
            with open(label_path, "w") as f:
                for b in boxes:
                    f.write(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n")
                    all_labels.append([os.path.basename(img_path), CLASSES[b[0]], b[1], b[2], b[3], b[4]])
            # Save annotated image
            annotated_img_path = os.path.join(LABELS_PATH, f"annotated_{os.path.basename(img_path)}")
            cv2.imwrite(annotated_img_path, current_img)
            print(f"Saved {label_path} and annotated image {annotated_img_path}")
            cv2.destroyAllWindows()
            break  # Move to next image immediately

        # ---------------------------
        # Skip image
        # ---------------------------
        elif key == 27:  # Esc
            print(f"Skipped {img_path}")
            cv2.destroyAllWindows()
            break

        # ---------------------------
        # Remove image
        # ---------------------------
        elif key == ord("r"):
            try:
                os.remove(current_img_path)
                print(f"Removed {current_img_path}")
            except Exception as e:
                print(f"Failed to remove {current_img_path}: {e}")
            cv2.destroyAllWindows()
            break

# ---------------------------
# MAIN LOOP: Traverse all class folders
# ---------------------------
for cls in CLASSES:
    folder = os.path.join(DATASET_PATH, cls)
    if not os.path.exists(folder):
        continue
    for file in os.listdir(folder):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(folder, file)
            annotate_image(img_path)

# ---------------------------
# SAVE FINAL CONSOLIDATED CSV
# ---------------------------
if all_labels:
    df_coords = pd.DataFrame(all_labels, columns=["filename", "class", "x_center", "y_center", "width", "height"])
    df_coords.to_csv(CSV_COORDS_PATH, index=False)
    print(f"✅ Overwritten {CSV_COORDS_PATH} with all annotation data.")
else:
    print("No annotations collected, CSV not created.")
