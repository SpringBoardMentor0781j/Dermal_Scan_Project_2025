import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img

# -----------------------------
# Paths
# -----------------------------
raw_data_dir = "raw_dataset"            # Input dataset folder
output_dir = "processed_dataset"        # Output dataset folder
os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# Augmentation Config
# -----------------------------
datagen = ImageDataGenerator(
    rescale=1./255,          # Normalize images
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest"
)

# -----------------------------
# Parameters
# -----------------------------
augment_per_image = 4   # how many augmented versions per image
img_size = (224, 224)

# -----------------------------
# Preprocess + Augment
# -----------------------------
for class_name in os.listdir(raw_data_dir):
    class_path = os.path.join(raw_data_dir, class_name)
    if not os.path.isdir(class_path):
        continue

    print(f"Processing class: {class_name}")
    output_class_dir = os.path.join(output_dir, class_name)
    os.makedirs(output_class_dir, exist_ok=True)

    for img_file in os.listdir(class_path):
        img_path = os.path.join(class_path, img_file)
        try:
            img = load_img(img_path, target_size=img_size)
            x = img_to_array(img)
            x = x.reshape((1,) + x.shape)

            prefix = os.path.splitext(img_file)[0]

            # Save original resized image
            array_to_img(x[0]).save(os.path.join(output_class_dir, img_file))

            # Save augmented images
            i = 0
            for batch in datagen.flow(
                x,
                batch_size=1,
                save_to_dir=output_class_dir,
                save_prefix=prefix + "_aug",
                save_format="jpg"
            ):
                i += 1
                if i >= augment_per_image:
                    break

        except Exception as e:
            print(f"❌ Error processing {img_path}: {e}")

    print(f"✅ Completed class: {class_name}")

print("🎉 Preprocessing + Augmentation finished. Check 'processed_dataset/'")
