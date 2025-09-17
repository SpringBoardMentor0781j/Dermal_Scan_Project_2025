import os, random, shutil
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import MobileNet

# --- Parameters ---
DATASET_DIR = r'dataset_final'
IMG_SIZE = (224,224)
BATCH_SIZE = 16
MAX_IMAGES_PER_CLASS = 150
DESIRED_CLASSES = ['clear_face', 'darkspots', 'puffy_eyes', 'wrinkles']

# --- Clean dataset ---
for folder in os.listdir(DATASET_DIR):
    path = os.path.join(DATASET_DIR, folder)
    if folder not in DESIRED_CLASSES:
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            os.remove(path)

# --- Load images ---
images, labels = [], []
for idx, category in enumerate(DESIRED_CLASSES):
    category_path = os.path.join(DATASET_DIR, category)
    img_files = [f for f in os.listdir(category_path) if f.lower().endswith(('jpg','jpeg','png'))]
    if len(img_files) > MAX_IMAGES_PER_CLASS:
        img_files = random.sample(img_files, MAX_IMAGES_PER_CLASS)
    for img_file in img_files:
        img_path = os.path.join(category_path, img_file)
        img = load_img(img_path, target_size=IMG_SIZE)
        img_array = img_to_array(img)/255.0
        images.append(img_array)
        labels.append(idx)

images = np.array(images, dtype=np.float32)
labels = np.array(labels)
num_classes = len(DESIRED_CLASSES)

# --- Train/Val/Test split ---
X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, stratify=labels, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

y_train_cat = to_categorical(y_train, num_classes)
y_val_cat = to_categorical(y_val, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# --- Class weights ---
class_weights_array = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights_array))
print("Class weights:", class_weights)

# --- Build MobileNet backbone + EfficientNet-style head ---
def build_model(num_classes):
    backbone = MobileNet(weights='imagenet', include_top=False, input_shape=(224,224,3))
    backbone.trainable = True  # can fine-tune later if needed
    x = backbone.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=backbone.input, outputs=predictions)
    return model

# --- Compile & train ---
def train_model(model, epochs=20, batch_size=BATCH_SIZE):
    model.compile(optimizer=Adam(1e-3),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        X_train, y_train_cat,
        validation_data=(X_val, y_val_cat),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weights,
        shuffle=True
    )

    model.save('mobilenet_effnet_head.keras')
    model.save('mobilenet_effnet_head.h5')
    print("Model saved as '.keras' and '.h5'")
    return model, history

# --- Evaluate and visualize predictions ---
def evaluate_and_show(model):
    preds = model.predict(X_test)
    pred_labels = np.argmax(preds, axis=1)
    test_acc = np.mean(pred_labels == y_test)
    print(f"\nTest accuracy: {test_acc*100:.2f}%")

    plt.figure(figsize=(12,8))
    for idx, category in enumerate(DESIRED_CLASSES,1):
        category_path = os.path.join(DATASET_DIR, category)
        img_files = [f for f in os.listdir(category_path) if f.lower().endswith(('jpg','jpeg','png'))]
        if not img_files: continue
        chosen_img = random.choice(img_files)
        img_path = os.path.join(category_path, chosen_img)
        img = load_img(img_path, target_size=IMG_SIZE)
        img_array = img_to_array(img)/255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array, verbose=0)
        predicted_class = np.argmax(pred, axis=1)[0]
        confidence = np.max(pred)
        predicted_label = DESIRED_CLASSES[predicted_class]

        plt.subplot(1, len(DESIRED_CLASSES), idx)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"True: {category}\nPred: {predicted_label} ({confidence:.2f})")
    plt.tight_layout()
    plt.show()

# --- Main ---
if __name__ == '__main__':
    model = build_model(num_classes)
    model, history = train_model(model, epochs=20, batch_size=BATCH_SIZE)
    evaluate_and_show(model)
