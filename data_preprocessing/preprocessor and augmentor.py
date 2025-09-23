# python
import os # Used for navigating file systems and interacting with directories/files.
import cv2 # Imports OpenCV, the core library for all image processing tasks.
import numpy as np # Used for numerical operations, especially for handling image arrays.
import random # Used for making random choices, like which augmentation to apply.
import pandas as pd # [NEW] Imports the pandas library for data manipulation and creating CSV files.

# ==============================================================================
# 1. CONFIGURATION - Adjust these parameters to fit your needs
# ==============================================================================

# The root directory containing your category folders (e.g., 'dataset/train').
ROOT_DIR = r'D:\Projects\skin-age-detection\datasets\dataset_final'

# The target size for all images.
TARGET_SIZE = (224, 224)

# The minimum number of images you want in each category folder after preprocessing.
TARGET_IMG_COUNT = 800

# The number of new augmented images to generate from EACH original image.
AUGMENTATIONS_PER_IMAGE = 4


# ==============================================================================
# 2. HELPER & AUGMENTATION FUNCTIONS
# ==============================================================================

def preprocess_and_convert_originals(folder_path):
    """
    Resizes all images in a folder to a target size and converts them to PNG.
    The original files are deleted after successful conversion.
    """
    print(f"  Preprocessing originals in '{os.path.basename(folder_path)}'...") # Status message.
    files_to_process = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    for filename in files_to_process: # Loop through each image file.
        original_path = os.path.join(folder_path, filename) # Get the full path to the image.
        base_name, extension = os.path.splitext(filename) # Split the filename from its extension.
        new_path = os.path.join(folder_path, f"{base_name}.png") # Define the new path with a .png extension.

        try:
            image = cv2.imread(original_path) # Read the image file.
            if image is None: # Check if the image failed to load.
                print(f"    - Warning: Could not read {filename}. Skipping.") # Print a warning.
                continue # Move to the next file.

            resized_image = cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_AREA) # Resize the image.
            cv2.imwrite(new_path, resized_image) # Save the resized image as a lossless PNG.

            if original_path != new_path: # Prevents trying to delete a file that was just created.
                os.remove(original_path) # Delete the original file.

        except Exception as e: # Catch any potential errors during processing.
            print(f"    - Error processing {filename}: {e}") # Print the error message.
    
    print("  Preprocessing complete.") # Final status message.

# --- Augmentation Functions ---
# These functions remain the same.
def augment_brightness(image):
    """Applies a random brightness or darkness adjustment."""
    brightness_factor = random.uniform(0.7, 1.3)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    v = np.clip(cv2.multiply(v, np.array([brightness_factor])), 0, 255).astype(np.uint8)
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def augment_flip(image):
    """Applies a horizontal flip."""
    return cv2.flip(image, 1)

def augment_rotate(image):
    """Applies a small, random rotation."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    angle = random.uniform(-10, 10)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, rotation_matrix, (w, h))

def augment_shear(image):
    """Applies a slight horizontal shear."""
    (h, w) = image.shape[:2]
    shear_factor = random.uniform(-0.1, 0.1)
    pts1 = np.float32([[5, 5], [w - 5, 5], [5, h - 5]])
    pts2 = np.float32([[5 + h * shear_factor, 5], [w - 5 + h * shear_factor, 5], [5, h - 5]])
    shear_matrix = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(image, shear_matrix, (w, h))

def augment_sharpen(image):
    """Applies a sharpening filter to the image to enhance details."""
    kernel = np.array([[-1, -1, -1], [-1,  9, -1], [-1, -1, -1]])
    return cv2.filter2D(image, -1, kernel)


# ==============================================================================
# 3. [NEW] CSV LABEL CREATION FUNCTION
# ==============================================================================

def create_label_csvs(root_dir):
    """
    [NEW] Scans the final dataset and creates 'labels.csv' and 'labels_onehot.csv'.
    Args:
      root_dir (str): The path to the root dataset directory.
    """
    print("\n--- Creating label CSV files ---") # Initial status message.
    
    # Get a sorted list of class names (sub-folder names) to ensure consistent order.
    class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    
    all_filenames = [] # Initialize a list to store all final filenames.
    all_labels = [] # Initialize a list to store the corresponding string labels.

    # Iterate through the sorted class names and their corresponding folders.
    for label in class_names:
        category_path = os.path.join(root_dir, label) # Construct the path to the category folder.
        
        # Get all the final PNG images from the folder.
        images_in_folder = [f for f in os.listdir(category_path) if f.lower().endswith('.png')]
        
        all_filenames.extend(images_in_folder) # Add the list of filenames to the master list.
        all_labels.extend([label] * len(images_in_folder)) # Add the string label for each image found.

    # --- Create the first CSV: labels.csv ---
    # Create a pandas DataFrame with two columns: 'filename' and 'label'.
    df_labels = pd.DataFrame({
        'filename': all_filenames,
        'label': all_labels
    })
    # Define the path for the first CSV file.
    labels_csv_path = os.path.join(root_dir, 'labels.csv')
    # Save the DataFrame to a CSV file, without the pandas index column.
    df_labels.to_csv(labels_csv_path, index=False)
    print(f"Successfully created '{labels_csv_path}' with {len(df_labels)} entries.") # Confirmation message.

    # --- Create the second CSV: labels_onehot.csv ---
    # Use pandas' get_dummies function to automatically one-hot encode the 'label' column.
    # prefix='' and prefix_sep='' ensure the new columns are named 'cat', 'dog', etc., not 'label_cat', 'label_dog'.
    df_onehot = pd.get_dummies(df_labels, columns=['label'], prefix='', prefix_sep='')
    # Define the path for the second CSV file.
    onehot_csv_path = os.path.join(root_dir, 'labels_onehot.csv')
    # Save the one-hot encoded DataFrame to a CSV file.
    df_onehot.to_csv(onehot_csv_path, index=False)
    print(f"Successfully created '{onehot_csv_path}' with one-hot encoded labels.") # Confirmation message.


# ==============================================================================
# 4. MAIN SCRIPT LOGIC
# ==============================================================================

def process_and_augment_dataset():
    """
    Main function to preprocess all images and then apply augmentations where needed.
    """
    print(f"Starting dataset processing in: {ROOT_DIR}")
    print(f"All images will be converted to {TARGET_SIZE} PNGs.")
    print(f"Target image count per category is {TARGET_IMG_COUNT}\n")

    available_augmentations = [augment_brightness, augment_flip, augment_rotate, augment_shear, augment_sharpen]

    if not os.path.isdir(ROOT_DIR):
        print(f"Error: The specified directory '{ROOT_DIR}' does not exist.")
        return

    for category_folder in os.listdir(ROOT_DIR):
        category_path = os.path.join(ROOT_DIR, category_folder)

        if os.path.isdir(category_path):
            print(f"--- Processing folder: '{category_folder}' ---")
            
            # Step 1: Preprocess all original images in the folder first.
            preprocess_and_convert_originals(category_path)
            
            # Step 2: Now count the uniform PNG images and decide if augmentation is needed.
            image_files = [f for f in os.listdir(category_path) if f.lower().endswith('.png')]
            current_img_count = len(image_files)
            print(f"Found {current_img_count} preprocessed images.")

            if 0 < current_img_count < TARGET_IMG_COUNT:
                print(f"Image count is below {TARGET_IMG_COUNT}. Starting augmentation...\n")
                
                for img_name in image_files:
                    original_img_path = os.path.join(category_path, img_name)
                    original_image = cv2.imread(original_img_path)

                    if original_image is None:
                        print(f"Warning: Could not read preprocessed image {img_name}. Skipping.")
                        continue

                    for i in range(AUGMENTATIONS_PER_IMAGE):
                        augmentation_function = random.choice(available_augmentations)
                        augmented_image = augmentation_function(original_image)
                        
                        base_name, extension = os.path.splitext(img_name)
                        new_filename = f"{base_name}_aug_{i+1}{extension}"
                        new_image_path = os.path.join(category_path, new_filename)
                        
                        cv2.imwrite(new_image_path, augmented_image)
                
                final_count = len([f for f in os.listdir(category_path) if f.lower().endswith('.png')])
                print(f"Augmentation for '{category_folder}' complete. New image count: {final_count}\n")
            
            elif current_img_count == 0:
                 print("Folder is empty. Skipping.\n")
            else:
                print(f"Image count is sufficient. Skipping augmentation.\n")

    print("--- All folders processed. ---")
    
    # [NEW] Step 3: After all processing is done, create the label CSV files.
    create_label_csvs(ROOT_DIR)
    
    print("\n--- Script finished. ---")


# ==============================================================================
# 5. SCRIPT EXECUTION
# ==============================================================================
if __name__ == '__main__':
    process_and_augment_dataset()