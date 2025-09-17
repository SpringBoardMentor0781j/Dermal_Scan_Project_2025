# This file is bit unpredictable(for me) and should not be run unless reviewed

import os # Provides a way of using operating system dependent functionality.
import cv2 # OpenCV library for computer vision tasks like image reading, resizing, and manipulation.
import numpy as np # Library for numerical operations, especially with arrays.
import json # Library for working with JSON data, used for configuration management.
import shutil # Provides high-level file operations, like removing directories.
import csv # Implements classes to read and write tabular data in CSV format.
from tensorflow.keras.applications import EfficientNetB0 # The specific pre-trained model we will use for transfer learning.
from tensorflow.keras.applications.efficientnet import preprocess_input # A function to prepare images for the EfficientNetB0 model.
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Keras utility for real-time data augmentation.
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D # Keras layers to build the custom head of our model.
from tensorflow.keras.models import Model # The base class for creating models in Keras.
from tensorflow.keras.utils import to_categorical # Converts a class vector (integers) to a binary class matrix (one-hot encoding).

class AutomatedDatasetCurator:
    """
    A class to automate the process of cleaning, classifying, preprocessing,
    and augmenting a raw image dataset to create a final, balanced dataset
    ready for model training.
    """

    def __init__(self, raw_dir="dataset_raw", final_dir="dataset_final", config_path="config.json"):
        """
        Initializes the curator with directory paths and loads the configuration.
        It also pre-loads the face detection and classification models into memory.

        Parameters:
        - raw_dir (str): The path to the directory containing the unorganized raw images.
        - final_dir (str): The path where the curated, final dataset will be saved.
        - config_path (str): The path to the JSON configuration file.
        """
        self.raw_dir = raw_dir # Assign the raw dataset directory path.
        self.final_dir = final_dir # Assign the final dataset directory path.
        self.config_path = config_path # Assign the configuration file path.
        self.config = self._load_config() # Load the configuration from the JSON file.

        # --- Extract parameters from the loaded configuration ---
        self.classes = self.config['classes'] # Dictionary of class names.
        self.classification_keywords = self.config['classification_keywords'] # Keywords for rule-based classification.
        self.augmentation_params = self.config['augmentation_params'] # Parameters for image augmentation.
        self.processing_params = self.config['processing_params'] # Parameters for image preprocessing.

        # --- Initialize data storage and counters ---
        self.class_stats = {class_name: [] for class_name in self.classes.values()} # A dictionary to store statistics for each class.
        self.class_counters = {class_name: 1 for class_name in self.classes.values()} # A counter for unique filenames in each class.
        self.rejected_images = [] # A list to store information about images that were rejected.

        # --- Initialize Models ---
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # Load the Haar Cascade for face detection.
        self.classification_model = self._build_model() # Build and compile the image classification model.

        self._setup_final_directories() # Set up the directory structure for the final dataset.

    def _load_config(self):
        """
        Loads the configuration from a JSON file. If the file doesn't exist,
        it creates a default one.

        Returns:
        - dict: A dictionary containing the configuration parameters.
        """
        if not os.path.exists(self.config_path): # Check if the config file exists.
            self._create_default_config() # If not, create a default config file.
        with open(self.config_path, 'r') as f: # Open the config file in read mode.
            return json.load(f) # Load and return the JSON content as a dictionary.

    def _create_default_config(self):
        """
        Creates a default config.json file with standard parameters. This is
        useful for the first run or if the config file is deleted.
        """
        default_config = { # Define the default configuration dictionary.
            "classes": { # Define the class labels.
                'clear_skin': 'clear_skin',
                'puffy_eyes': 'puffy_eyes',
                'dark_spots': 'dark_spots',
                'wrinkles': 'wrinkles'
            },
            "classification_keywords": { # Define keywords for classifying images based on their filename/folder.
                "clear_skin": ["clear", "normal", "healthy", "young", "smooth", "clean"],
                "puffy_eyes": ["puffy", "swollen", "bags", "under_eye", "tired", "baggy"],
                "dark_spots": ["spots", "dark", "pigment", "age_spot", "blemish", "mark"],
                "wrinkles": ["wrinkles", "lines", "aged", "old", "crow", "forehead"]
            },
            "augmentation_params": { # Define parameters for the ImageDataGenerator.
                "rotation_range": 15,
                "width_shift_range": 0.1,
                "height_shift_range": 0.1,
                "zoom_range": 0.1,
                "horizontal_flip": True,
                "brightness_range": [0.8, 1.2],
                "fill_mode": "nearest"
            },
            "processing_params": { # Define parameters for image validation and preprocessing.
                "target_size": 224, # Target image size for the model.
                "min_dim": 50, # Minimum dimension for an image to be considered valid.
                "brightness_range": [10, 245], # Valid brightness range.
                "contrast_std": 15, # Minimum standard deviation for contrast.
                "hybrid_confidence_threshold": 0.7 # Confidence threshold for using model prediction over keyword prediction.
            }
        }
        with open(self.config_path, 'w') as f: # Open the config file in write mode.
            json.dump(default_config, f, indent=4) # Write the default config dictionary to the file with nice formatting.
        print(f"Default config file created at {self.config_path}. You can modify it.") # Inform the user.

    def _build_model(self):
        """
        Builds the EfficientNetB0 model with a custom classification head for transfer learning.

        Returns:
        - tensorflow.keras.Model: The compiled Keras model.
        """
        target_size = self.processing_params['target_size'] # Get the target image size from config.
        # Load the EfficientNetB0 base model, pre-trained on ImageNet, without its top classification layer.
        base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(target_size, target_size, 3))

        x = base_model.output # Get the output of the base model.
        x = GlobalAveragePooling2D()(x) # Add a global average pooling layer to reduce dimensions.
        x = Dense(128, activation='relu')(x) # Add a fully-connected layer with 128 units and ReLU activation.
        # Add the final prediction layer with softmax activation for multi-class classification.
        predictions = Dense(len(self.classes), activation='softmax')(x)

        model = Model(inputs=base_model.input, outputs=predictions) # Create the final model.
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) # Compile the model.
        return model # Return the compiled model.

    def _setup_final_directories(self):
        """
        Creates a clean directory structure for the final dataset. It will delete
        the existing final directory to ensure a fresh start.
        """
        if os.path.exists(self.final_dir): # Check if the final directory already exists.
            shutil.rmtree(self.final_dir) # If it exists, remove it and all its contents.
        os.makedirs(self.final_dir) # Create the main final directory.

        for class_name in self.classes.values(): # Loop through all class names.
            class_dir = os.path.join(self.final_dir, class_name) # Create the path for the class subdirectory.
            os.makedirs(class_dir, exist_ok=True) # Create the class subdirectory.

        print(f"Created a clean {self.final_dir} directory structure.") # Inform the user.

    def _get_all_image_paths(self):
        """
        Traverses the raw dataset directory to find all image files.

        Returns:
        - list: A list of full paths to all found image files.
        """
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff') # Define valid image file extensions.
        image_paths = [] # Initialize an empty list to store image paths.
        for root, _, files in os.walk(self.raw_dir): # Recursively walk through the raw directory.
            for file in files: # Loop through all files in the current directory.
                if file.lower().endswith(image_extensions): # Check if the file has a valid image extension.
                    image_paths.append(os.path.join(root, file)) # If it's an image, add its full path to the list.
        return image_paths # Return the list of image paths.

    def _validate_image_quality(self, image_path):
        """
        Validates an image based on its dimensions, brightness, and contrast.

        Parameters:
        - image_path (str): The path to the image file.

        Returns:
        - tuple (bool, str): A tuple containing a boolean indicating validity and a reason string.
        """
        try:
            image = cv2.imread(image_path) # Read the image using OpenCV.
            if image is None: # Check if the image failed to load.
                return False, "Cannot load image" # Return invalid.

            h, w, _ = image.shape # Get the height and width of the image.
            if h < self.processing_params['min_dim'] or w < self.processing_params['min_dim']: # Check if dimensions are too small.
                return False, f"Image too small: {w}x{h}" # Return invalid.

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert the image to grayscale for analysis.
            mean_brightness = np.mean(gray) # Calculate the mean brightness.
            min_bright, max_bright = self.processing_params['brightness_range'] # Get brightness range from config.

            if not (min_bright <= mean_brightness <= max_bright): # Check if brightness is out of range.
                return False, "Image too dark or too bright" # Return invalid.

            if np.std(gray) < self.processing_params['contrast_std']: # Check if contrast (standard deviation) is too low.
                return False, "Insufficient contrast" # Return invalid.

            return True, "Valid" # If all checks pass, the image is valid.
        except Exception as e: # Catch any other errors during validation.
            return False, f"Error validating: {str(e)}" # Return invalid with the error message.

    def _classify_by_folder_name(self, image_path):
        """
        Classifies an image based on keywords found in its file path.

        Parameters:
        - image_path (str): The path to the image file.

        Returns:
        - tuple (str or None, int): The predicted class name and a score.
        """
        folder_path = os.path.dirname(image_path) # Get the directory path of the image.
        filename = os.path.basename(image_path) # Get the filename of the image.
        combined_text = f"{folder_path.lower()} {filename.lower()}" # Combine folder and file name into one string.

        class_scores = {class_name: 0 for class_name in self.classes.values()} # Initialize scores for each class to 0.

        for class_name, keywords in self.classification_keywords.items(): # Loop through classes and their keywords.
            for keyword in keywords: # Loop through each keyword.
                if keyword in combined_text: # Check if the keyword is in the combined path text.
                    class_scores[class_name] += 1 # If found, increment the score for that class.

        best_class = max(class_scores, key=class_scores.get) # Find the class with the highest score.
        best_score = class_scores[best_class] # Get the highest score.

        return (best_class, best_score) if best_score > 0 else (None, 0) # Return the best class if score is positive.

    def _classify_by_content(self, image):
        """
        Classifies an image using the pre-trained EfficientNetB0 model.

        Parameters:
        - image (np.array): The image data as a NumPy array.

        Returns:
        - tuple (str, float): The predicted class name and the confidence score.
        """
        target_size = self.processing_params['target_size'] # Get the target size from config.
        resized_img = cv2.resize(image, (target_size, target_size)) # Resize the image to the model's expected input size.
        img_array = np.expand_dims(resized_img, axis=0) # Add a batch dimension.
        img_array = preprocess_input(img_array) # Preprocess the image for the model.

        predictions = self.classification_model.predict(img_array, verbose=0)[0] # Get model predictions.

        predicted_class_idx = np.argmax(predictions) # Get the index of the class with the highest probability.
        confidence = predictions[predicted_class_idx] # Get the confidence score for that class.
        predicted_class = list(self.classes.values())[predicted_class_idx] # Get the class name from the index.

        return predicted_class, confidence # Return the prediction and confidence.

    def _hybrid_classification(self, image_path):
        """
        Combines keyword-based and content-based classification for a robust prediction.

        Parameters:
        - image_path (str): The path to the image.

        Returns:
        - tuple (str or None, float): The final predicted class name and confidence.
        """
        image = cv2.imread(image_path) # Read the image file.
        if image is None: # Check if image loading failed.
            return None, 0 # Return no prediction.

        content_class, content_confidence = self._classify_by_content(image) # Get prediction from the ML model.
        keyword_class, _ = self._classify_by_folder_name(image_path) # Get prediction from keywords.

        # If the model is confident enough, use its prediction. Otherwise, fall back to the keyword-based prediction.
        if content_confidence >= self.processing_params['hybrid_confidence_threshold']:
            return content_class, content_confidence # Return model's confident prediction.
        elif keyword_class:
            return keyword_class, 0.5 # Return keyword prediction with a medium confidence score.
        else:
            return None, 0 # If neither method works, return no prediction.

    def _smart_resize(self, image):
        """
        Resizes an image to 224x224. If a face is detected, it crops around the
        face first. Otherwise, it resizes and pads the image.

        Parameters:
        - image (np.array): The input image.

        Returns:
        - np.array: The resized 224x224 image.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert to grayscale for face detection.
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50)) # Detect faces.
        target_size = self.processing_params['target_size'] # Get target size from config.

        if len(faces) > 0: # If at least one face is found.
            (x, y, w, h) = faces[0] # Get coordinates of the first face.
            padding = int(max(w, h) * 0.2) # Calculate padding as 20% of the largest dimension.
            # Define crop boundaries, ensuring they don't go outside the image.
            x_crop, y_crop = max(0, x - padding), max(0, y - padding)
            w_crop, h_crop = min(image.shape[1] - x_crop, w + 2 * padding), min(image.shape[0] - y_crop, h + 2 * padding)
            cropped_img = image[y_crop:y_crop+h_crop, x_crop:x_crop+w_crop] # Crop the image.
            result = cv2.resize(cropped_img, (target_size, target_size), interpolation=cv2.INTER_AREA) # Resize the crop.
        else: # If no face is found.
            h, w = image.shape[:2] # Get image dimensions.
            scale = target_size / max(h, w) # Calculate scaling factor to fit within 224x224.
            new_w, new_h = int(w * scale), int(h * scale) # Calculate new dimensions.
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA) # Resize the image.
            # Create a gray canvas and paste the resized image in the center.
            result = np.ones((target_size, target_size, 3), dtype=np.uint8) * 128
            start_h, start_w = (target_size - new_h) // 2, (target_size - new_w) // 2
            result[start_h:start_h+new_h, start_w:start_w+new_w] = resized
        return result # Return the processed image.

    def _apply_enhancements(self, image):
        """
        Applies gentle contrast and sharpness enhancements using OpenCV.

        Parameters:
        - image (np.array): The input image.

        Returns:
        - np.array: The enhanced image.
        """
        # Increase contrast slightly using cv2.convertScaleAbs.
        enhanced_image = cv2.convertScaleAbs(image, alpha=1.1, beta=5)
        # Apply a sharpening kernel using cv2.filter2D.
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        enhanced_image = cv2.filter2D(enhanced_image, -1, kernel)
        return enhanced_image # Return the final enhanced image.


    def _preprocess_image(self, image_path):
        """
        A complete preprocessing pipeline for a single image.

        Parameters:
        - image_path (str): Path to the image file.

        Returns:
        - np.array or None: The preprocessed image data or None if processing fails.
        """
        try:
            image = cv2.imread(image_path) # Read the image.
            if image is None: return None # If loading fails, return None.

            image = self._smart_resize(image) # Apply smart resizing.
            image = self._apply_enhancements(image) # Apply enhancements.
            return image # Return the processed image.
        except Exception as e: # Catch any processing errors.
            print(f"Error processing {image_path}: {str(e)}") # Print the error.
            return None # Return None on failure.

    def _calculate_image_stats(self, image):
        """
        Calculates brightness and the ratio of Red to Green channels.

        Parameters:
        - image (np.array): The input image data.

        Returns:
        - tuple (float, float): The calculated brightness and R/G ratio.
        """
        brightness = np.mean(image) # Calculate mean brightness.
        # Calculate the mean Red-to-Green channel ratio, avoiding division by zero.
        r_channel = image[:, :, 2].astype(np.float32) # Red channel is the last one in BGR.
        g_channel = image[:, :, 1].astype(np.float32) # Green channel.
        rg_ratio = np.mean(r_channel / np.maximum(g_channel, 1)) # Divide R by G, with G being at least 1.
        return brightness, rg_ratio # Return the stats.

    def run_automated_classification(self):
        """
        The main loop that classifies all raw images and saves them to the
        final directory structure.
        """
        image_paths = self._get_all_image_paths() # Get all image paths from the raw directory.
        if not image_paths: # Check if any images were found.
            print("No images found in the raw dataset directory. Exiting.")
            return # Exit if no images.

        print(f"\nStarting automated classification of {len(image_paths)} images...")
        for image_path in image_paths: # Loop through each image path.
            is_valid, reason = self._validate_image_quality(image_path) # Validate the image.
            if not is_valid: # If the image is not valid.
                self.rejected_images.append({'path': image_path, 'reason': reason}) # Add it to the rejected list.
                continue # Skip to the next image.

            predicted_class, confidence = self._hybrid_classification(image_path) # Classify the image.
            if predicted_class is None: # If classification is uncertain.
                self.rejected_images.append({'path': image_path, 'reason': 'Uncertain classification'}) # Reject it.
                continue # Skip to the next image.

            processed_image = self._preprocess_image(image_path) # Preprocess the image.
            if processed_image is None: # If preprocessing fails.
                self.rejected_images.append({'path': image_path, 'reason': 'Processing failed'}) # Reject it.
                continue # Skip.

            brightness, rg_ratio = self._calculate_image_stats(processed_image) # Calculate stats for the processed image.

            # --- Save the processed image and its metadata ---
            output_filename = f"{self.class_counters[predicted_class]:04d}.png" # Create a new filename.
            output_path = os.path.join(self.final_dir, predicted_class, output_filename) # Create the full output path.
            cv2.imwrite(output_path, processed_image) # Save the image.

            # Store the metadata for later analysis.
            self.class_stats[predicted_class].append({
                'filename': output_filename,
                'brightness': brightness,
                'rg_ratio': rg_ratio,
                'original_file': os.path.basename(image_path),
                'confidence': confidence
            })
            self.class_counters[predicted_class] += 1 # Increment the filename counter for that class.

        classified_count = sum(len(stats) for stats in self.class_stats.values()) # Count total classified images.
        print(f"\nAutomated classification complete!")
        print(f"  Classified: {classified_count} images")
        print(f"  Rejected: {len(self.rejected_images)} images")

    def run_balancing(self, target_count=None):
        """
        Balances the dataset classes by applying data augmentation to the
        minority classes until they reach the target count.

        Parameters:
        - target_count (int, optional): The target number of images per class.
          If None, it defaults to the size of the largest class.
        """
        print("\nBalancing classes...")
        current_counts = {cls: len(stats) for cls, stats in self.class_stats.items()} # Get current counts.
        if not current_counts or max(current_counts.values()) == 0: # Check if there are images to balance.
            print("No classified images found. Skipping balancing.")
            return

        if not target_count: # If no target count is provided.
            target_count = max(current_counts.values()) # Use the count of the largest class as the target.

        print(f"Target count per class: {target_count}")

        # Initialize the Keras data augmentation generator.
        datagen = ImageDataGenerator(**self.augmentation_params)

        for class_name, current_count in current_counts.items(): # Loop through each class.
            if current_count >= target_count: # If the class is already balanced or over-represented.
                print(f"  {class_name}: {current_count} samples (no augmentation needed)")
                continue # Skip it.

            needed = target_count - current_count # Calculate how many new images are needed.
            print(f"  {class_name}: {current_count} -> {target_count} (+{needed} augmented)")
            class_dir = os.path.join(self.final_dir, class_name) # Get the directory for the current class.

            # Load all existing images for this class into memory for augmentation.
            existing_images = []
            for stats in self.class_stats[class_name]:
                img_path = os.path.join(class_dir, stats['filename'])
                img = cv2.imread(img_path)
                if img is not None:
                    existing_images.append(img)
            
            if not existing_images: continue # Skip if no images could be loaded.

            existing_images = np.array(existing_images) # Convert the list of images to a NumPy array.
            
            generated_count = 0 # Counter for generated images.
            for batch in datagen.flow(existing_images, batch_size=len(existing_images), shuffle=True):
                for i in range(len(batch)): # Loop through images in the augmented batch.
                    if generated_count >= needed: break # Stop if we've generated enough images.

                    aug_img = batch[i].astype(np.uint8) # Get the augmented image.
                    aug_filename = f"aug_{self.class_counters[class_name]:04d}.png" # Create a filename for it.
                    aug_path = os.path.join(class_dir, aug_filename) # Create the full path.
                    cv2.imwrite(aug_path, aug_img) # Save the augmented image.
                    
                    self.class_counters[class_name] += 1 # Increment the counter.
                    generated_count += 1 # Increment the generated count.
                if generated_count >= needed: break # Exit the outer loop as well.

    def _write_list_of_dicts_to_csv(self, data, filepath):
        """
        A helper function to write a list of dictionaries to a CSV file.

        Parameters:
        - data (list): The list of dictionaries to write.
        - filepath (str): The path to the output CSV file.
        """
        if not data: return # If there's no data, do nothing.
        with open(filepath, 'w', newline='') as f: # Open the file in write mode.
            writer = csv.DictWriter(f, fieldnames=data[0].keys()) # Create a CSV writer.
            writer.writeheader() # Write the header row.
            writer.writerows(data) # Write all the data rows.

    def generate_reports(self):
        """
        Generates CSV reports for the final dataset metadata and rejected images.
        """
        print("\nGenerating analysis reports...")
        
        # --- Create and save the metadata report ---
        all_stats = [] # Initialize an empty list for all image statistics.
        for class_name, stats_list in self.class_stats.items(): # Loop through the stats.
            for stats in stats_list: # Loop through each image's stats.
                stats['class'] = class_name # Add the class name to the stats dictionary.
                all_stats.append(stats) # Append it to the main list.

        if not all_stats: # Check if there's any data.
            print("Warning: No data available for reporting.")
            return

        metadata_path = os.path.join(self.final_dir, 'dataset_metadata.csv') # Define the path for the metadata CSV.
        self._write_list_of_dicts_to_csv(all_stats, metadata_path) # Write the CSV file.
        print(f"  Dataset metadata saved to: {metadata_path}")

        # --- Create and save the rejected images report ---
        if self.rejected_images: # Check if there were any rejected images.
            rejected_path = os.path.join(self.final_dir, 'rejected_images_report.csv') # Define the path for the report.
            self._write_list_of_dicts_to_csv(self.rejected_images, rejected_path) # Write the report.
            print(f"  Rejected images report saved to: {rejected_path}")

        # --- Create labels and a final JSON metadata file ---
        self._create_final_metadata(all_stats)


    def _create_final_metadata(self, all_stats):
        """
        Creates a final JSON file containing one-hot encoded labels and other
        useful metadata about the dataset creation process.
        """
        print("\nCreating final labels and metadata JSON...")
        
        # --- Create Labels ---
        unique_classes = sorted(list(self.classes.values())) # Get a sorted list of unique class names.
        class_to_int = {class_name: i for i, class_name in enumerate(unique_classes)} # Create a mapping from class name to integer.
        
        filenames = [stats['filename'] for stats in all_stats] # Get a list of filenames.
        class_names = [stats['class'] for stats in all_stats] # Get a list of class names.
        labels_numeric = [class_to_int[name] for name in class_names] # Convert class names to integers.
        onehot_labels = to_categorical(labels_numeric, num_classes=len(unique_classes)) # Convert integers to one-hot encoding.

        # --- Assemble the final metadata dictionary ---
        final_metadata = {
            'dataset_info': {
                'total_samples': len(all_stats),
                'num_classes': len(unique_classes),
                'classes': unique_classes,
                'image_size': [self.processing_params['target_size'], self.processing_params['target_size'], 3],
            },
            'labels': {
                'filenames': filenames,
                'classes': class_names,
                'labels_one_hot': onehot_labels.tolist() # Convert numpy array to list for JSON serialization.
            }
        }
        
        metadata_path = os.path.join(self.final_dir, 'final_dataset_metadata.json') # Define the output path.
        with open(metadata_path, 'w') as f: # Open the file in write mode.
            json.dump(final_metadata, f, indent=4) # Save the dictionary as a JSON file.
        
        print(f"  Final metadata and labels saved to: {metadata_path}")


    def run_pipeline(self):
        """
        Runs the entire curation pipeline from start to finish.
        """
        print("--- Starting Automated Dataset Curation Pipeline ---")
        self.run_automated_classification() # Step 1: Classify and preprocess all images.
        self.run_balancing() # Step 2: Balance the dataset using augmentation.
        self.generate_reports() # Step 3: Generate the final CSV and JSON reports.
        print("\n--- Pipeline finished successfully! ---")
        print(f"The curated dataset is ready in the '{self.final_dir}' folder.")

# This block ensures the code inside only runs when the script is executed directly.
if __name__ == '__main__':
    curator = AutomatedDatasetCurator() # Create an instance of the curator.
    curator.run_pipeline() # Run the entire pipeline.