"""
Inference Module for AI DermalScan
Handles face detection and skin condition classification using the trained model.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import tensorflow as tf
from typing import Dict, Tuple, List, Union
import time
from dataclasses import dataclass

# Reduce TensorFlow logging verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

@dataclass
class PredictionResult:
    class_name: str
    probability: float
    region: Tuple[int, int, int, int]  # x, y, w, h

class DermalScanInference:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.model_path = self.project_root / "model" / "efficientnetb0_dermalscan.h5"
        self.cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        
        # Load face detection model
        self.face_cascade = cv2.CascadeClassifier(self.cascade_path)
        if self.face_cascade.empty():
            raise ValueError("Error loading face cascade classifier")
        
        # Load classification model
        self.model = self.load_model()
        
        # Define class labels
        self.class_labels = ['clear face', 'darkspots', 'puffy eyes', 'wrinkles']
        
        # Define input shape for the model
        self.input_shape = (224, 224)

    def load_model(self) -> tf.keras.Model:
        try:
            model = tf.keras.models.load_model(self.model_path)
            return model
        except Exception as e:
            raise RuntimeError(f"Error loading model: {str(e)}")

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image_resized = cv2.resize(image_rgb, self.input_shape)
        
        # Normalize
        image_normalized = image_resized / 255.0
        
        # Add batch dimension
        return np.expand_dims(image_normalized, axis=0)

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        return faces

    def predict_region(self, image: np.ndarray, region: Tuple[int, int, int, int]) -> PredictionResult:
        x, y, w, h = region
        
        # Extract and preprocess region
        region_image = image[y:y+h, x:x+w]
        processed_region = self.preprocess_image(region_image)
        
        # Make prediction
        predictions = self.model.predict(processed_region, verbose=0)
        
        # Get class and probability
        class_idx = np.argmax(predictions[0])
        probability = float(predictions[0][class_idx])
        
        return PredictionResult(
            class_name=self.class_labels[class_idx],
            probability=probability,
            region=region
        )

    def annotate_image(self, image: np.ndarray, predictions: List[PredictionResult]) -> np.ndarray:
        image_annotated = image.copy()
        
        for pred in predictions:
            x, y, w, h = pred.region
            cv2.rectangle(
                image_annotated,
                (x, y),
                (x+w, y+h),
                (0, 255, 0),
                2
            )
            text = f"{pred.class_name}: {pred.probability:.2%}"
            cv2.putText(
                image_annotated,
                text,
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
        
        return image_annotated

    def process_image(self, image_path: Union[str, Path]) -> Tuple[np.ndarray, List[Dict]]:
        start_time = time.time()
        
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Error loading image: {image_path}")
        
        # Detect faces
        faces = self.detect_faces(image)
        
        # Get predictions for each face
        predictions = []
        for face in faces:
            prediction = self.predict_region(image, face)
            predictions.append(prediction)
            
        # Annotate image
        annotated_image = self.annotate_image(image, predictions)
        
        # Convert predictions to dictionary format
        prediction_dicts = [
            {
                "class_name": pred.class_name,
                "probability": pred.probability,
                "region": {
                    "x": pred.region[0],
                    "y": pred.region[1],
                    "width": pred.region[2],
                    "height": pred.region[3]
                }
            }
            for pred in predictions
        ]
        
        # Check processing time
        processing_time = time.time() - start_time
        if processing_time > 5:
            print(f"Warning: Processing time ({processing_time:.2f}s) exceeds 5 seconds target")
        
        return annotated_image, prediction_dicts

def process_single_image(image_path: Union[str, Path]) -> Tuple[np.ndarray, List[Dict]]:
    """Convenience function to process a single image"""
    processor = DermalScanInference()
    return processor.process_image(image_path)

if __name__ == "__main__":
    # Example usage
    processor = DermalScanInference()
    
    # Test with a sample image if provided
    test_image_path = Path("path_to_test_image.jpg")  # Replace with actual test image path
    if test_image_path.exists():
        try:
            annotated_image, predictions = processor.process_image(test_image_path)
            
            # Save annotated image
            output_path = Path("output_annotated.jpg")
            cv2.imwrite(str(output_path), annotated_image)
            
            # Print predictions
            print("\nPredictions:")
            for pred in predictions:
                print(f"Class: {pred['class_name']}")
                print(f"Probability: {pred['probability']:.2%}")
                print("Region:", pred['region'])
                print()
            
        except Exception as e:
            print(f"Error processing image: {str(e)}")
    else:
        print("Please provide a test image path to run inference")
