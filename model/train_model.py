"""
Model Training Script for AI DermalScan using EfficientNetB0
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from collections import Counter

# Reduce TensorFlow logging verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class DermalScanModel:
    def __init__(self, input_shape=(224, 224, 3), batch_size=16, epochs=80):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.epochs = epochs
        self.project_root = Path(__file__).parent.parent
        self.dataset_path = self.project_root.parent / "infosys_dataset"
        self.model_path = self.project_root / "model" / "efficientnetb0_dermalscan.h5"
        self.plots_path = self.project_root / "outputs" / "training_plots"
        
        # Create output directories
        os.makedirs(self.plots_path, exist_ok=True)
        os.makedirs(self.model_path.parent, exist_ok=True)
        
        # Initialize class weights dictionary
        self.class_weights = None

    def create_model(self, num_classes):
        """Create EfficientNetB0 model optimized for high accuracy"""
        # Load pre-trained EfficientNetB0 model
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Initially freeze base model
        base_model.trainable = False
        
        # Optimized architecture for better performance
        inputs = layers.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)
        
        # Enhanced feature extraction
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        
        # Optimized dense layers with better regularization
        x = layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Final classification layer
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = tf.keras.Model(inputs, outputs)
        
        # Use simple learning rate (no schedule that causes errors)
        initial_learning_rate = 0.001
        
        # Compile with optimized settings
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=initial_learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model

    def calculate_class_weights(self, generator):
        """Calculate balanced class weights."""
        counter = Counter(generator.classes)
        total = sum(counter.values())
        return {class_id: total / (len(counter) * num_images) for class_id, num_images in counter.items()}

    def create_data_generators(self):
        """Create optimized data generators for medical images"""
        # Conservative augmentation for medical images
        train_datagen = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            brightness_range=[0.9, 1.1],
            zoom_range=0.15,
            horizontal_flip=True,
            fill_mode='constant',
            cval=0,
            validation_split=0.2
        )

        # Validation generator - no augmentation
        valid_datagen = ImageDataGenerator(
            preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
            validation_split=0.2
        )

        # Training generator
        train_generator = train_datagen.flow_from_directory(
            self.dataset_path,
            target_size=self.input_shape[:2],
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True,
            seed=42
        )

        # Validation generator
        valid_generator = valid_datagen.flow_from_directory(
            self.dataset_path,
            target_size=self.input_shape[:2],
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False,
            seed=42
        )

        # Calculate class weights
        self.class_weights = self.calculate_class_weights(train_generator)
        print("\nDataset Info:")
        print(f"Classes: {list(train_generator.class_indices.keys())}")
        print(f"Training samples: {train_generator.samples}")
        print(f"Validation samples: {valid_generator.samples}")
        print(f"Class weights: {self.class_weights}")

        return train_generator, valid_generator

    def plot_training_history(self, history):
        """Plot and save comprehensive training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(history.history['loss'], label='Training Loss')
        axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision
        if 'precision' in history.history:
            axes[1, 0].plot(history.history['precision'], label='Training Precision')
            axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
            axes[1, 0].set_title('Model Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Recall
        if 'recall' in history.history:
            axes[1, 1].plot(history.history['recall'], label='Training Recall')
            axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
            axes[1, 1].set_title('Model Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_path / 'training_history.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_confusion_matrix(self, model, generator):
        """Generate and plot enhanced confusion matrix"""
        # Get predictions
        predictions = model.predict(generator, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = generator.classes

        # Get class labels
        class_labels = list(generator.class_indices.keys())

        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_labels,
                   yticklabels=class_labels)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.plots_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Enhanced classification report
        print("\n" + "="*60)
        print("DETAILED CLASSIFICATION REPORT")
        print("="*60)
        print(classification_report(y_true, y_pred, target_names=class_labels, digits=4))
        
        # Calculate overall metrics
        accuracy = np.mean(y_true == y_pred)
        print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    def train(self):
        """Enhanced training process for 95%+ accuracy"""
        print("Initializing optimized training process for 95%+ accuracy...")

        # Create data generators
        train_generator, valid_generator = self.create_data_generators()
        num_classes = len(train_generator.class_indices)

        # Create and compile model
        model = self.create_model(num_classes)
        print(f"\nModel created successfully! Trainable parameters: {model.count_params():,}")

        # Enhanced callbacks for Phase 1
        phase1_callbacks = [
            ModelCheckpoint(
                str(self.model_path),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_accuracy',
                patience=25,
                restore_best_weights=True,
                mode='max',
                verbose=1,
                min_delta=0.001
            ),
            ReduceLROnPlateau(
                monitor='val_accuracy',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                mode='max',
                verbose=1
            )
        ]

        # Phase 1: Train with frozen base
        print("\nPHASE 1: Training with frozen base model...")
        history1 = model.fit(
            train_generator,
            validation_data=valid_generator,
            epochs=50,
            callbacks=phase1_callbacks,
            class_weight=self.class_weights,
            verbose=1
        )

        # Plot initial training history
        self.plot_training_history(history1)
        
        # Generate initial confusion matrix
        print("\nGenerating initial confusion matrix...")
        self.plot_confusion_matrix(model, valid_generator)

        # Check initial accuracy
        phase1_accuracy = max(history1.history['val_accuracy'])
        print(f"\nPhase 1 completed! Best accuracy: {phase1_accuracy:.4f} ({phase1_accuracy*100:.2f}%)")

        # Phase 2: Advanced fine-tuning if needed
        if phase1_accuracy < 0.95:
            print("\nPHASE 2: Starting advanced fine-tuning...")
            self.smart_fine_tune_model(model, train_generator, valid_generator, phase1_accuracy)
        else:
            print("Target 95% accuracy achieved in Phase 1!")

    def smart_fine_tune_model(self, model, train_generator, valid_generator, phase1_accuracy):
        """SMART FINE-TUNING: Only unfreeze specific layers that help"""
        print("\nStarting SMART fine-tuning...")
        
        # Get the base model
        base_model = model.layers[1]
        
        # STRATEGY: Only unfreeze specific block layers, not all layers
        # EfficientNetB0 has blocks 1-7. We'll unfreeze from the top blocks downward
        unfreezing_strategies = [
            # (blocks_to_unfreeze, learning_rate, epochs, description)
            ([6, 7], 0.0001, 20, "Unfreeze only blocks 6-7"),  # Last 2 blocks
            ([5, 6, 7], 0.00005, 15, "Unfreeze blocks 5-7"),   # Last 3 blocks  
            ([4, 5, 6, 7], 0.00002, 12, "Unfreeze blocks 4-7"), # Last 4 blocks
        ]
        
        best_accuracy = phase1_accuracy
        best_model_weights = model.get_weights()
        
        print(f"Starting Phase 2 with accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
        
        for blocks_to_unfreeze, learning_rate, epochs, description in unfreezing_strategies:
            print(f"\nFine-tuning stage: {description}")
            print(f"   Learning rate: {learning_rate:.1e}, Epochs: {epochs}")
            
            if best_accuracy >= 0.95:
                print("   Target already achieved, skipping stage.")
                continue
            
            # SMART UNFREEZING: Only unfreeze specific blocks
            base_model.trainable = True
            
            # Freeze all layers initially
            for layer in base_model.layers:
                layer.trainable = False
            
            # Unfreeze only the specified blocks
            blocks_unfrozen = 0
            for layer in base_model.layers:
                # Check if layer belongs to any of the blocks we want to unfreeze
                for block_num in blocks_to_unfreeze:
                    if f'block{block_num}' in layer.name:
                        layer.trainable = True
                        blocks_unfrozen += 1
                        break
            
            # Also unfreeze the top layers (stem and head)
            for layer in base_model.layers[-20:]:  # Last 20 layers
                layer.trainable = True
            
            trainable_count = sum([l.trainable for l in base_model.layers])
            print(f"   Trainable layers: {trainable_count}/{len(base_model.layers)}")
            print(f"   Blocks unfrozen: {blocks_to_unfreeze}")
            
            # Use very conservative compilation
            model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=learning_rate
                ),
                loss='categorical_crossentropy',  # No label smoothing for fine-tuning
                metrics=['accuracy']
            )
            
            # Conservative callbacks for fine-tuning
            stage_callbacks = [
                ModelCheckpoint(
                    str(self.model_path),
                    monitor='val_accuracy',
                    save_best_only=True,
                    mode='max',
                    verbose=1
                ),
                EarlyStopping(
                    monitor='val_accuracy',
                    patience=10,
                    restore_best_weights=True,
                    mode='max',
                    verbose=1,
                    min_delta=0.0005
                ),
                ReduceLROnPlateau(
                    monitor='val_accuracy',
                    factor=0.3,  # More gentle reduction
                    patience=6,
                    min_lr=1e-8,
                    mode='max',
                    verbose=1
                )
            ]
            
            # Train with smaller batches for stability
            history_stage = model.fit(
                train_generator,
                validation_data=valid_generator,
                epochs=epochs,
                callbacks=stage_callbacks,
                class_weight=self.class_weights,
                verbose=1
            )
            
            # Get the best accuracy from this stage
            stage_accuracy = max(history_stage.history['val_accuracy'])
            
            # STRICT IMPROVEMENT CHECKING
            if stage_accuracy > best_accuracy:
                improvement = stage_accuracy - best_accuracy
                best_accuracy = stage_accuracy
                best_model_weights = model.get_weights()
                print(f"   IMPROVEMENT: {stage_accuracy:.4f} ({stage_accuracy*100:.2f}%) +{improvement:.4f}")
                
                # If we get good improvement, we might not need more aggressive unfreezing
                if improvement > 0.02:  # 2% improvement
                    print("   Excellent improvement! Considering stopping early.")
            else:
                print(f"   NO IMPROVEMENT: {stage_accuracy:.4f} ({stage_accuracy*100:.2f}%)")
                print(f"   Keeping best: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
                model.set_weights(best_model_weights)
                
                # If no improvement with current strategy, skip to next
                continue
            
            # Early exit if target achieved
            if best_accuracy >= 0.95:
                print(f"\nTARGET ACHIEVED: 95% accuracy reached!")
                break
        
        # Final evaluation with best weights
        model.set_weights(best_model_weights)
        
        # Calculate final accuracy properly
        final_predictions = model.predict(valid_generator, verbose=0)
        final_y_pred = np.argmax(final_predictions, axis=1)
        final_y_true = valid_generator.classes
        final_accuracy = np.mean(final_y_true == final_y_pred)
        
        print(f"\nFINAL RESULTS:")
        print(f"Phase 1 accuracy: {phase1_accuracy:.4f} ({phase1_accuracy*100:.2f}%)")
        print(f"Phase 2 accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
        print(f"Improvement: {final_accuracy - phase1_accuracy:.4f} ({(final_accuracy - phase1_accuracy)*100:.2f}%)")
        
        if final_accuracy >= 0.95:
            print("SUCCESS: 95%+ accuracy target achieved!")
        elif final_accuracy > phase1_accuracy:
            print("SUCCESS: Accuracy improved in Phase 2.")
        else:
            print("STABLE: Accuracy maintained. Model is already well-trained.")
            
        # Generate final confusion matrix
        print("\nGenerating final confusion matrix...")
        self.plot_confusion_matrix(model, valid_generator)

if __name__ == "__main__":
    # Initialize with optimized parameters
    trainer = DermalScanModel(
        input_shape=(224, 224, 3),
        batch_size=16,
        epochs=80,
    )
    trainer.train()