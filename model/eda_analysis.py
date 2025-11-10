import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from pathlib import Path
import shutil

# Set style for better visualizations
plt.style.use('seaborn')
sns.set_palette("husl")

class DermalScanEDA:
    def __init__(self):
        # Define paths
        self.project_root = Path(__file__).parent.parent
        self.dataset_path = self.project_root.parent / "infosys_dataset"
        self.output_path = self.project_root / "outputs" / "eda_plots"
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_path, exist_ok=True)

    def load_dataset_info(self):
        """Load dataset information and return class distribution"""
        class_counts = {}
        image_dimensions = []
        pixel_values = []

        # Iterate through each class directory
        for class_dir in self.dataset_path.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                images = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.jpg_'))
                class_counts[class_name] = len(images)

                # Sample some images for statistics
                for img_path in list(images)[:10]:  # Take first 10 images from each class
                    try:
                        with Image.open(img_path) as img:
                            img_array = np.array(img)
                            image_dimensions.append(img_array.shape)
                            pixel_values.extend(img_array.ravel())
                    except Exception as e:
                        print(f"Error processing {img_path}: {e}")

        return class_counts, image_dimensions, pixel_values

    def plot_class_distribution(self, class_counts):
        """Plot and save class distribution as bar chart and pie chart"""
        # Bar Plot
        plt.figure(figsize=(10, 6))
        plt.bar(class_counts.keys(), class_counts.values())
        plt.title('Class Distribution in Dataset')
        plt.xlabel('Skin Condition Class')
        plt.ylabel('Number of Images')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_path / 'class_distribution_bar.png')
        plt.close()

        # Pie Chart
        plt.figure(figsize=(10, 8))
        plt.pie(class_counts.values(), labels=class_counts.keys(), autopct='%1.1f%%')
        plt.title('Dataset Class Distribution (%)')
        plt.axis('equal')
        plt.savefig(self.output_path / 'class_distribution_pie.png')
        plt.close()

    def analyze_image_stats(self, image_dimensions, pixel_values):
        """Analyze and plot image statistics"""
        # Image dimensions statistics
        dim_df = pd.DataFrame(image_dimensions, columns=['Height', 'Width', 'Channels'])
        
        # Plot image dimension distribution
        plt.figure(figsize=(12, 4))
        plt.subplot(131)
        dim_df['Height'].hist()
        plt.title('Height Distribution')
        plt.subplot(132)
        dim_df['Width'].hist()
        plt.title('Width Distribution')
        plt.subplot(133)
        sns.boxplot(data=dim_df[['Height', 'Width']])
        plt.title('Image Dimensions')
        plt.tight_layout()
        plt.savefig(self.output_path / 'image_dimensions_stats.png')
        plt.close()

        # Pixel value distribution
        plt.figure(figsize=(10, 6))
        plt.hist(pixel_values, bins=50, density=True, alpha=0.7)
        plt.title('Pixel Value Distribution')
        plt.xlabel('Pixel Value')
        plt.ylabel('Density')
        plt.savefig(self.output_path / 'pixel_distribution.png')
        plt.close()

        # Print summary statistics
        print("\nImage Statistics:")
        print(f"Average dimensions: {dim_df.mean().to_dict()}")
        print(f"Pixel value mean: {np.mean(pixel_values):.2f}")
        print(f"Pixel value std: {np.std(pixel_values):.2f}")

    def plot_sample_images(self, num_samples=3):
        """Plot sample images from each class"""
        classes = [d.name for d in self.dataset_path.iterdir() if d.is_dir()]
        num_classes = len(classes)
        
        plt.figure(figsize=(15, 3*num_classes))
        for idx, class_name in enumerate(classes):
            class_path = self.dataset_path / class_name
            image_files = list(class_path.glob('*.jpg')) + list(class_path.glob('*.jpg_'))
            
            for i in range(min(num_samples, len(image_files))):
                plt.subplot(num_classes, num_samples, idx*num_samples + i + 1)
                with Image.open(image_files[i]) as img:
                    plt.imshow(img)
                plt.title(f"{class_name}\nSample {i+1}")
                plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'sample_images.png')
        plt.close()

    def run_eda(self):
        """Run complete EDA pipeline"""
        print("Starting Exploratory Data Analysis...")
        
        # Load dataset information
        print("\nLoading dataset...")
        class_counts, image_dimensions, pixel_values = self.load_dataset_info()
        
        # Generate and save visualizations
        print("\nGenerating visualizations...")
        self.plot_class_distribution(class_counts)
        self.analyze_image_stats(image_dimensions, pixel_values)
        self.plot_sample_images()
        
        print(f"\nEDA completed. Visualizations saved to: {self.output_path}")
        
        # Print class distribution
        print("\nClass Distribution:")
        for class_name, count in class_counts.items():
            print(f"{class_name}: {count} images")

if __name__ == "__main__":
    eda = DermalScanEDA()
    eda.run_eda()
