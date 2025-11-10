
import os
import csv
import pandas as pd
import cv2
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union, Optional
import shutil
import json

class DermalScanLogger:
    def __init__(self):
        """Initialize the logger with required paths and files"""
        self.project_root = Path(__file__).parent.parent
        self.outputs_dir = self.project_root / "outputs"
        self.annotated_images_dir = self.outputs_dir / "annotated_images"
        self.logs_file = self.outputs_dir / "prediction_logs.csv"
        
        # Create necessary directories
        self._setup_directories()
        
        # Initialize log file if it doesn't exist
        self._initialize_log_file()
        
    def _setup_directories(self):
        """Create required directories if they don't exist"""
        os.makedirs(self.outputs_dir, exist_ok=True)
        os.makedirs(self.annotated_images_dir, exist_ok=True)
    
    def _initialize_log_file(self):
        """Create log file with headers if it doesn't exist"""
        if not self.logs_file.exists():
            headers = [
                'timestamp',
                'filename',
                'class_name',
                'confidence',
                'region_x',
                'region_y',
                'region_width',
                'region_height',
                'image_path'
            ]
            with open(self.logs_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)

    def log_prediction(self, 
                      image_path: Union[str, Path],
                      predictions: List[Dict],
                      annotated_image: np.ndarray) -> None:
        # Generate timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Save annotated image
        filename = Path(image_path).name
        base_name = Path(filename).stem
        annotated_filename = f"{base_name}_annotated.jpg"
        annotated_path = self.annotated_images_dir / annotated_filename
        cv2.imwrite(str(annotated_path), annotated_image)
        
        # Log each prediction
        with open(self.logs_file, 'a', newline='') as f:
            writer = csv.writer(f)
            for pred in predictions:
                writer.writerow([
                    timestamp,
                    filename,
                    pred['class_name'],
                    f"{pred['probability']:.2%}",
                    pred['region']['x'],
                    pred['region']['y'],
                    pred['region']['width'],
                    pred['region']['height'],
                    str(annotated_path)
                ])

    def get_recent_predictions(self, 
                             limit: Optional[int] = None,
                             class_filter: Optional[str] = None) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.logs_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Apply filters
            if class_filter:
                df = df[df['class_name'] == class_filter]
            
            # Sort by timestamp (most recent first)
            df = df.sort_values('timestamp', ascending=False)
            
            # Apply limit
            if limit:
                df = df.head(limit)
            
            return df
            
        except pd.errors.EmptyDataError:
            # Return empty DataFrame with correct columns if log is empty
            return pd.DataFrame(columns=[
                'timestamp', 'filename', 'class_name', 'confidence',
                'region_x', 'region_y', 'region_width', 'region_height',
                'image_path'
            ])

    def export_logs(self, 
                   output_path: Union[str, Path],
                   format: str = 'csv') -> str:
        df = pd.read_csv(self.logs_file)
        output_path = Path(output_path)
        
        if format.lower() == 'csv':
            df.to_csv(output_path, index=False)
        elif format.lower() == 'json':
            df.to_json(output_path, orient='records', indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")
            
        return str(output_path)

    def get_class_statistics(self) -> Dict[str, Dict]:
        df = pd.read_csv(self.logs_file)
        stats = {}
        
        for class_name in df['class_name'].unique():
            class_df = df[df['class_name'] == class_name]
            stats[class_name] = {
                'total_predictions': len(class_df),
                'average_confidence': class_df['confidence'].str.rstrip('%').astype(float).mean(),
                'last_prediction': class_df['timestamp'].max()
            }
        
        return stats

    def clear_logs(self, keep_days: Optional[int] = None) -> None:
        if keep_days is None:
            # Clear all logs
            if self.logs_file.exists():
                self._initialize_log_file()  # Recreate with headers only
            
            # Clear all annotated images
            shutil.rmtree(self.annotated_images_dir)
            os.makedirs(self.annotated_images_dir)
        else:
            # Keep recent logs only
            df = pd.read_csv(self.logs_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=keep_days)
            
            # Filter recent logs
            recent_logs = df[df['timestamp'] > cutoff_date]
            
            # Save recent logs only
            recent_logs.to_csv(self.logs_file, index=False)
            
            # Remove old annotated images
            kept_images = set(recent_logs['image_path'].unique())
            for img_path in self.annotated_images_dir.glob('*'):
                if str(img_path) not in kept_images:
                    img_path.unlink()

    def get_logs_summary(self) -> Dict:
        try:
            df = pd.read_csv(self.logs_file)
            return {
                'total_predictions': len(df),
                'unique_images': len(df['filename'].unique()),
                'classes': df['class_name'].unique().tolist(),
                'date_range': {
                    'first': df['timestamp'].min(),
                    'last': df['timestamp'].max()
                },
                'storage': {
                    'log_size': os.path.getsize(self.logs_file),
                    'images_count': len(list(self.annotated_images_dir.glob('*')))
                }
            }
        except (pd.errors.EmptyDataError, FileNotFoundError):
            return {
                'total_predictions': 0,
                'unique_images': 0,
                'classes': [],
                'date_range': {'first': None, 'last': None},
                'storage': {
                    'log_size': 0,
                    'images_count': 0
                }
            }
