from ultralytics import YOLO
from pathlib import Path
import torch
import yaml

class PotholeTrainer:
    def __init__(self):
        self.data_dir = Path(r"C:\Users\ICYZO\Downloads\Pothole Detection\yolo_data")
        self.model_dir = Path("models/trained_models")
        self.device = "0" if torch.cuda.is_available() else "cpu"
        
    def verify_dataset(self):
        """Verify dataset meets minimum requirements"""
        print("🔍 Verifying dataset quality...")
        
        # Check training data
        train_images = list((self.data_dir / "images" / "train").glob("*"))
        train_labels = list((self.data_dir / "labels" / "train").glob("*.txt"))
        
        # Check validation data
        val_images = list((self.data_dir / "images" / "val").glob("*"))
        val_labels = list((self.data_dir / "labels" / "val").glob("*.txt"))
        
        print(f"📊 Dataset Statistics:")
        print(f"   Training images: {len(train_images)}")
        print(f"   Training labels: {len(train_labels)}")
        print(f"   Validation images: {len(val_images)}")
        print(f"   Validation labels: {len(val_labels)}")
        
        # Check if meets minimum requirements
        if len(train_images) < 100:
            print(f"❌ INSUFFICIENT DATA: Need 100+ images, have {len(train_images)}")
            return False
        
        # Check annotation quality
        annotation_counts = []
        for lbl_file in (self.data_dir / "labels" / "train").glob("*.txt"):
            with open(lbl_file, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
                annotation_counts.append(len(lines))
        
        if annotation_counts:
            avg_annotations = sum(annotation_counts) / len(annotation_counts)
            print(f"Average annotations per image: {avg_annotations:.2f}")
            
            if avg_annotations < 1.5:
                print("❌ WARNING: Low annotation density detected!")
                print("   Consider adding more annotations to each image")
        
        return True
    
    def train_model(self, model_size="m"):
        """Train YOLOv8 model with specified size"""
        if not self.verify_dataset():
            print("❌ Cannot proceed with training - dataset quality insufficient")
            return None
        
        print("🚀 Starting YOLOv8 training...")
        
        # Model selection based on size
        model_map = {
            "n": "yolov8n.pt",  # Nano
            "s": "yolov8s.pt",  # Small
            "m": "yolov8m.pt",  # Medium
            "l": "yolov8l.pt",  # Large
            "x": "yolov8x.pt",  # XLarge
        }
        
        model_name = model_map.get(model_size, "yolov8s.pt")
        
        # Load model
        model = YOLO(model_name)
        
        # Training parameters
        results = model.train(
            data=str(self.data_dir / "data.yaml"),
            epochs=1,
            imgsz=416,
            batch=4,
            patience=20,
            device=self.device,
            project=str(self.model_dir),
            name=f"pothole_detection_{model_size}",
            exist_ok=True,
            verbose=True,
            lr0=0.01,
            lrf=0.1,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=2.0,
            hsv_h=0.015,
            hsv_s=0.5,
            hsv_v=0.4,
            translate=0.05,
            scale=0.2,
            fliplr=0.5,
        )
        
        return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train YOLOv8 model for pothole detection')
    parser.add_argument('--size', type=str, default='m', 
                        help='Model size: n (nano), s (small), m (medium), l (large), x (xlarge)')
    
    args = parser.parse_args()
    
    trainer = PotholeTrainer()
    trainer.train_model(args.size)