import shutil
import random
from pathlib import Path
import yaml
from collections import Counter

class DatasetPreparer:
    def __init__(self):
        self.base_dir = Path(r"C:\Users\ICYZO\Downloads\Pothole Detection")
        self.raw_data_dir = self.base_dir / "Anamoly.v1i.yolov8"
        self.yolo_data_dir = self.base_dir / "yolo_data"
        
    def analyze_dataset(self):
        """Analyze dataset quality and statistics"""
        print("🔍 Analyzing dataset quality...")
        
        stats = {
            'train': {'images': 0, 'labels': 0, 'annotations': 0},
            'val': {'images': 0, 'labels': 0, 'annotations': 0},
            'test': {'images': 0, 'labels': 0, 'annotations': 0}
        }
        
        annotation_counts = []
        
        for split in ['train', 'val', 'test']:
            # Count images
            img_dir = self.raw_data_dir / split / "images"
            if img_dir.exists():
                stats[split]['images'] = len(list(img_dir.glob("*")))
            
            # Count labels and annotations
            lbl_dir = self.raw_data_dir / split / "labels"
            if lbl_dir.exists():
                stats[split]['labels'] = len(list(lbl_dir.glob("*.txt")))
                
                for lbl_file in lbl_dir.glob("*.txt"):
                    with open(lbl_file, 'r') as f:
                        lines = [line.strip() for line in f.readlines() if line.strip()]
                        stats[split]['annotations'] += len(lines)
                        if split == 'train':  # Only track training for average
                            annotation_counts.append(len(lines))
        
        # Print statistics
        print("\n📊 DATASET STATISTICS:")
        for split in stats:
            imgs = stats[split]['images']
            lbls = stats[split]['labels']
            anns = stats[split]['annotations']
            avg_ann = anns / imgs if imgs > 0 else 0
            
            print(f"   {split.upper():6}: {imgs:4} images, {lbls:4} labels, {anns:4} annotations")
            print(f"          Avg annotations per image: {avg_ann:.2f}")
        
        # Check if validation set needs balancing
        if stats['val']['images'] < 100 and stats['train']['images'] > 500:
            print("\n⚠️  Validation set too small. Balancing dataset...")
            self.balance_dataset()
            
        return stats, annotation_counts
    
    def balance_dataset(self):
        """Balance the dataset by moving some training data to validation"""
        train_img_dir = self.raw_data_dir / "train" / "images"
        train_lbl_dir = self.raw_data_dir / "train" / "labels"
        val_img_dir = self.raw_data_dir / "val" / "images"
        val_lbl_dir = self.raw_data_dir / "val" / "labels"
        
        # Create directories if they don't exist
        val_img_dir.mkdir(parents=True, exist_ok=True)
        val_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        # Get training images
        train_images = list(train_img_dir.glob("*"))
        if not train_images:
            print("❌ No training images found!")
            return False
        
        # Move 15% of training data to validation
        move_count = max(100, int(len(train_images) * 0.15))
        images_to_move = random.sample(train_images, min(move_count, len(train_images)))
        
        moved_count = 0
        for img_path in images_to_move:
            try:
                # Move image
                dest_img = val_img_dir / img_path.name
                shutil.move(str(img_path), str(dest_img))
                
                # Move corresponding label
                label_path = train_lbl_dir / f"{img_path.stem}.txt"
                if label_path.exists():
                    dest_label = val_lbl_dir / label_path.name
                    shutil.move(str(label_path), str(dest_label))
                    moved_count += 1
                    
            except Exception as e:
                print(f"Warning: Could not move {img_path.name}: {e}")
        
        print(f"✅ Moved {moved_count} images to validation set")
        return True
    
    def convert_to_yolo_format(self):
        """Convert dataset to YOLO format"""
        print("Converting dataset to YOLO format...")
        
        # Create YOLO directory structure
        splits = ['train', 'val', 'test']
        for split in splits:
            (self.yolo_data_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (self.yolo_data_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        # Copy images and labels to YOLO format
        for split in splits:
            src_images = self.raw_data_dir / split / "images"
            src_labels = self.raw_data_dir / split / "labels"
            
            if src_images.exists():
                for img_file in src_images.glob("*"):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                        dest = self.yolo_data_dir / 'images' / split / img_file.name
                        shutil.copy2(img_file, dest)
            
            if src_labels.exists():
                for lbl_file in src_labels.glob("*.txt"):
                    dest = self.yolo_data_dir / 'labels' / split / lbl_file.name
                    shutil.copy2(lbl_file, dest)
        
        print("Dataset converted to YOLO format!")
    
    def create_data_yaml(self):
        """Create data.yaml configuration file"""
        yaml_content = {
            'path': str(self.yolo_data_dir.resolve()),
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'nc': 1,
            'names': ['pothole']
        }
        
        with open(self.yolo_data_dir / "data.yaml", 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        print("Created data.yaml configuration file")
    
    def verify_dataset(self):
        """Verify the dataset structure"""
        print("Verifying dataset...")
        
        for split in ['train', 'val', 'test']:
            images_dir = self.yolo_data_dir / 'images' / split
            labels_dir = self.yolo_data_dir / 'labels' / split
            
            images = list(images_dir.glob("*"))
            labels = list(labels_dir.glob("*.txt"))
            
            print(f"{split}: {len(images)} images, {len(labels)} labels")
            
            # Check for matching image-label pairs
            image_stems = {img.stem for img in images}
            label_stems = {lbl.stem for lbl in labels}
            
            missing_labels = image_stems - label_stems
            missing_images = label_stems - image_stems
            
            if missing_labels:
                print(f"  ⚠️  {len(missing_labels)} images missing labels")
            if missing_images:
                print(f"  ⚠️  {len(missing_images)} labels missing images")
    
    def run(self):
        """Run full preparation process"""
        print("Starting dataset preparation...")
        self.analyze_dataset()
        self.convert_to_yolo_format()
        self.create_data_yaml()
        self.verify_dataset()
        print("Dataset preparation completed!")

if __name__ == "__main__":
    preparer = DatasetPreparer()
    preparer.run()