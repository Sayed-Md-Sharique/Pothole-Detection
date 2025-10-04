from ultralytics import YOLO
from pathlib import Path
import torch

class ModelValidator:
    def __init__(self):
        self.data_dir = Path(r"C:\Users\ICYZO\Downloads\Pothole Detection\yolo_data2")
        self.model_dir = Path("models/trained_models")
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    def validate_model(self, model_path=None):
        """Validate a trained YOLO model"""
        if model_path is None:
            # Find the latest trained model
            model_files = sorted(
                self.model_dir.glob("**/weights/best.pt"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            if not model_files:
                print("❌ No trained model found. Please train first.")
                return None
            
            model_path = model_files[0]
            print(f"📂 Using latest model: {model_path}")
        
        # Load model
        model = YOLO(model_path)
        
        print("\n📊 Evaluating model performance...\n")
        results = model.val(
            data=str(self.data_dir / "data.yaml"),
            split="val",
            conf=0.001,
            iou=0.6,
            device=self.device,
            save_json=True  # saves COCO-style metrics
        )
        
        # Extract metrics
        metrics = results.results_dict
        
        precision = metrics['metrics/precision(B)']
        recall = metrics['metrics/recall(B)']
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        
        print("="*60)
        print("🎯 MODEL PERFORMANCE REPORT")
        print("="*60)
        print(f"Precision     : {precision:.2%}")
        precision = metrics['metrics/precision(B)']
        recall = metrics['metrics/recall(B)']

        # Approximate confusion counts
        tp = recall  # normalized value
        # Can't get absolute counts, so instead approximate detection accuracy
        detection_accuracy = (2 * precision * recall) / (precision + recall + 1e-6)

        print(f"Detection Accuracy (approx): {detection_accuracy:.2%}")

        print(f"Recall        : {recall:.2%}")
        print(f"F1-Score      : {f1:.2%}")
        print(f"mAP@0.5       : {metrics['metrics/mAP50(B)']:.2%}")
        print(f"mAP@0.5–0.95  : {metrics['metrics/mAP50-95(B)']:.2%}")
        print("-"*60)
        print(f"Test Images   : {metrics.get('val/seen', 'N/A')}")
        print(f"Inference time: {metrics.get('speed/inference', 0):.2f} ms per image")
        print(f"NMS time      : {metrics.get('speed/nms', 0):.2f} ms per image")
        print(f"Total time    : {metrics.get('speed/total', 0):.2f} ms per image")
        print("="*60)
        
        # Performance assessment
        if metrics['metrics/mAP50(B)'] >= 0.85:
            print("✅ EXCELLENT: High accuracy achieved!")
        elif metrics['metrics/mAP50(B)'] >= 0.7:
            print("✅ GOOD: Reasonable accuracy")
        elif metrics['metrics/mAP50(B)'] >= 0.5:
            print("⚠️  FAIR: Needs improvement")
        else:
            print("❌ POOR: Model needs more training/data")
                
        return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate YOLOv8 model")
    parser.add_argument("--model", type=str, help="Path to model weights (optional)")
    
    args = parser.parse_args()
    
    validator = ModelValidator()
    validator.validate_model(args.model)
