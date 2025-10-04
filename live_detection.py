import cv2
import torch
import numpy as np
from pathlib import Path
import time
from datetime import datetime
from ultralytics import YOLO
import json


class LivePotholeDetector:
    def __init__(self, model_path=None):
        # Force CPU since GT 730 is not CUDA-supported
        self.device = "cpu"

        # Find latest trained model if none provided
        if model_path is None:
            model_dir = Path("models/trained_models")
            model_files = list(model_dir.glob("**/weights/best.pt"))
            if not model_files:
                print("❌ No trained model found. Please train first.")
                exit()
            model_path = max(model_files, key=lambda f: f.stat().st_mtime)

        print(f"📂 Loading model: {model_path}")
        self.model = YOLO(model_path)
        self.model.to(self.device)
        print(f"✅ Model loaded on {self.device.upper()}")

    def detect(self, frame, conf_threshold=0.25):
        """Run detection on a single frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.model.predict(
            rgb_frame, conf=conf_threshold, imgsz=640, verbose=False, device=self.device
        )

        detections = []
        if results and results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls[0])  # class index
                class_name = self.model.names[class_id]  # class label

                detections.append({
                    'bbox': box.xyxy[0].cpu().numpy().astype(int).tolist(),
                    'confidence': float(box.conf[0]),
                    'class_id': class_id,
                    'class': class_name
                })

        if detections:
            print(f"Detections: {detections}")
        return detections

    def draw_detections(self, frame, detections):
        """Draw detections on frame"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['confidence']
            label = f"{det['class']} {conf:.2f}"

            # Color based on confidence
            if conf > 0.7:
                color = (0, 0, 255)  # Red
            elif conf > 0.4:
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 255, 0)  # Green

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)

            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return frame

    def run_webcam(self, camera_index=0):
        """Run live detection from webcam"""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"❌ Cannot open webcam (index {camera_index})")
            return

        print("🎥 Live detection started! Press 'q' to quit, 's' to save screenshot, +/- to adjust confidence")

        conf_threshold = 0.25
        frame_count = 0
        start_time = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize for faster processing
                frame = cv2.resize(frame, (640, 480))

                # Detect
                detections = self.detect(frame, conf_threshold)

                # Draw detections
                processed_frame = self.draw_detections(frame.copy(), detections)

                # FPS calculation
                frame_count += 1
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0

                # Overlay info
                info_text = [
                    f"FPS: {fps:.1f}",
                    f"Detections: {len(detections)}",
                    f"Conf: {conf_threshold:.2f}",
                    f"Device: CPU"
                ]
                for i, text in enumerate(info_text):
                    cv2.putText(processed_frame, text, (10, 30 + i * 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Show frame
                cv2.imshow('Pothole Detection', processed_frame)

                # Key press handling
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    self.save_screenshot(frame, detections)
                elif key == ord('+'):
                    conf_threshold = min(conf_threshold + 0.05, 0.95)
                    print(f"Confidence: {conf_threshold:.2f}")
                elif key == ord('-'):
                    conf_threshold = max(conf_threshold - 0.05, 0.05)
                    print(f"Confidence: {conf_threshold:.2f}")

        except KeyboardInterrupt:
            print("\n🛑 Stopped by user")

        cap.release()
        cv2.destroyAllWindows()

    def save_screenshot(self, frame, detections):
        """Save current frame and detections (image + JSON)"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_filename = f"detection_{timestamp}.jpg"
        json_filename = f"detection_{timestamp}.json"

        # Save annotated image
        result_frame = self.draw_detections(frame.copy(), detections)
        cv2.imwrite(img_filename, result_frame)

        # Save detections as JSON
        with open(json_filename, "w") as f:
            json.dump(detections, f, indent=4)

        print(f"💾 Screenshot saved: {img_filename}")
        print(f"💾 Detections saved: {json_filename}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run live road hazard detection')
    parser.add_argument('--model', type=str, help='Path to model weights')
    parser.add_argument('--camera', type=int, default=0, help='Webcam index (default=0)')

    args = parser.parse_args()

    detector = LivePotholeDetector(args.model)
    detector.run_webcam(camera_index=args.camera)
