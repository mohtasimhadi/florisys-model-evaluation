import argparse
from ultralytics import YOLO

def train(model_dir, data_dir):
    model = YOLO(model_dir)
    
    print(f"Starting training with model: {model_dir} and dataset: {data_dir}")
    model.train(data=data_dir, epochs=50, batch=16, imgsz=640)
