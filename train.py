import argparse
from ultralytics import YOLO

def train(model_dir, data_dir):
    model = YOLO(model_dir)
    
    print(f"Starting training with model: {model_dir} and dataset: {data_dir}")
    model.train(data=data_dir, epochs=50, batch=16, imgsz=640)

def evaluate(model_dir, data_dir):
    model = YOLO(model_dir)
    
    print(f"Starting evaluation with model: {model_dir} and dataset: {data_dir}")
    result = model.val(data=data_dir)
    
    print("Evaluation Results:")
    print(f"mAP@0.5: {result.maps[0]}")
    print(f"mAP@0.5:0.95: {result.maps[1]}")
    print(f"Precision: {result.results_dict()['precision']}")
    print(f"Recall: {result.results_dict()['recall']}")
    print(f"F1-Score: {result.results_dict()['f1']}")
    print(f"IDF1: {result.results_dict()['IDF1']}")

def main():
    parser = argparse.ArgumentParser(description="Train or Evaluate YOLOv8 Model on Custom Dataset")
    
    parser.add_argument('--model-dir', type=str, required=True, help='Directory of the pre-trained or custom YOLO model')
    parser.add_argument('--data-dir', type=str, required=True, help='Directory of the dataset')
    parser.add_argument('--mode', choices=['train', 'evaluate'], required=True, help='Mode: train or evaluate')
    
    args = parser.parse_args()

    if args.mode == 'train':
        train(args.model_dir, args.data_dir)
    elif args.mode == 'evaluate':
        evaluate(args.model_dir, args.data_dir)

if __name__ == "__main__":
    main()
