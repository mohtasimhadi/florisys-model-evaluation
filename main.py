import argparse
from train import train
from evaluate import evaluate

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