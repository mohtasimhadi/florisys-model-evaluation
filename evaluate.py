from ultralytics import YOLO

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