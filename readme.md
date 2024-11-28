# YOLOv8 Custom Training and Evaluation

This repository provides a script to train or evaluate a YOLOv8 model on your custom dataset. You can easily choose whether to train the model from scratch or evaluate an existing model using command-line arguments. The script uses the [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) library.

## Prerequisites

Before using the script, ensure you have the following dependencies installed:

- Python 3.7 or higher
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) library
- `torch` (PyTorch) for model training and inference
- A custom dataset formatted for YOLO (with a `data.yaml` file)
- A GPU is recommended for faster training and evaluation, although the script will work on a CPU (with slower performance).

You can install the necessary dependencies by running:

```bash
pip install ultralytics torch
```

## Directory Structure

Ensure your dataset and model files are organized as follows:

```
project/
├── train_or_evaluate.py    # Python script for training or evaluating
├── model/                  # Directory containing your pre-trained YOLO model (e.g., yolov8n.pt)
├── datasets/
│   └── OrnamentalPlants/   # Example dataset folder
│       ├── images/         # Images for training/validation
│       ├── labels/         # Annotations in YOLO format
│       └── data.yaml       # Dataset configuration file
└── README.md               # This README file
```

## Usage

The script allows you to either **train** a model from scratch or **evaluate** an existing model on your custom dataset. You can specify the action, model, and dataset using command-line arguments.

### Arguments

- `--model-dir`: Path to your pre-trained YOLO model or a custom model. Example: `model/yolov8n.pt`.
- `--data-dir`: Path to your dataset directory that contains the `data.yaml` file. Example: `datasets/OrnamentalPlants/data.yaml`.
- `--mode`: Mode to run the script in. You can either choose `train` to train the model or `evaluate` to evaluate an existing model.

### Example Commands

#### 1. **Training the Model**

To train the model on your custom dataset:

```bash
python train_or_evaluate.py --model-dir 'model/yolov8n.pt' --data-dir 'datasets/OrnamentalPlants/data.yaml' --mode train
```

This command will start training a model using the pre-trained `yolov8n.pt` weights and the dataset specified in the `data.yaml` file.

#### 2. **Evaluating the Model**

To evaluate an existing model on your custom dataset:

```bash
python train_or_evaluate.py --model-dir 'model/yolov8n.pt' --data-dir 'datasets/OrnamentalPlants/data.yaml' --mode evaluate
```

This command will evaluate the model using the dataset and print metrics like mAP, Precision, Recall, and F1-Score.

## Dataset Format

Your dataset should be in the YOLO format, which requires the following structure:

```
datasets/
├── images/
│   ├── train/                # Training images
│   ├── val/                  # Validation images
│   └── test/                 # Test images (optional)
├── labels/
│   ├── train/                # Annotations for training images
│   ├── val/                  # Annotations for validation images
│   └── test/                 # Annotations for test images (optional)
└── data.yaml                 # Dataset configuration file
```

The `data.yaml` file should contain the paths to the `train`, `val`, and optionally `test` images and their corresponding labels, along with the number of classes and class names. Here’s an example `data.yaml` file:

```yaml
train: /path/to/train/images
val: /path/to/val/images
test: /path/to/test/images   # Optional

nc: 3                       # Number of classes
names: ['class1', 'class2', 'class3']  # List of class names
```

### Example Label Format

Each annotation file (located in the `labels/` directory) should have the same name as the corresponding image file and contain one row per object in the image. Each row should have the following format:

```
<class_id> <x_center> <y_center> <width> <height>
```

Where:
- `<class_id>`: The class index (0-based).
- `<x_center> <y_center>`: The center of the bounding box, normalized between 0 and 1.
- `<width> <height>`: The width and height of the bounding box, normalized between 0 and 1.

## Training Parameters

The `train` function in the script uses the following default parameters:
- **Epochs**: 50
- **Batch size**: 16
- **Image size**: 640 (can be adjusted)

You can modify these parameters in the `train()` function as needed, or pass additional arguments to the script to configure them.

## Evaluation Metrics

During evaluation, the following metrics are displayed:

- **mAP@0.5**: Mean Average Precision at IoU threshold of 0.5.
- **mAP@0.5:0.95**: Mean Average Precision at multiple IoU thresholds (0.5 to 0.95).
- **Precision**: Proportion of true positive predictions.
- **Recall**: Proportion of true positives among actual objects.
- **F1-Score**: Harmonic mean of Precision and Recall.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

This README provides detailed instructions for using the script for training or evaluation, and it explains the expected dataset structure and command-line arguments. Feel free to modify the text as needed!