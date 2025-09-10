from ultralytics import YOLO
import yaml
import os

model = YOLO("weights/best.pt")

dataset_yaml = {
    'path': os.path.abspath('../validation-dataset'),  
    'val': 'images',  
    'nc': 1,  
    'names': ['T-Bank logo'] 
}

yaml_path = 'validate_config.yaml'
with open(yaml_path, 'w') as f:
    yaml.dump(dataset_yaml, f, default_flow_style=False)

metrics = model.val(
    data=yaml_path,
    imgsz=640,
    batch=16,
    conf=0.4,
    iou=0.5
)

print(f"Precision: {metrics.results_dict['metrics/precision(B)']:.3f}")
print(f"Recall: {metrics.results_dict['metrics/recall(B)']:.3f}")
print(f"F1-score: {2 * metrics.results_dict['metrics/precision(B)'] * metrics.results_dict['metrics/recall(B)'] / (metrics.results_dict['metrics/precision(B)'] + metrics.results_dict['metrics/recall(B)'] + 1e-10):.3f}")

