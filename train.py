import os
import yaml
import argparse
import mlflow
import torch
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', type=str, required=True, choices=['run_1', 'run_2', 'run_3'])
    return parser.parse_args()

def main():
    args = parse_args()
    with open("configs/config.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
        
    # Check Apple Silicon MPS Availability
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"[*] Hardware Acceleration Engine: {device.upper()}")
    
    epochs = cfg['yolo']['runs'][args.run_id]['epochs']
    
    # FOOLPROOF FIX: Force MLflow to save logs offline to a local directory
    # This stops it from trying to connect to a crashed web server on port 5000
    local_tracking_path = f"file://{os.path.abspath('mlruns')}"
    mlflow.set_tracking_uri(local_tracking_path)
    mlflow.set_experiment(cfg['project_name'])
    
    with mlflow.start_run(run_name=f"YOLOv8m_{epochs}_Epochs"):
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("device", device)
        mlflow.log_param("optimizer", cfg['yolo']['optimizer'])
        
        model = YOLO(cfg['yolo']['model_type'])
        
        results = model.train(
            data=os.path.abspath(f"{cfg['data']['local_data_dir']}/data.yaml"),
            epochs=epochs,
            imgsz=cfg['yolo']['image_size'],
            batch=cfg['yolo']['batch_size'],
            device=device,
            project="models",
            name=args.run_id
        )
        
        best_weight = f"models/{args.run_id}/weights/best.pt"
        if os.path.exists(best_weight):
            mlflow.log_artifact(best_weight, artifact_path="model_weights")
            
        print(f"[+] Training run {args.run_id} complete. Logs saved locally to mlruns/ folder.")

if __name__ == "__main__":
    main()