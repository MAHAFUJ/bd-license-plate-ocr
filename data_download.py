import os
import yaml
import zipfile
import shutil
import pandas as pd
from datasets import load_dataset

def load_config():
    with open("configs/config.yaml", "r") as f:
        return yaml.safe_load(f)

def setup_yolo_dataset(config):
    print("\n[+] Downloading YOLO Plate Detection Dataset (Boxes) from Hugging Face...")
    print("    (Note: This is required because YOLO needs X/Y coordinate labels to train)")
    
    data_dir = config['data']['local_data_dir']
    os.makedirs(data_dir, exist_ok=True)
    
    dataset = load_dataset(config['data']['hf_dataset_url'], name=config['data']['hf_version'])
    
    for split in ['train', 'validation', 'test']:
        yolo_split = 'val' if split == 'validation' else split
        os.makedirs(f"{data_dir}/images/{yolo_split}", exist_ok=True)
        os.makedirs(f"{data_dir}/labels/{yolo_split}", exist_ok=True)
        
        print(f"[-] Processing {split} split records...")
        for idx, item in enumerate(dataset[split]):
            img = item['image']
            img.save(f"{data_dir}/images/{yolo_split}/{split}_{idx}.jpg")
            
            with open(f"{data_dir}/labels/{yolo_split}/{split}_{idx}.txt", "w") as lf:
                objects = item['objects']
                
                # Handle Hugging Face's dict-of-lists format
                if isinstance(objects, dict) and 'bbox' in objects:
                    for bbox in objects['bbox']:
                        # Dataset format is usually [x_min, y_min, width, height]
                        x_min, y_min, bw, bh = bbox
                        w, h = img.size
                        # Convert to YOLO format: center_x, center_y, width, height (normalized)
                        x_c = (x_min + bw / 2.0) / w
                        y_c = (y_min + bh / 2.0) / h
                        bw_norm = bw / w
                        bh_norm = bh / h
                        lf.write(f"0 {x_c:.6f} {y_c:.6f} {bw_norm:.6f} {bh_norm:.6f}\n")
                        
    yaml_content = {
        'path': os.path.abspath(data_dir), 
        'train': 'images/train', 
        'val': 'images/val', 
        'test': 'images/test', 
        'names': {0: 'plate'}
    }
    with open(f"{data_dir}/data.yaml", 'w') as f:
        yaml.dump(yaml_content, f)
    print("[+] YOLO Dataset Ready.")

def process_mendeley_dataset(config):
    print("\n=======================================================")
    print("STEP: Process Your Mendeley Dataset (For OCR & UI Testing)")
    print("Please download your dataset from Mendeley: https://data.mendeley.com/datasets/nntnnzffw4/3")
    zip_path = input("Enter the full path to your downloaded dataset ZIP file (or drag and drop it here):\n> ").strip().strip("'").strip('"')
    
    extract_path = os.path.join(config['data']['local_data_dir'], "mendeley_data")
    
    if os.path.exists(zip_path):
        print(f"[-] Extracting Mendeley dataset on Apple Silicon...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
            
        # Create OCR Eval template
        images_found = []
        for root, _, files in os.walk(extract_path):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    images_found.append(os.path.join(root, file))
                    if len(images_found) >= 50: break # Grab 50 samples
            if len(images_found) >= 50: break
            
        df = pd.DataFrame({'image_path': images_found, 'ground_truth_plate': ['' for _ in images_found]})
        df.to_csv(config['data']['ocr_eval_csv'], index=False)
        print(f"\n[!] ACTION REQUIRED: Open '{config['data']['ocr_eval_csv']}' in VS Code and fill in the 'ground_truth_plate' column for evaluation.")
    else:
        print("[!] File not found. Make sure you typed the path correctly.")

if __name__ == "__main__":
    cfg = load_config()
    setup_yolo_dataset(cfg)
    process_mendeley_dataset(cfg)