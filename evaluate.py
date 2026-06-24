import os
import cv2
import yaml
import easyocr
import pandas as pd
import numpy as np
from ultralytics import YOLO
from jiwer import cer

def enhance_plate(img, cfg):
    h, w = img.shape[:2]
    img = cv2.resize(img, (w * cfg['ocr']['upscale_factor'], h * cfg['ocr']['upscale_factor']), interpolation=cv2.INTER_CUBIC)
    if cfg['ocr']['apply_grayscale']:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if cfg['ocr']['apply_thresholding']:
        img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return img

def main():
    with open("configs/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
        
    print("\n--- PHASE 1: YOLO DETECTION METRICS ---")
    best_model = "models/run_3/weights/best.pt" # Defaults to highest epoch run
    if not os.path.exists(best_model):
        best_model = "models/run_1/weights/best.pt"
        
    detector = YOLO(best_model)
    metrics = detector.val(data=os.path.abspath(f"{cfg['data']['local_data_dir']}/data.yaml"))
    
    print(f"mAP50:     {metrics.results_dict['metrics/mAP50(B)']:.4f}")
    print(f"Precision: {metrics.results_dict['metrics/precision(B)']:.4f}")
    print(f"Recall:    {metrics.results_dict['metrics/recall(B)']:.4f}")
    
    print("\n--- PHASE 2: OCR RECOGNITION METRICS ---")
    if not os.path.exists(cfg['data']['ocr_eval_csv']):
        print("[!] Evaluation CSV missing. Please run src/data_download.py and fill out the CSV first.")
        return
        
    df = pd.read_csv(cfg['data']['ocr_eval_csv']).dropna()
    if len(df) == 0:
        print("[!] No ground truth text found in CSV. Please open data/ocr_eval_subset.csv and add actual license plate text.")
        return
        
    reader = easyocr.Reader(cfg['ocr']['languages'], gpu=False) # CPU fallback for Mac
    cer_scores = []
    
    for _, row in df.iterrows():
        img = cv2.imread(row['image_path'])
        if img is None: continue
        
        results = detector(img, verbose=False)
        
        # FOOLPROOF FIX: Check the length of the tensor directly, bypassing the object buggy index
        if len(results[0].boxes.xyxy) > 0:
            box = results[0].boxes.xyxy[0].cpu().numpy().astype(int)
            crop = enhance_plate(img[box[1]:box[3], box[0]:box[2]], cfg)
            ocr_res = reader.readtext(crop)
            pred_text = " ".join([res[1] for res in ocr_res]).strip()
            
            true_text = str(row['ground_truth_plate']).strip()
            cer_scores.append(cer(true_text, pred_text))

    mean_cer = np.mean(cer_scores) if cer_scores else 1.0
    print(f"Character Error Rate (CER): {mean_cer:.4f}")
    print(f"Character Accuracy:         {(1.0 - mean_cer)*100:.2f}%")

if __name__ == "__main__":
    main()