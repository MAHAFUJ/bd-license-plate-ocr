import os
import cv2
import yaml
import easyocr
from ultralytics import YOLO

class ANPRPipeline:
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)
            
        best_model = "models/run_2/weights/best.pt"
        if not os.path.exists(best_model):
            best_model = "models/run_1/weights/best.pt"
            
        self.detector = YOLO(best_model)
        self.reader = easyocr.Reader(self.cfg['ocr_settings']['languages'], gpu=True)
        
    def process_image(self, img_path):
        img = cv2.imread(img_path)
        if img is None:
            return None, []
            
        results = self.detector(img, verbose=False)
        outputs = []
        annotated_img = img.copy()
        
        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            
            crop = img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
            
            # Applied operational preprocessing pipeline
            prep = self.cfg['ocr_settings']['preprocessing']
            h, w = crop.shape[:2]
            crop_res = cv2.resize(crop, (w * prep['upscale_factor'], h * prep['upscale_factor']), interpolation=cv2.INTER_CUBIC)
            crop_gray = cv2.cvtColor(crop_res, cv2.COLOR_BGR2GRAY)
            crop_thresh = cv2.threshold(crop_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            
            ocr_res = self.reader.readtext(crop_thresh)
            plate_text = " ".join([res[1] for res in ocr_res])
            
            outputs.append({
                "box": xyxy.tolist(),
                "confidence": conf,
                "text": plate_text
            })
            
            # Standard annotation drawing
            cv2.rectangle(annotated_img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 3)
            cv2.putText(annotated_img, plate_text, (xyxy[0], max(xyxy[1] - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
        return annotated_img, outputs

if __name__ == "__main__":
    pipeline = ANPRPipeline()
    # Test sample target verification
    sample_img = "data/images/test/test_0.jpg"
    if os.path.exists(sample_img):
        res_img, data = pipeline.process_image(sample_img)
        print(f"[+] Operational Check Verification Output Data: {data}")