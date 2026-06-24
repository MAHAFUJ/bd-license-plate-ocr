import os
import cv2
import gradio as gr
import easyocr
import yaml
from ultralytics import YOLO

with open("configs/config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Find the most trained model available
model_path = "yolov8m.pt"
for run in ["run_3", "run_2", "run_1"]:
    path = f"models/{run}/weights/best.pt"
    if os.path.exists(path):
        model_path = path
        break

detector = YOLO(model_path)
reader = easyocr.Reader(cfg['ocr']['languages'], gpu=False)

def process_image(img):
    if img is None: return None, "No image provided"
    
    results = detector(img, verbose=False)
    output_log = ""
    out_img = img.copy()
    
    # FOOLPROOF FIX: Pull raw tensors out immediately to avoid YOLO indexing bugs
    boxes_tensor = results[0].boxes.xyxy.cpu().numpy().astype(int)
    confs_tensor = results[0].boxes.conf.cpu().numpy()
    
    for i in range(len(boxes_tensor)):
        xyxy = boxes_tensor[i]
        conf = float(confs_tensor[i])
        
        crop = img[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
        h, w = crop.shape[:2]
        
        # Safety check if the model predicts an invalid tiny box
        if h == 0 or w == 0:
            continue
            
        crop = cv2.resize(crop, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
        crop = cv2.threshold(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        ocr_res = reader.readtext(crop)
        text = " ".join([r[1] for r in ocr_res])
        
        cv2.rectangle(out_img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 3)
        cv2.putText(out_img, text, (xyxy[0], max(xyxy[1]-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3)
        
        output_log += f"Plate {i+1}:\n- Text: {text}\n- YOLO Confidence: {conf:.2%}\n\n"
        
    return cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB), output_log

demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="numpy"),
    outputs=[gr.Image(label="Processed Image"), gr.Textbox(label="Extracted Data")],
    title="Bangladeshi ANPR System (M4 Mac Accelerated)"
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)