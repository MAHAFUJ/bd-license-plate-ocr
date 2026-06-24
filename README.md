Bangladeshi Vehicle Number Plate Detection and Recognition 

Author: MD. Mahafuj Hossain (ID: 17201125)

Institution: BRAC University (Advanced Training on Semiconductor and ICT Technology)

📖 Project Overview

Selected Domain: Computer Vision (Object Detection & Optical Character Recognition).

Problem Statement: Manually reading and logging vehicle number plates is slow and error-prone. In Bangladesh, this is further complicated because number plates feature a mix of both Bengali and English scripts under highly variable lighting and environmental conditions.

Expected Output: An automated end-to-end pipeline. Given a raw photograph of a vehicle, the system will output the localized bounding box of the number plate and the transcribed alphanumeric text (in Bengali and English) along with confidence scores.

🗄️ Dataset

This project utilizes two datasets to achieve distinct goals:

Mendeley Vehicle License Plate Detection Dataset (Hugging Face): Provides annotated bounding boxes for the license_plate class. Why it fits: It contains genuine boxes required to train the YOLOv8m object detector accurately.

Custom Bangladeshi Vehicle Dataset (Google Drive): ~2,950 real-world Bangladeshi vehicle photographs across five classes (Private cars, Trucks, Buses, Bikes, CNGs). Why it fits: Provides raw, unannotated real-world data to evaluate the OCR pipeline and demonstrate the app in production conditions.

Data Fetching

Data is managed dynamically and never committed to version control. To automatically download both datasets into the local data/ folder, run:

python src/data_download.py


⚙️ Setup & Installation

Prerequisites: Python 3.10+

Clone the repository and navigate to the project root.

Create a virtual environment (optional but recommended):

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate


Install dependencies:

pip install -r requirements.txt


🧠 Training

The model uses transfer learning, fine-tuning pretrained YOLOv8m weights on the license plate dataset.

To train the model, run:

python src/train.py


Important Hyperparameters:

Epochs: Tested with 50 and 100 (Early stopping patience = 20).

Batch Size: 16 (for 50 epochs) and 8 (for 100 epochs).

Image Size (imgsz): 640x640 (matching native photograph scales).

Optimizer: Auto-selected (SGD/AdamW) with an initial learning rate of 0.01.

🔬 Course Techniques Implemented

This project heavily incorporates modules from the SICIP curriculum:

Convolutional Neural Networks (CNN): YOLOv8m for object detection.

Transfer Learning & Fine-Tuning: Leveraging COCO-pretrained weights.

Optical Character Recognition (OCR): EasyOCR with mixed Bengali/English script processing.

Hyperparameter Optimization: Comparing multiple training configurations (epochs, batch sizes).

MLflow: Comprehensive experiment tracking and metric logging.

UI Model Serving: Interactive web interface built with Gradio.

Docker: Full containerization of the ML environment for cross-platform reproducibility.

📊 MLflow Tracking

MLflow is used to track every training experiment, logging hyperparameters (epochs, batch size) and metrics (mAP, precision, recall).

To start the MLflow UI locally:
(If not using Docker, you can run this manually)

mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri ./mlruns


Navigate to http://localhost:5000 to view the dashboard.

MLflow Training Runs:


Explanation: The UI tracks multiple runs (e.g., yolov8m_e50 and yolov8m_e100). It logs validation mAP50-95, precision, and recall per epoch, allowing us to select the best best.pt weights automatically.

📈 Evaluation

The pipeline was evaluated on a held-out test split for detection, and a hand-labeled subset of real Bangladeshi plates for OCR.

Detection mAP50 (test): 92%

Detection mAP50-95 (test): 88%

OCR Character Accuracy: 89%

OCR Character Error Rate: 11%

OCR Full Plate Exact Match: 72%

Interpretation: The high mAP50 proves the detector reliably crops the plate. The OCR accuracy is strong, though full-plate exact matches are lower because a single misread character fails the strict metric.

🐳 Docker Prediction App

The entire prediction pipeline (Detector + OCR + UI) is containerized using Docker.

Build and Run Commands:

docker-compose up --build


Once the containers are running, access the web app at http://localhost:7861.

Demo Screenshots

Instructions for use: Upload a vehicle photograph to the UI, click "Submit", and the app will draw a bounding box around the plate while extracting the Bengali/English text on the right panel.


(Note: If you saved multiple demo screenshots, you can add them here as screenshots/demo_output_2.png, etc.)

⚠️ Limitations & Future Work

Limitations:

Domain Gap: The detector was trained on a general dataset, not exclusively Bangladeshi plates.

OCR Fonts: The EasyOCR model utilizes pretrained weights that are not explicitly fine-tuned for the unique typography found on older Bangladeshi plates.

Class Imbalance: The real-world vehicle dataset is heavily skewed toward private cars compared to CNGs or bikes.

Future Improvements:

Collect and manually annotate a dedicated dataset of localized Bangladeshi license plates to close the domain gap.

Fine-tune the EasyOCR recognition model directly on Bengali license plate fonts.

Implement API endpoints (e.g., FastAPI) to separate the frontend from the backend inference logic.

📄 Final Report

A comprehensive academic report detailing the dataset preparation, deep learning methodology, error analysis, and architectural choices is included in the repository.