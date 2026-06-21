Omnivision_Classifier — Transfer Learning Image Classifier
ML & Programming for AI Project | Spring 2026
Riphah International University, Lahore Campus

Project Description
Omnivision_Classifier is a deep learning-based multi-class image classification system that classifies images into 30 categories spanning animals, vehicles, fruits, flowers, and structures. The project implements and compares four state-of-the-art transfer learning architectures (EfficientNetV2B3, ResNet50V2, MobileNetV3Large, and ConvNeXtTiny) with comprehensive experiment tracking, visualization, and an interactive web interface. The entire project is designed to run on Google Colab with GPU acceleration.

30 Classes — 11 Semantic Groups with Intentional Intra-Group Overlap
Group	Classes	Overlap Reason
🐾 Big Cats	lion, tiger, leopard	4-legged predator body
🐾 Large Animals	elephant, camel, bear	large distinct shapes
🐾 Small Rodents	rabbit, mouse, squirrel	small furry round body
🐾 Aquatic	dolphin, whale, shark	streamlined marine body
🐾 Reptiles	snake, lizard, crocodile	scales + crawling
🐾 Road Vehicles	bus, pickup_truck, train	large ground transport
🐾 Two-Wheelers	bicycle, motorcycle	two wheels + rider
🍎 Fruits	apple, orange, pear	round colourful fruit
🌸 Flowers	rose, tulip, sunflower	colourful petals
🏢 Structures	skyscraper, house	building context
🚀 Distinct	rocket, mushroom	high-accuracy anchors
Project Structure (Google Colab)
text
/content/Omnivision_Classifier/
├── .git/
├── .gitignore
├── README.md
├── app.py                         # Gradio web application
├── mlflow_ui.py                   # MLflow UI launcher with ngrok tunnel
├── models/
│   ├── best_model.keras           # Best performing model (ConvNeXtTiny)
│   └── class_names.json           # 30 class labels mapping
├── src/
│   ├── __init__.py
│   ├── config.py                  # Configuration settings
│   ├── model_utils.py             # Model loading and prediction utilities
│   └── preprocessing.py           # Custom preprocessing functions
├── notebooks/
│   └── omnivision-classifier.ipynb  # Main notebook with full pipeline
├── results/
│   ├── confusion_matrix.png       # Confusion matrix visualization
│   ├── model_comparison.csv       # Model performance comparison
│   ├── roc_curves.png            # ROC curves (best and worst classes)
│   ├── sample_predictions.png     # Sample predictions with confidence scores
│   └── training_curves.png        # Training and validation curves
├── mlruns/                        # MLflow experiment tracking data
└── requirements.txt               # Pinned dependencies
Google Colab Setup Instructions
1. Open Google Colab
bash
https://colab.research.google.com/
2. Set Runtime to GPU
Go to Runtime → Change runtime type

Select GPU as Hardware accelerator

3. Mount Google Drive (Optional, for backup)
python
from google.colab import drive
drive.mount('/content/drive')
4. Create Project Structure
python
import os
os.makedirs("/content/Omnivision_Classifier/src", exist_ok=True)
os.makedirs("/content/Omnivision_Classifier/models", exist_ok=True)
os.makedirs("/content/Omnivision_Classifier/results", exist_ok=True)
os.makedirs("/content/Omnivision_Classifier/mlruns", exist_ok=True)
5. Install Dependencies
python
!pip install -q gradio pyngrok mlflow tensorflow huggingface_hub datasets
6. Run the Complete Pipeline
python
%run /content/Omnivision_Classifier/notebooks/omnivision-classifier.ipynb
How to Run (Complete Colab Workflow)
Step 1: Load and Preprocess Data
python
from datasets import load_dataset
import numpy as np
from sklearn.model_selection import train_test_split

# Load CIFAR-100 dataset
dataset = load_dataset("uof-tcs/cifar100")

# Filter 30 classes and map labels
# Full code in notebook
Step 2: Build and Train Models
python
# Four models trained with transfer learning
# - EfficientNetV2B3
# - ResNet50V2
# - MobileNetV3Large
# - ConvNeXtTiny

# Each model uses:
# - Pre-trained ImageNet weights
# - Custom classification head
# - Data augmentation pipeline
# - Early stopping and learning rate reduction
Step 3: Evaluate and Visualize
python
# Training curves, confusion matrix, ROC curves
# Model comparison table
# Sample predictions with confidence scores
Step 4: Save Best Model
python
best_model.save('/content/Omnivision_Classifier/models/best_model.keras')
with open('/content/Omnivision_Classifier/models/class_names.json', 'w') as f:
    json.dump(SELECTED_CLASSES, f)
Step 5: Launch Gradio Web App
python
%run /content/Omnivision_Classifier/app.py
This will generate a public shareable link for live classification.

Step 6: Launch MLflow UI with ngrok
python
# Start MLflow server and create ngrok tunnel
import subprocess, time, urllib.request
from pyngrok import ngrok

ngrok.set_auth_token("YOUR_NGROK_TOKEN")
proc = subprocess.Popen(
    ["python", "-m", "mlflow", "ui",
     "--host", "0.0.0.0",
     "--port", "5001",
     "--backend-store-uri", "file:///content/Omnivision_Classifier/mlruns"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

tunnel = ngrok.connect(5001)
print(f"MLflow UI: {tunnel.public_url}")
Model Architectures
The project implements and compares four transfer learning architectures:

Model	Preprocessing Function	Key Features
EfficientNetV2B3	preprocess_efficientnetv2b3	Efficient compound scaling, balanced speed/accuracy
ResNet50V2	preprocess_resnet50v2	Improved residual connections, stable training
MobileNetV3Large	preprocess_mobilenetv3large	Mobile-optimized, efficient inference
ConvNeXtTiny	preprocess_convnexttiny	Modern CNN design, best performance
Model Architecture Details
All models follow the same transfer learning pattern:

Base model (pre-trained on ImageNet) with frozen weights

Global Average Pooling layer

Dropout layer (0.35)

Dense layer (512 units, ReLU activation)

Dropout layer (0.35)

Output layer (30 units, Softmax activation)

Results Summary
Model	Accuracy	Precision	Recall	F1 Score
ConvNeXtTiny	84.77%	84.89%	84.77%	84.74%
EfficientNetV2B3	83.97%	84.35%	83.97%	83.96%
MobileNetV3Large	80.53%	80.67%	80.53%	80.35%
ResNet50V2	75.20%	75.43%	75.20%	75.13%
Best Model: ConvNeXtTiny (84.77% test accuracy)

Configuration (src/config.py)
python
BASE_DIR        = "/content/Omnivision_Classifier"
MODEL_PATH      = f"{BASE_DIR}/models/best_model.keras"
CLASS_JSON_PATH = f"{BASE_DIR}/models/class_names.json"
MLFLOW_DIR      = f"{BASE_DIR}/mlruns"
IMG_SIZE        = 128  # Input image size
Features
📊 Comprehensive Visualization
Training Curves: Accuracy and loss for all 4 models

Confusion Matrix: Per-class performance for best model

ROC Curves: ROC and AUC for best and worst classes

Sample Predictions: Visual examples with confidence scores

Model Comparison: Side-by-side performance metrics

🧪 Experiment Tracking (MLflow)
Hyperparameter Logging: All model parameters

Metric Logging: Accuracy, loss, F1 score per epoch

Artifact Storage: Model weights, visualizations, CSV files

Model Versioning: Compare different runs

🌐 Interactive Web Interface (Gradio)
Live Classification: Upload images for prediction

Top-5 Predictions: Display with confidence scores

Performance Dashboard: Visualize metrics and graphs

Shareable Link: Public URL for anyone to use

🔄 Colab Integration
GPU Acceleration: Fast training with Colab GPU

Persistent Storage: Save models to Drive or working directory

One-Click Setup: All dependencies installed automatically

ngrok Integration: Public URLs for Gradio and MLflow

Sample Predictions
True Label	Predicted Label	Confidence
rose	rose	82.9%
skyscraper	skyscraper	100.0%
tulip	tulip	89.6%
house	house	77.5%
bicycle	bicycle	99.7%
squirrel	squirrel	99.3%
motorcycle	motorcycle	99.6%
lizard	lizard	71.0%
orange	orange	99.7%
pickup_truck	pickup_truck	86.0%
AI Usage Statement
AI tools (Claude by Anthropic and ChatGPT) were used for code structure suggestions, debugging assistance, and documentation formatting. All model architecture decisions, hyperparameter tuning, evaluation methodology, and analysis were completed independently. The project demonstrates transfer learning implementation on CIFAR-100 with comprehensive experiment tracking and deployment.

Contributors
Abdullah - Riphah International University, Lahore Campus

Mustafa - Riphah International University, Lahore Campus

License
This project is part of the ML & Programming for AI course at Riphah International University, Lahore Campus.

Additional Resources
Google Colab Links
Project Notebook: Run on Colab with GPU

Gradio App: Public URL for live classification

MLflow UI: Public URL for experiment tracking

GitHub Repository
bash
https://github.com/chabdullahtariq0566-design/Omnivision_Classifier
Hugging Face Model Hub
Model uploaded to Hugging Face for easy access.

Requirements
text
tensorflow>=2.13.0
gradio>=4.0.0
mlflow>=2.5.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
datasets>=2.14.0
huggingface_hub>=0.16.0
pyngrok>=6.0.0
