# Plant Disease Detection

A deep learning model to detect plant diseases using TensorFlow and MobileNetV2.

## Project Structure
```
PlantDisease/
├── images/             # Dataset directory
├── train_model.py      # Training script
├── app.py             # Gradio web interface
└── requirements.txt   # Python dependencies
```

## Setup and Installation

1. Clone the repository
```bash
git clone <your-repository-url>
cd PlantDisease
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Train the model
```bash
python train_model.py
```

4. Run the web interface
```bash
python app.py
```

## Model Details
- Base model: MobileNetV2
- Input size: 128x128
- Classes: Healthy, TobacoStrick, and various disease combinations

## Dataset
The dataset contains images of plants with different conditions:
- Healthy plants
- Tobacco Strick disease
- Various combinations of diseases
