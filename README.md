# Plant Disease Detection

A comprehensive plant disease detection system using multiple deep learning architectures. This project implements several state-of-the-art models for accurate plant disease classification.

## Available Models

1. **ResNet50**
   - Deep residual learning
   - Input size: 224x224
   - Excellent feature extraction
   - Good for complex patterns

2. **GoogLeNet (Inception V3)**
   - Efficient inception modules
   - Input size: 299x299
   - Balanced computational cost
   - Strong performance on varied scales

3. **AlexNet**
   - Classic CNN architecture
   - Input size: 227x227
   - 5 convolutional layers
   - Fast training and inference

4. **DenseNet121**
   - Dense connectivity
   - Input size: 224x224
   - Efficient feature reuse
   - Reduced parameter count

## Project Structure
```
PlantDisease/
├── images/                     # Dataset directory
├── train_model_resnet.py      # ResNet50 training
├── train_model_googlenet.py   # GoogLeNet training
├── train_model_alexnet.py     # AlexNet training
├── train_model_densenet.py    # DenseNet training
├── app.py                     # Multi-model web interface
└── requirements.txt           # Python dependencies
```

## Setup and Installation

1. Clone the repository
```bash
git clone https://github.com/MISTERNEGATIVE21/PlantDisease.git
cd PlantDisease
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Train the models (you can train any or all models)
```bash
# Train ResNet50
python train_model_resnet.py

# Train GoogLeNet
python train_model_googlenet.py

# Train AlexNet
python train_model_alexnet.py

# Train DenseNet
python train_model_densenet.py
```

4. Run the web interface
```bash
python app.py
```

## Model Features

Common features across all implementations:
- Data augmentation (rotation, flip, zoom, etc.)
- Batch normalization
- Dropout for regularization
- Early stopping
- Learning rate scheduling
- Top-k accuracy metrics

## Web Interface

The Gradio web interface supports:
- Multiple model selection
- Top 3 predictions with confidence scores
- Easy-to-use image upload
- Real-time inference

## Dataset
The dataset contains images of plants with various conditions:
- Healthy plants
- Tobacco Strick disease
- Cercospora
- Powdery Mildew
- Target Leaf Spot
- Various disease combinations

Each model is trained on high-quality images with proper augmentation to ensure robust performance across different conditions.

## Performance Comparison

Each model has its strengths:
- ResNet50: Best overall accuracy
- GoogLeNet: Good balance of speed and accuracy
- AlexNet: Fastest inference
- DenseNet: Best parameter efficiency

## Requirements

- Python 3.7+
- TensorFlow 2.x
- Gradio
- NumPy
- Pillow

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Credits

This repository uses the [Cotton Dataset for Multi-Disease Classification](https://www.kaggle.com/datasets/vaishalibhujadesfdc/cotton-dataset-for-multi-disease-classification) by Vaishali BhujadeSFDC. The dataset contains real field images of cotton plants, collected from [ICAR-CICR, Nagpur](https://cicr.org.in/) and this research centre.

Special thanks to Vaishali BhujadeSFDC and the [ICAR-CICR, Nagpur research centre](https://cicr.org.in/) for providing this valuable resource.
