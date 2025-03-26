# Fashion MNIST Model

This repository contains a trained PyTorch model for classifying fashion items from the [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist).

## Overview

The model is a Convolutional Neural Network (CNN) trained to classify 10 different clothing items:
- T-shirt/top
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle boot

## Setup

1. Create and activate a Python virtual environment:

```bash
python -m venv pytorch_env
source pytorch_env/bin/activate  # On Windows: pytorch_env\Scripts\activate
```

2. Install the required dependencies:

```bash
pip install torch torchvision numpy matplotlib Pillow
```

3. If you encounter NumPy compatibility issues, downgrade NumPy:

```bash
pip install "numpy<2"
```

## Files

- `model.net` - The trained PyTorch model
- `fashion_mnist_app.py` - A standalone app that demonstrates model usage with random data
- `predict_from_image.py` - Tool to predict fashion classes from custom images
- `use_model.py` - Simple example showing how to load and use the model

## Usage

### Basic Example

Run the basic example to verify the model works:

```bash
python fashion_mnist_app.py
```

This will load the model, make a prediction on random data, and show the prediction results.

### Making Predictions with Your Own Images

To classify your own images:

```bash
python predict_from_image.py path/to/your/image.jpg
```

This will:
1. Load and preprocess your image
2. Run the model to classify the clothing item
3. Show prediction probabilities for all classes
4. Save visualizations of the processed image and results

#### Image Requirements
- Images should be grayscale or will be converted to grayscale
- Any size is acceptable (will be resized to 28x28 pixels)
- Best results will be achieved with images similar to the Fashion-MNIST style:
  - Single item centered in the image
  - Plain background
  - Clear contrast between item and background

## Model Architecture

The model uses a LeNet-style architecture:
- Two convolutional layers with max pooling
- Three fully connected layers
- ReLU activation functions

```python
class FashionMNISTModel(nn.Module):
    def __init__(self):
        super(FashionMNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
```

## Extending This Project

Some ideas for extending this project:
- Build a web app with Flask or FastAPI to serve predictions
- Improve the model architecture for better accuracy
- Fine-tune on your own custom clothing dataset
- Add data augmentation to improve robustness

## Acknowledgments

This model was trained using the tutorial from [OVHcloud AI Training](https://support.us.ovhcloud.com/hc/en-us/articles/37640590695443-AI-Training-Tutorial-Train-your-first-ML-model).