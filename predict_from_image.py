import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import sys
import matplotlib.pyplot as plt

# Define the model architecture (same as in your trained model)
class FashionMNISTModel(nn.Module):
    def __init__(self):
        super(FashionMNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(256, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        
    def forward(self, x):
        x = nn.functional.max_pool2d(nn.functional.relu(self.conv1(x)), 2)
        x = nn.functional.max_pool2d(nn.functional.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Class labels for Fashion MNIST
fashion_mnist_labels = {
    0: "T-shirt/top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}

def load_model(model_path):
    # Initialize model architecture
    model = FashionMNISTModel()
    # Load the state dictionary
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    # Apply weights to model
    model.load_state_dict(state_dict)
    # Set to evaluation mode
    model.eval()
    return model

def preprocess_image(image_path):
    """Preprocess an input image for the model."""
    # Open the image
    try:
        img = Image.open(image_path).convert('L')  # Convert to grayscale
    except Exception as e:
        print(f"Error opening image: {e}")
        return None
    
    # Show original image
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(img, cmap='gray')
    
    # Resize to 28x28
    img_resized = img.resize((28, 28))
    
    # Show resized image
    plt.subplot(1, 2, 2)
    plt.title("Resized to 28x28")
    plt.imshow(img_resized, cmap='gray')
    plt.tight_layout()
    plt.savefig('processed_image.png')
    plt.close()
    
    print(f"Processed image saved as 'processed_image.png'")
    
    # Convert to numpy array and normalize
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    
    # Add channel dimension and convert to tensor
    img_tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)  # [1, 1, 28, 28]
    
    return img_tensor

def predict_fashion_item(model, image_tensor):
    """Make a prediction for a fashion item."""
    with torch.no_grad():
        output = model(image_tensor)
    
    # Get predicted class
    _, predicted = torch.max(output, 1)
    class_idx = predicted.item()
    
    # Get class probabilities
    probabilities = torch.nn.functional.softmax(output, dim=1)[0]
    
    # Create visualization of the probabilities
    plt.figure(figsize=(10, 6))
    
    classes = list(fashion_mnist_labels.values())
    probs = [prob.item() for prob in probabilities]
    
    # Sort by probability
    sorted_idx = np.argsort(probs)[::-1]  # Descending order
    classes = [classes[i] for i in sorted_idx]
    probs = [probs[i] for i in sorted_idx]
    
    plt.barh(classes, probs)
    plt.xlabel('Probability')
    plt.title('Class Probabilities')
    plt.tight_layout()
    plt.savefig('prediction_results.png')
    plt.close()
    
    print(f"Prediction visualization saved as 'prediction_results.png'")
    
    return {
        'class_index': class_idx,
        'class_name': fashion_mnist_labels[class_idx],
        'confidence': probabilities[class_idx].item(),
        'probabilities': {fashion_mnist_labels[i]: prob.item() for i, prob in enumerate(probabilities)}
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_from_image.py <image_path>")
        print("Example: python predict_from_image.py my_tshirt.jpg")
        return
        
    image_path = sys.argv[1]
    model_path = "model.net"
    
    # Load the model
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    print("Model loaded successfully!")
    
    # Preprocess the image
    print(f"Processing image from {image_path}...")
    image_tensor = preprocess_image(image_path)
    
    if image_tensor is None:
        print("Failed to process the image.")
        return
    
    # Make prediction
    print("Making prediction...")
    prediction = predict_fashion_item(model, image_tensor)
    
    # Print results
    print("\nPrediction Results:")
    print(f"Predicted class: {prediction['class_name']} (index: {prediction['class_index']})")
    print(f"Confidence: {prediction['confidence']*100:.2f}%")
    
    print("\nTop 3 Class Probabilities:")
    for class_name, prob in sorted(prediction['probabilities'].items(), key=lambda x: x[1], reverse=True)[:3]:
        print(f"{class_name}: {prob*100:.2f}%")

if __name__ == "__main__":
    main()