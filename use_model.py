import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)  # 1 input channel, 6 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(6, 16, 5)  # 6 input channels, 16 output channels, 5x5 kernel
        self.fc1 = nn.Linear(256, 120)  # Fully connected layer
        self.fc2 = nn.Linear(120, 84)  # Fully connected layer
        self.fc3 = nn.Linear(84, 10)  # Output layer for 10 classes
        
    def forward(self, x):
        # Apply convolutional layers with ReLU activation and max pooling
        x = nn.functional.max_pool2d(nn.functional.relu(self.conv1(x)), 2)
        x = nn.functional.max_pool2d(nn.functional.relu(self.conv2(x)), 2)
        # Flatten the tensor for the fully connected layers
        x = torch.flatten(x, 1)
        # Apply fully connected layers with ReLU activation
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_model(model_path):
    # Initialize model architecture
    model = SimpleModel()
    # Load the state dictionary with CPU mapping
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    # Load weights into model
    model.load_state_dict(state_dict)
    # Set to evaluation mode
    model.eval()
    return model

def predict(model, input_data):
    # Convert input to tensor if it's not already
    if not isinstance(input_data, torch.Tensor):
        input_data = torch.tensor(input_data, dtype=torch.float32)
    
    # Make sure input has batch dimension
    if len(input_data.shape) == 3:  # For images: [channels, height, width]
        input_data = input_data.unsqueeze(0)
    
    # Make prediction
    with torch.no_grad():
        output = model(input_data)
    
    return output

if __name__ == "__main__":
    # Path to your trained model
    model_path = "model.net"
    
    # Load the model
    model = load_model(model_path)
    
    # Example: create dummy input (adjust shape based on your model's expected input)
    # For Fashion MNIST, input would be 1x1x28x28 image (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 1, 28, 28)
    
    # Get prediction
    prediction = predict(model, dummy_input)
    
    # Process the prediction (depends on your model's output format)
    predicted_class = torch.argmax(prediction, dim=1).item()
    print(f"Predicted class: {predicted_class}")