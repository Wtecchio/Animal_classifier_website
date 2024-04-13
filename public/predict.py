import sys
import os
import torch
from torchvision import transforms
from PIL import Image

# Add the parent directory to sys.path to be able to import AnimalClassifier
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import AnimalClassifier

# A list that maps the class indices to actual animal names
class_names = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]



# Define transformations for input images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_model(model_path):
    # Number of classes and other parameters should match the training setup
    model = AnimalClassifier(num_classes=10, feature_size=2048, num_layers=2, num_heads=8, dropout=0.1)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Function to make predictions
def predict(image_path, model):

    # Open the image
    image = Image.open(image_path)
    # Apply transformations
    image = transform(image).unsqueeze(0).to(device)  # Add a batch dimension

    # Make prediction
    model.eval()
    with torch.no_grad():
        output = model(image)  # Assuming model.forward expects just src for inference
        prediction_index = torch.argmax(output, 1).item()
        # Map the predicted index to the corresponding animal name
        predicted_animal = class_names[prediction_index]
        return predicted_animal

if __name__ == "__main__":
    # Get the image path from command line arguments
    if len(sys.argv) < 2:
        print("Error: No image file path provided.")
        sys.exit(1)

    # Determine the device to use: GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Specify the correct path to the model file
    model_path = os.path.join(os.path.dirname(__file__), 'model.pth')
    assert os.path.isfile(model_path), f"Model file not found at {model_path}"

    # Load the entire model, not just the state dict
    model = torch.load(model_path, map_location=device)
    model.to(device)

    # Call the predict function and print the prediction
    image_path = sys.argv[1]
    prediction = predict(image_path, model)
    print(f"The model believes the picture contains a: {prediction}")