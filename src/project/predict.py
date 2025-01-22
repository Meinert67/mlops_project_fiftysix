import torch
from data import preprocess
from model import MyAwesomeModel
import os
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms

# Select the device for evaluation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


# Define the evaluation function
def predict(model_path: Path, image_path: Path):
    """Predict random image based on trained model"""

    # Load the saved model
    model = MyAwesomeModel()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # Define transformations for the input image
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to match model input size
        transforms.ToTensor(),          # Convert PIL Image to Tensor
    ])

    pred_list = []
    # Evaluate the model
    with torch.no_grad():

        # TODO Test image parameters (size, color)
        img = Image.open(image_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)  # Add batch dimension and move to DEVICE
            
        # print(img_tensor.shape)
        y_pred = model(img_tensor)
        pred_list.append(y_pred.argmax(dim=1))
    
    return pred_list

  


if __name__ == "__main__":
    # Define the path to the saved model and evaluation parameters
    main_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "../")
    model_path = os.path.join(main_path, "models/model.pth")
    image_path = os.path.join(main_path, "data/image_test/0000.jpg")
    print(predict(model_path,image_path))

