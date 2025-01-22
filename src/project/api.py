from contextlib import asynccontextmanager
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from src.project.model import MyAwesomeModel

from torchvision import transforms


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and clean up model on startup and shutdown."""
    global model, device

    print("Loading model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "../")
    model_path = os.path.join(main_path, "models/model.pth")
    model = MyAwesomeModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    yield

    print("Cleaning up")
    del model, device


app = FastAPI(lifespan=lifespan)


@app.post("/caption/")
async def caption(data: UploadFile = File(...)):
    """Generate a caption for an image."""
    
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Resize to match model input size
        transforms.ToTensor(),          # Convert PIL Image to Tensor
    ])

    # Evaluate the model
    with torch.no_grad():
        # TODO Test image parameters (size, color)
        img = Image.open(data.file).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)  # Add batch dimension and move to DEVICE

        y_pred = model(img_tensor)
        # y_pred.argmax(dim=1)
    
    return y_pred.argmax(dim=1)


@app.get("/index/")
async def hello_world():
    return "Hello API"