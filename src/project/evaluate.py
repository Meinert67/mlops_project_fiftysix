import torch
from data import preprocess
from model import MyAwesomeModel
import os
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import wandb
import hydra
from omegaconf import DictConfig

# Select the device for evaluation
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

api_key = os.getenv("WANDB_API_KEY")

if api_key:
    print("Successfully retrieved API key")
else:
    print("Error: WANDB_API_KEY is not set!")


@hydra.main(version_base="1.1", config_path="../../configs", config_name="config.yaml")
# Define the evaluation function
def evaluate(cfg: DictConfig):
    """Evaluate the trained model on the test dataset."""
    logger.info("Starting evaluation")

    lr = cfg.hyperparameters.learning_rate
    batch_size = cfg.hyperparameters.batch_size
    epochs = cfg.hyperparameters.epochs
    dropout_rate = cfg.hyperparameters.dropout_rate

    logger.info(f"{lr=}, {batch_size=}, {epochs=}")

    main_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "../")
    model_path = os.path.join(main_path, "models/model.pth")

    # Load the saved model
    model = MyAwesomeModel()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    if api_key:
        wandb.init(
            project="cifar-10",
            config={
                "lr": lr,
                "batch_size": batch_size,
                "epochs": epochs,
                "optimizer": "adam",
                "dropout_rate": dropout_rate,
            },
        )

    # Load the test dataset
    data = preprocess()
    test_set = data.test_set

    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Loss function
    loss_fn = torch.nn.CrossEntropyLoss()

    # Initialize metrics
    total_loss = 0.0
    total_accuracy = 0.0
    all_predictions = []
    all_targets = []

    # Evaluate the model
    with torch.no_grad():
        for i, (img, target) in enumerate(test_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            # Forward pass
            y_pred = model(img)
            loss = loss_fn(y_pred, target)

            # Track loss and accuracy
            total_loss += loss.item()
            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            total_accuracy += accuracy

            if api_key:
                wandb.log({"test_loss": loss.item(), "test_accuracy": accuracy})

            # Store predictions and targets for analysis
            all_predictions.extend(y_pred.argmax(dim=1).cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    # Compute average loss and accuracy
    avg_loss = total_loss / len(test_dataloader)
    avg_accuracy = total_accuracy / len(test_dataloader)

    logger.info(f"Evaluation complete. Average Loss: {avg_loss:.4f}, Average Accuracy: {avg_accuracy:.4f}")

    # Save evaluation metrics as a plot
    confusion_matrix = np.zeros((10, 10), dtype=int)  # Assuming 10 classes
    for true_label, pred_label in zip(all_targets, all_predictions):
        confusion_matrix[true_label, pred_label] += 1
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix Heatmap")

    main_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "../")
    plt.savefig(os.path.join(main_path, "reports/figures/evaluation_heatmap.png"))
    logger.info("Confusion matrix heatmap saved as a plot.")
    if api_key:
        wandb.save(os.path.join(main_path, "models/model.pth"))


if __name__ == "__main__":
    evaluate()
