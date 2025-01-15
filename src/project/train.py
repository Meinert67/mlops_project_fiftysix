import matplotlib.pyplot as plt
import torch
import typer
from data import preprocess
from model import MyAwesomeModel
import os
from loguru import logger
import wandb

# Select the device for training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 5) -> None:
    """Train a model on CIFAR-10."""
    logger.info("Training day and night")
    logger.info(f"{lr=}, {batch_size=}, {epochs=}")
    
    # defining weight and bias (name for runs can be added)
    wandb.init(
        project="cifar-10",
        config={"lr": lr, 
                "batch_size": batch_size, 
                "epochs": epochs, 
                "optimizer": "adam", 
                "dropout_rate": 0.5},
    )
    # Initialize the model and move it to the selected device
    model = MyAwesomeModel().to(DEVICE)

    data = preprocess()
    train_set = data.train_set
    test_set = data.test_set

    # Create data loaders for training
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # Loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # For tracking statistics
    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)

            # Forward pass
            y_pred = model(img)
            loss = loss_fn(y_pred, target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track loss and accuracy
            epoch_loss += loss.item()
            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            epoch_accuracy += accuracy
            
            wandb.log({"train_loss": loss.item(), "train_accuracy": accuracy})
            if i % 100 == 0:
                logger.info(f"Epoch {epoch+1}, Iter {i}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

        # Average loss and accuracy for the epoch
        epoch_loss /= len(train_dataloader)
        epoch_accuracy /= len(train_dataloader)
        statistics["train_loss"].append(epoch_loss)
        statistics["train_accuracy"].append(epoch_accuracy)
        

        logger.info(f"Epoch {epoch+1}: Average Loss: {epoch_loss:.4f}, Average Accuracy: {epoch_accuracy:.4f}")

    logger.info("Training complete")

    # Save the model
    main_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '../')
    #main_path = os.path.dirname(os.path.dirname(__file__))
    logger.info(main_path)
    torch.save(model.state_dict(), os.path.join(main_path, "models/model.pth"))
    # uploads model to wandb
    wandb.save(os.path.join(main_path, "models/model.pth"))

    # Plot and save training statistics
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")

    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")

    fig.savefig(os.path.join(main_path, "reports/figures/training_statistics.png"))
    logger.info("Training statistics saved as a plot.")

if __name__ == "__main__":
    typer.run(train)