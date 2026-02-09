import logging
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import datasets, transforms
from tqdm import tqdm
from pathlib import Path


class ClassifierTrainer():
    """
    Some functions in this class are inspired by Kaggle examples
    and have been adapted for this project.
    """

    def __init__(self, model, train_dataloader, val_dataloader, optimizer, loss_fn, device, logger):
        self.model = model                       #  torch.nn.Module,
        self.train_dataloader = train_dataloader #  torch.utils.data.DataLoader
        self.val_dataloader = val_dataloader     #  torch.utils.data.DataLoader
        self.optimizer = optimizer               #  torch.optim.Optimizer
        self.loss_fn = loss_fn                   #  torch.nn.Module
        self.device = device                     #  string
        self.logger = logger                     #  logging
        
        
        
    def train_step(self):
        # Put model in train mode
        self.model.train()
        
        # Setup train loss and train accuracy values
        train_loss, train_acc = 0, 0
        
        # Loop through data loader data batches
        for batch, (X, y) in enumerate(self.train_dataloader):
            # Send data to target device
            X, y = X.to(self.device), y.to(self.device)
            
            # 1. Forward pass
            y_pred = self.model(X)

            # 2. Calculate  and accumulate loss
            loss = self.loss_fn(y_pred, y)
            train_loss += loss.item() 

            # 3. Optimizer zero grad
            self.optimizer.zero_grad()

            # 4. Loss backward
            loss.backward()

            # 5. Optimizer step
            self.optimizer.step()

            # Calculate and accumulate accuracy metric across all batches
            y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
            train_acc += (y_pred_class == y).sum().item()/len(y_pred)

        # Adjust metrics to get average loss and accuracy per batch 
        train_loss = train_loss / len(self.train_dataloader)
        train_acc = train_acc / len(self.train_dataloader)
        return train_loss, train_acc

    def test_step(self, save_errors=False):
        # save errors images in repo
        error_dir = Path("output/errors")
        error_dir.mkdir(exist_ok=True)

        # Put model in eval mode
        self.model.eval() 

        # Setup test loss and test accuracy values
        test_loss, test_acc = 0, 0
        
        # Turn on inference context manager
        with torch.inference_mode():
            # Loop through DataLoader batches
            for X, y in self.val_dataloader:
                # Send data to target device
                X, y = X.to(self.device), y.to(self.device)

                # 1. Forward pass
                test_pred_logits = self.model(X)

                # 2. Calculate and accumulate loss
                loss = self.loss_fn(test_pred_logits, y)
                test_loss += loss.item()

                # Calculate and accumulate accuracy
                test_pred_labels = test_pred_logits.argmax(dim=1)
                test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))

                if save_errors:
                    idx_to_class = {v: k for k, v in self.val_dataloader.dataset.class_to_idx.items()}
                    for i in range(len(y)):
                        if test_pred_labels[i] != y[i]:
                            true_label = idx_to_class[y[i].item()]
                            pred_label = idx_to_class[test_pred_labels[i].item()]
                            save_image(
                                X[i].cpu(),
                                error_dir / f"true_{true_label}_pred_{pred_label}.png"
                            )


        # Adjust metrics to get average loss and accuracy per batch 
        test_loss = test_loss / len(self.val_dataloader)
        test_acc = test_acc / len(self.val_dataloader)
        return test_loss, test_acc

    # 1. Take in various parameters required for training and test steps
    def train(self, epochs,
            threshold, 
            save_errors=False):

        # 2. Create empty results dictionary
        results = {"train_loss": [],
            "train_acc": [],
            "test_loss": [],
            "test_acc": []
        }
        best_acc = 0.0
        
        # 3. Loop through training and testing steps for a number of epochs
        for epoch in tqdm(range(epochs)):
            train_loss, train_acc = self.train_step()
            test_loss, test_acc = self.test_step(save_errors)

            # 4. Print out what's happening
            print(
                f"Epoch: {epoch+1} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} | "
                f"test_loss: {test_loss:.4f} | "
                f"test_acc: {test_acc:.4f}"
            )

            # 5. Update results dictionary
            results["train_loss"].append(train_loss)
            results["train_acc"].append(train_acc)
            results["test_loss"].append(test_loss)
            results["test_acc"].append(test_acc)

            # keep best accuracy over threshold
            if test_acc > best_acc:
                best_acc = test_acc
                if test_acc >= threshold:
                    save_path = Path("output/model")
                    save_path.mkdir(parents=True, exist_ok=True)
                    torch.save(self.model.state_dict(), f"{save_path}/material_recognizer_model_{best_acc}.pth")
                    self.logger.info(f"Model save with accuracy {best_acc}")

        self.logger.info(f"Best accuracy {best_acc}")
        # 6. Return the filled results at the end of the epochs
        return results
