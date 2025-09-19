"""
Implement a CNN classifier for image classification using Pytorch
@author: Thanh Le
"""
import torch
print(torch.__version__)
print(torch.cuda.is_available())
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from torchmetrics.functional import accuracy, precision, recall, f1_score
from torchvision.transforms import ToTensor, Resize
from torch.utils.data import random_split
import torch.nn as nn
from timeit import default_timer as timer
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

class FlowerClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Please design your CNN and insert your code here. 


        # Convolutional layers

          # 1st block: Conv2d + ReLU + MaxPool2d
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), #ouput (16, 256, 256)
            nn.ReLU(),
            nn.MaxPool2d(2, 2), #output (16, 128, 128)

        #2nd block: Conv2d + ReLU + MaxPool2d

            nn.Conv2d(16, 32, kernel_size=3, padding=1),   #output (32, 128, 128)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Output: (32, 64, 64)
        )

        # Flatten layer — prepares for linear layer
        self.flatten = nn.Flatten()

        # Fully connected (linear) layers
        self.fc_block = nn.Sequential(
            nn.Linear(32 * 64 * 64, 128),  # Input dims depend on image size
            nn.ReLU(),
            nn.Linear(128, 10)  # 5 output classes
        )
        # Refer to examples from: https://github.com/pytorch/examples/blob/main/mnist/main.py


    def forward(self, x):
        x = self.conv_block(x)
        x = self.flatten(x)
        x = self.fc_block(x)
        return x
        
        


def train_model():
    """
    Train the model over a single epoch
    :return: training loss and training accuracy
    """
    train_loss = 0.0
    train_acc = 0.0
    model.train()

    for (img, label) in tqdm(train_loader, ncols=80, desc='Training'):
        # Get a batch
        img, label = img.to(device, dtype=torch.float), label.to(device, dtype=torch.long)

        # Set the gradients to zero before starting backpropagation
        optimizer.zero_grad()

        # Perform a feed-forward pass
        logits = model(img)

        # Compute the batch loss
        loss = loss_fn(logits, label)

        # Compute gradient of the loss fn w.r.t the trainable weights
        loss.backward()

        # Update the trainable weights
        optimizer.step()

        # Accumulate the batch loss
        train_loss += loss.item()

        # Get the predictions to calculate the accuracy for every iteration. Remember to accumulate the accuracy
        prediction = logits.argmax(axis=1)
        train_acc += accuracy(prediction, label, task='multiclass', average='macro', num_classes=10).item()

    return train_loss / len(train_loader), train_acc / len(train_loader)


def validate_model():
    """
    Validate the model over a single epoch
    :return: validation loss and validation accuracy
    """
    model.eval()
    valid_loss = 0.0
    val_acc = 0.0

    with torch.no_grad():
        for (img, label) in tqdm(val_loader, ncols=80, desc='Valid'):
            # Get a batch
            img, label = img.to(device, dtype=torch.float), label.to(device, dtype=torch.long)

            # Perform a feed-forward pass
            logits = model(img)

            # Compute the batch loss
            loss = loss_fn(logits, label)

            # Accumulate the batch loss
            valid_loss += loss.item()

            # Get the predictions to calculate the accuracy for every iteration. Remember to accumulate the accuracy
            prediction = logits.argmax(axis=1)
            val_acc += accuracy(prediction, label, task='multiclass', average='macro', num_classes=10).item()

    return valid_loss / len(val_loader), val_acc / len(val_loader)


def test_model():
    """
    Test the trained model
    :return: classification metrics on the test set
    """
    model.eval()
    y_true=[]
    y_pred=[]
    with torch.no_grad():
        
        for img, label in tqdm(test_loader, ncols=80, desc='Testing'):
            img = img.to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.long)
            logits = model(img)
            predictions = logits.argmax(dim=1) 
            

            y_true.extend(label.cpu())
            y_pred.extend(predictions.cpu())

    #compute metrics 
    acc = accuracy(torch.tensor(y_pred), torch.tensor(y_true), task='multiclass', average='macro', num_classes=10)
    prec = precision(torch.tensor(y_pred), torch.tensor(y_true), task='multiclass', average='macro', num_classes=10)
    rec = recall(torch.tensor(y_pred), torch.tensor(y_true), task='multiclass', average='macro', num_classes=10)
    f1 = f1_score(torch.tensor(y_pred), torch.tensor(y_true), task='multiclass', average='macro', num_classes=10)
    print("\nFinal Evaluation on Test Set:")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")






    # Please insert your code here to complete this function.
    # You can re-use the large code from the function 'validate_model()'. 
    # Note that, the only change needed is that the loss calculation is not required, and this function should return all classification metrics (i.e., f1 score, accuracy, recall, and precision)



def plot_conf_matrix():
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for img, label in test_loader:
            img = img.to(device, dtype=torch.float)
            label = label.to(device, dtype=torch.long)

            logits = model(img)
            predictions = logits.argmax(dim=1)

            y_true.extend(label.cpu())
            y_pred.extend(predictions.cpu())

    # Compute the confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    #class names for display
    class_names = ['Bulbasaur', 'Meowth', 'Mew', 'Pidgeot', 'Pikachu', 'Snorlax', 'Squirtle', 'Venusaur', 'Wartortle', 'Zubat']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names )

    #plotting
 
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()



if __name__ == "__main__":

    # STEP 1. Load the dataset
    transform = transforms.Compose([Resize((256, 256)), ToTensor()])
    data = ImageFolder(root='PokemonData', transform=transform)
    train_set, test_set, val_set = random_split(data, [0.6, 0.2, 0.2])

    # STEP 2. Create data loaders
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=True)

    # STEP 3. Create a CNN model
    model= FlowerClassificationModel().to(device)



    
    

    # STEP 4. Specify loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)



    # STEP 5. Train the model with 30 epochs
    best_val_acc= 0.0
    for epoch in range(30):
        print(f"\nEpoch {epoch+1}/30") 
        # STEP 5.1. Train the model over a single epoch
        train_loss, train_acc = train_model()
        print(f"Train Loss: {train_loss:.4f}, Train Acc:{train_acc:.4f}")
        
    


        # STEP 5.2. Validate the model after training
        val_loss, val_acc= validate_model()
        print(f"Val Loss: {val_loss:.4f}, Val Acc :{val_acc:.4f}")


        # STEP 5.3. Save the model if the validation accuracy is increasing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("Model saved!")

        



    # STEP 6. Evaluate the CNN on the test dataset
    model.load_state_dict(torch.load("best_model.pth"))  # Load the best saved model
    test_model()
    
    
    # STEP 7. Plot the confusion matrix
  
    plot_conf_matrix()
