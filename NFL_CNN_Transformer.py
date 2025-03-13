import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

# Define the CNN + Transformer model with multiple independent output classifiers
class CNN_Transformer_MultiOutput_Model(nn.Module):
    def __init__(self, input_size=256, hidden_dim=512, num_classes_1=10, num_classes_2=5, num_classes_3=3, num_heads=8, num_layers=6, ff_hid_dim=2048):
        super(CNN_Transformer_MultiOutput_Model, self).__init__()
        
        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Transformer encoder layer
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=input_size, 
            nhead=num_heads, 
            dim_feedforward=ff_hid_dim
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_layer, 
            num_layers=num_layers
        )
        
        # Output classifiers (separate for each task)
        self.fc_1 = nn.Linear(input_size, num_classes_1)  # Output 1 (classification task)
        self.fc_2 = nn.Linear(input_size, num_classes_2)  # Output 2 (classification task)
        self.fc_3 = nn.Linear(input_size, num_classes_3)  # Output 3 (multi-label classification)
    
    def forward(self, x):
        # x: (batch_size, seq_len, channels, height, width)
        
        batch_size, seq_len, channels, height, width = x.size()
        
        # Apply CNN to each frame in the sequence
        cnn_out = []
        for i in range(seq_len):
            frame = x[:, i, :, :, :]  # Get the ith frame
            frame_features = self.cnn(frame)  # Apply CNN to the frame
            frame_features = frame_features.view(frame_features.size(0), -1)  # Flatten the output
            cnn_out.append(frame_features)
        
        # Stack CNN outputs to form the sequence of feature vectors
        cnn_out = torch.stack(cnn_out, dim=1)  # (batch_size, seq_len, feature_size)
        
        # Pass through the Transformer encoder
        transformer_out = self.transformer_encoder(cnn_out.transpose(0, 1))  # (seq_len, batch_size, feature_size)
        
        # Use the output of the Transformer for the last time step (this can be adjusted based on your needs)
        last_hidden_state = transformer_out[-1, :, :]  # (batch_size, feature_size)
        
        # Pass through separate fully connected layers for each output task
        out_1 = self.fc_1(last_hidden_state)  # Output 1 (e.g., classification task)
        out_2 = self.fc_2(last_hidden_state)  # Output 2 (e.g., classification task)
        out_3 = self.fc_3(last_hidden_state)  # Output 3 (e.g., multi-label classification)
        
        return out_1, out_2, out_3


# Custom Dataset for sequential image input
class SequentialImageDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None, seq_length=5):
        self.image_dir = image_dir
        self.label_file = label_file
        self.transform = transform
        self.seq_length = seq_length
        
        # Load image paths and labels from label file (assuming CSV format)
        with open(label_file, 'r') as f:
            lines = f.readlines()

        self.image_paths = [line.strip().split(',')[0] for line in lines]
        self.labels_1 = [int(line.strip().split(',')[1]) for line in lines]  # Labels for output 1
        self.labels_2 = [int(line.strip().split(',')[2]) for line in lines]  # Labels for output 2
        self.labels_3 = [list(map(int, line.strip().split(',')[3].split())) for line in lines]  # Multi-label for output 3

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the sequence of images for the given index
        image_sequence = []
        for i in range(self.seq_length):
            img_name = os.path.join(self.image_dir, self.image_paths[idx] + f'_frame{i+1}.jpg')
            image = Image.open(img_name)
            if self.transform:
                image = self.transform(image)
            image_sequence.append(image)
        
        # Stack the images to form a sequence (seq_len, channels, height, width)
        image_sequence = torch.stack(image_sequence, dim=0)  # (seq_len, channels, height, width)
        
        label_1 = self.labels_1[idx]  # Label for output 1 (classification)
        label_2 = self.labels_2[idx]  # Label for output 2 (classification)
        label_3 = torch.tensor(self.labels_3[idx], dtype=torch.float32)  # Multi-label for output 3
        
        return image_sequence, label_1, label_2, label_3


# Example image transformations (resize, normalization, etc.)
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Dataset and DataLoader
image_dir = './data/images'
label_file = './data/labels.csv'

dataset = SequentialImageDataset(image_dir=image_dir, label_file=label_file, transform=transform, seq_length=5)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Model initialization
model = CNN_Transformer_MultiOutput_Model(input_size=256, hidden_dim=512, num_classes_1=10, num_classes_2=5, num_classes_3=3)

# Loss functions for the three outputs
criterion_classification_1 = nn.CrossEntropyLoss()  # For output 1 (classification)
criterion_classification_2 = nn.CrossEntropyLoss()  # For output 2 (classification)
criterion_bce = nn.BCEWithLogitsLoss()  # For output 3 (multi-label classification)

# Optimizers for each output (separate optimizers for each task)
optimizer_1 = optim.Adam(model.fc_1.parameters(), lr=0.001)
optimizer_2 = optim.Adam(model.fc_2.parameters(), lr=0.001)
optimizer_3 = optim.Adam(model.fc_3.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss_1 = 0.0
    running_loss_2 = 0.0
    running_loss_3 = 0.0
    
    for inputs, labels_1, labels_2, labels_3 in train_loader:
        # Zero gradients for each optimizer
        optimizer_1.zero_grad()
        optimizer_2.zero_grad()
        optimizer_3.zero_grad()
        
        # Forward pass
        outputs_1, outputs_2, outputs_3 = model(inputs)
        
        # Compute losses for each output
        loss_1 = criterion_classification_1(outputs_1, labels_1)
        loss_2 = criterion_classification_2(outputs_2, labels_2)
        loss_3 = criterion_bce(outputs_3, labels_3)
        
        # Backpropagate and update weights for each classifier independently
        loss_1.backward()
        optimizer_1.step()
        
        loss_2.backward()
        optimizer_2.step()
        
        loss_3.backward()
        optimizer_3.step()
        
        # Track loss for logging
        running_loss_1 += loss_1.item()
        running_loss_2 += loss_2.item()
        running_loss_3 += loss_3.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss 1: {running_loss_1/len(train_loader):.4f}, Loss 2: {running_loss_2/len(train_loader):.4f}, Loss 3: {running_loss_3/len(train_loader):.4f}")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# Assuming the `CNN_LSTM_MultiOutput_Model` and `SequentialImageDataset` are already defined
# and the model has been trained previously.

def test_model(model, test_loader, criterion_classification_1, criterion_classification_2, criterion_bce, device):
    model.eval()  # Set model to evaluation mode
    running_loss_1 = 0.0
    running_loss_2 = 0.0
    running_loss_3 = 0.0

    all_preds_1 = []
    all_labels_1 = []
    
    all_preds_2 = []
    all_labels_2 = []
    
    all_preds_3 = []
    all_labels_3 = []

    with torch.no_grad():  # No gradient calculation during testing
        for inputs, labels_1, labels_2, labels_3 in test_loader:
            inputs = inputs.to(device)
            labels_1 = labels_1.to(device)
            labels_2 = labels_2.to(device)
            labels_3 = labels_3.to(device)

            # Forward pass
            outputs_1, outputs_2, outputs_3 = model(inputs)

            # Compute losses for each output
            loss_1 = criterion_classification_1(outputs_1, labels_1)
            loss_2 = criterion_classification_2(outputs_2, labels_2)
            loss_3 = criterion_bce(outputs_3, labels_3)

            # Accumulate the loss for logging
            running_loss_1 += loss_1.item()
            running_loss_2 += loss_2.item()
            running_loss_3 += loss_3.item()

            # Collect predictions for classification tasks
            _, predicted_1 = torch.max(outputs_1, 1)  # For classification (Output 1)
            _, predicted_2 = torch.max(outputs_2, 1)  # For classification (Output 2)

            predicted_3 = (outputs_3 > 0.5).float()  # For multi-label classification (Output 3)

            # Store predictions and true labels
            all_preds_1.extend(predicted_1.cpu().numpy())
            all_labels_1.extend(labels_1.cpu().numpy())
            
            all_preds_2.extend(predicted_2.cpu().numpy())
            all_labels_2.extend(labels_2.cpu().numpy())
            
            all_preds_3.extend(predicted_3.cpu().numpy())
            all_labels_3.extend(labels_3.cpu().numpy())

    # Calculate average loss for each output
    avg_loss_1 = running_loss_1 / len(test_loader)
    avg_loss_2 = running_loss_2 / len(test_loader)
    avg_loss_3 = running_loss_3 / len(test_loader)

    # Compute accuracy for classification tasks
    accuracy_1 = accuracy_score(all_labels_1, all_preds_1)
    accuracy_2 = accuracy_score(all_labels_2, all_preds_2)

    # For multi-label classification (Output 3), calculate accuracy per label
    accuracy_3 = accuracy_score(all_labels_3, all_preds_3)

    print(f"Test Losses - Output 1: {avg_loss_1:.4f}, Output 2: {avg_loss_2:.4f}, Output 3: {avg_loss_3:.4f}")
    print(f"Accuracy - Output 1: {accuracy_1:.4f}, Output 2: {accuracy_2:.4f}, Output 3 (multi-label): {accuracy_3:.4f}")

    return avg_loss_1, avg_loss_2, avg_loss_3, accuracy_1, accuracy_2, accuracy_3


# Testing the model on the test dataset
def evaluate_model(model, test_loader, device):
    # Move model to the device (CPU or GPU)
    model.to(device)

    # Define the loss functions for each output classifier
    criterion_classification_1 = nn.CrossEntropyLoss()
    criterion_classification_2 = nn.CrossEntropyLoss()
    criterion_bce = nn.BCEWithLogitsLoss()

    # Call the test function
    avg_loss_1, avg_loss_2, avg_loss_3, accuracy_1, accuracy_2, accuracy_3 = test_model(
        model, test_loader, criterion_classification_1, criterion_classification_2, criterion_bce, device
    )

    return avg_loss_1, avg_loss_2, avg_loss_3, accuracy_1, accuracy_2, accuracy_3


# Assuming `test_loader` is the DataLoader for your test dataset
# `model` is the trained CNN-LSTM model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Example: Assuming you have the test loader already
test_loader = DataLoader(dataset, batch_size=32, shuffle=False)  # Replace 'dataset' with your test dataset

# Evaluate the model
avg_loss_1, avg_loss_2, avg_loss_3, accuracy_1, accuracy_2, accuracy_3 = evaluate_model(model, test_loader, device)

